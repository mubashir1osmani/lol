"""Ack reactions, lazy per-guild emoji resolution, and the Discord verdict path."""

import asyncio
import inspect
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from contributor_bot.ack import (
    EYES,
    FALLBACK_TY,
    TY_EMOJI_NAMES,
    acknowledge_message,
    apply_reactions,
    parse_ack_reactions,
    reset_ack_cache,
    resolve_ack_reactions,
    resolve_reactions_for_guild,
    resolve_reactions_for_guild_async,
)
from contributor_bot.config import Config, get_config
from contributor_bot.gateway import _message_payload
from contributor_bot.interactions import handle_interaction
from contributor_bot.review_agent import tool_submit_verdict
from contributor_bot.review_job import handle_channel_message, handle_thread_message
from contributor_bot.store import Store
import contributor_bot.review_job as review_job


def _config(**overrides):
    config = replace(
        Config("token", "123", "456", github_token="gh"),
        pr_review_enabled=True,
        review_channel_ids=("321",),
        review_model="openai/gpt-4o-mini",
        review_bot_logins=("cursor[bot]", "greptile-apps[bot]", "veria-ai[bot]"),
        review_maintainer_role_id="555",
        discord_guild_id="333",
        discord_application_id="444",
        review_cooldown_seconds=600,
    )
    return replace(config, **overrides) if overrides else config


class _FakeGuild:
    def __init__(self, guild_id: int, emojis=(), fetched=(), *, on_fetch=None):
        self.id = guild_id
        self.emojis = tuple(emojis)
        self._fetched = list(fetched)
        self._on_fetch = on_fetch
        self.fetch_calls = 0

    async def fetch_emojis(self):
        self.fetch_calls += 1
        if self._on_fetch is not None:
            self._on_fetch()
        return list(self._fetched)


class AckResolutionTests(unittest.TestCase):
    def tearDown(self):
        reset_ack_cache()

    def test_default_names_and_fallbacks(self):
        self.assertEqual(TY_EMOJI_NAMES, ("ty", "litellm_ty", "thankyou", "thank_you"))
        self.assertEqual(FALLBACK_TY, "🙏")
        api, line = resolve_ack_reactions(None, [])
        self.assertEqual(api, (EYES, FALLBACK_TY))
        self.assertIn("fallback; no guild emoji named ty, litellm_ty, thankyou, thank_you", line)

        guild_ty = [SimpleNamespace(name="TY", id=111111111111111111)]
        api, line = resolve_ack_reactions(None, guild_ty)
        self.assertEqual(api, (EYES, "TY:111111111111111111"))
        self.assertIn("TY:111111111111111111 (ty)", line)

        guild_variant = [("LiteLLM_ty", "222222222222222222")]
        api, line = resolve_ack_reactions(None, guild_variant)
        self.assertEqual(api, (EYES, "LiteLLM_ty:222222222222222222"))
        self.assertIn("LiteLLM_ty:222222222222222222 (litellm_ty)", line)

        guild_thankyou = [("ThankYou", "333333333333333333")]
        api, line = resolve_ack_reactions(None, guild_thankyou)
        self.assertEqual(api, (EYES, "ThankYou:333333333333333333"))
        self.assertIn("ThankYou:333333333333333333 (thankyou)", line)

    def test_empty_cache_falls_back_to_fetch_emojis_and_caches_with_ttl(self):
        now = {"t": 100.0}
        guild = _FakeGuild(
            333,
            emojis=(),
            fetched=[
                SimpleNamespace(name="shipit", id=999999999999999999),
                SimpleNamespace(name="TY", id=444444444444444444),
            ],
        )
        config = _config(ack_reactions=None)
        with self.assertLogs(level="INFO") as logs:
            first = asyncio.run(
                resolve_reactions_for_guild_async(
                    config, guild, ttl_seconds=60.0, clock=lambda: now["t"]
                )
            )
            second = asyncio.run(
                resolve_reactions_for_guild_async(
                    config, guild, ttl_seconds=60.0, clock=lambda: now["t"] + 30.0
                )
            )
        self.assertEqual(first, (EYES, "TY:444444444444444444"))
        self.assertEqual(second, (EYES, "TY:444444444444444444"))
        self.assertEqual(guild.fetch_calls, 1)

        joined = "\n".join(logs.output)
        self.assertEqual(joined.count("Guild 333 emojis"), 1)
        self.assertIn("Guild 333 emojis (2): shipit, TY", joined)
        self.assertIn("TY:444444444444444444 (ty)", joined)

        # Past TTL, fetch_emojis runs again and sees updated emojis.
        guild._fetched = [SimpleNamespace(name="litellm_ty", id=555555555555555555)]
        now["t"] = 161.0
        refreshed = asyncio.run(
            resolve_reactions_for_guild_async(
                config, guild, ttl_seconds=60.0, clock=lambda: now["t"]
            )
        )
        self.assertEqual(refreshed, (EYES, "litellm_ty:555555555555555555"))
        self.assertEqual(guild.fetch_calls, 2)

        # Another guild has an independent cache entry and logs its own emoji list.
        other = _FakeGuild(777, emojis=(), fetched=[SimpleNamespace(name="party", id=888888888888888888)])
        with self.assertLogs(level="INFO") as other_logs:
            fallback = asyncio.run(
                resolve_reactions_for_guild_async(
                    config, other, ttl_seconds=60.0, clock=lambda: now["t"]
                )
            )
        self.assertEqual(fallback, (EYES, FALLBACK_TY))
        self.assertEqual(other.fetch_calls, 1)
        other_joined = "\n".join(other_logs.output)
        self.assertIn("Guild 777 emojis (1): party", other_joined)
        self.assertIn("🙏 (fallback;", other_joined)

    def test_populated_guild_cache_skips_fetch_emojis(self):
        guild = _FakeGuild(
            333,
            emojis=[SimpleNamespace(name="ty", id=111111111111111111)],
            fetched=[SimpleNamespace(name="wrong", id=222222222222222222)],
        )
        with self.assertLogs(level="INFO") as logs:
            chosen = resolve_reactions_for_guild(_config(ack_reactions=None), guild)
        self.assertEqual(chosen, (EYES, "ty:111111111111111111"))
        self.assertEqual(guild.fetch_calls, 0)
        self.assertIn("Guild 333 emojis (1): ty", "\n".join(logs.output))

    def test_ack_reactions_env_resolves_bare_ty_name_with_empty_cache_fallback(self):
        parsed = parse_ack_reactions("👀,ty")
        self.assertEqual(parsed, ("👀", "ty"))
        guild = _FakeGuild(
            333,
            emojis=(),
            fetched=[SimpleNamespace(name="ty", id=444444444444444444)],
        )
        chosen = asyncio.run(resolve_reactions_for_guild_async(_config(ack_reactions=parsed), guild))
        self.assertEqual(chosen, (EYES, "ty:444444444444444444"))
        self.assertEqual(guild.fetch_calls, 1)

        custom, _line = resolve_ack_reactions(("👀", "<:wave:333333333333333333>", "TY"), guild._fetched)
        self.assertEqual(custom, (EYES, "wave:333333333333333333", "ty:444444444444444444"))

    def test_fetch_emojis_bridges_onto_running_gateway_loop(self):
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=2))
        seen_loops = []
        guild = _FakeGuild(
            333,
            emojis=(),
            fetched=[SimpleNamespace(name="ty", id=123123123123123123)],
            on_fetch=lambda: seen_loops.append(asyncio.get_running_loop()),
        )
        reacted = []
        try:
            with mock.patch(
                "contributor_bot.ack.add_reaction",
                side_effect=lambda _c, _m, emoji, _t: reacted.append(emoji),
            ):
                apply_reactions(_config(), "321", "654", guild=guild, loop=loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()

        self.assertEqual(seen_loops, [loop])
        self.assertEqual(reacted, [EYES, "ty:123123123123123123"])

    def test_explicit_list_rejects_a_broken_entry(self):
        with self.assertRaises(ValueError):
            parse_ack_reactions("👀,,ty")
        with self.assertRaises(ValueError):
            parse_ack_reactions("<:not-an-emoji>")

    def test_a_forbidden_reaction_does_not_stop_the_next_one(self):
        guild = _FakeGuild(333, emojis=[SimpleNamespace(name="ty", id=111111111111111111)])
        seen = []

        def react(channel_id, message_id, emoji, token):
            seen.append(emoji)
            if emoji == EYES:
                raise RuntimeError("discord.com returned HTTP 403")

        with mock.patch("contributor_bot.ack.add_reaction", side_effect=react):
            apply_reactions(_config(), "321", "654", guild=guild)
        self.assertEqual(seen, [EYES, "ty:111111111111111111"])

    def test_message_payload_carries_guild_and_loop(self):
        guild = _FakeGuild(333)
        sentinel_loop = object()
        message = SimpleNamespace(
            id=654,
            reference=None,
            content="hello",
            webhook_id=None,
            author=SimpleNamespace(id=111, bot=False),
            channel=SimpleNamespace(id=321),
            guild=guild,
        )
        payload = _message_payload(message, loop=sentinel_loop)
        self.assertEqual(payload["guild_id"], "333")
        self.assertIs(payload["_guild"], guild)
        self.assertIs(payload["_loop"], sentinel_loop)


class DiscordAckFlowTests(unittest.TestCase):
    def setUp(self):
        reset_ack_cache()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(str(Path(self.directory.name) / "roles.db"))
        self.addCleanup(self.store.close)
        self.config = _config(review_classify=False)

    def tearDown(self):
        reset_ack_cache()

    def _join_acks(self):
        for thread in threading.enumerate():
            if thread.name == "ack-react":
                thread.join(timeout=2)

    def test_review_request_reacts_via_fetch_emojis_then_posts_a_verdict_in_a_thread(self):
        discord = SimpleNamespace(reactions=[], posts=[], threads=[])
        guild = _FakeGuild(
            333,
            emojis=(),
            fetched=[SimpleNamespace(name="ty", id=111111111111111111)],
        )

        def add_reaction(channel_id, message_id, emoji, token):
            discord.reactions.append((channel_id, message_id, emoji))

        def post_message(channel_id, token, message, mention_roles=None):
            discord.posts.append((channel_id, message["content"], mention_roles))
            return {"id": str(len(discord.posts)), "channel_id": channel_id}

        def start_thread(channel_id, message_id, token, name):
            discord.threads.append((channel_id, message_id, name))
            return {"id": "900"}

        order = []

        def session(config, store, number, thread_id, mode, history=""):
            for thread in threading.enumerate():
                if thread.name == "ack-react":
                    thread.join(timeout=2)
            order.append(("agent", tuple(discord.reactions)))
            from contributor_bot.review_agent import ReviewContext
            ctx = ReviewContext(config=config, store=store, number=number, thread_id=thread_id)
            tool_submit_verdict(ctx, {"verdict": "looks good", "findings": []})
            from contributor_bot.review_agent import AgentResult
            return AgentResult("posted", verdict_posted=True, ready=True, verdict="looks good", head_sha="ab" * 20)

        message = {
            "id": "654",
            "guild_id": "333",
            "_guild": guild,
            "author": {"id": "111", "bot": False},
            "content": "please review https://github.com/BerriAI/litellm/pull/12",
        }
        with mock.patch("contributor_bot.ack.add_reaction", side_effect=add_reaction), \
                mock.patch("contributor_bot.review_agent.post_message", side_effect=post_message), \
                mock.patch("contributor_bot.review_job.start_public_thread", side_effect=start_thread), \
                mock.patch("contributor_bot.review_job.fetch_recent_messages", return_value=[]), \
                mock.patch("contributor_bot.review_job.run_review_session", side_effect=session), \
                mock.patch("contributor_bot.review_agent.gather_snapshot", return_value=SimpleNamespace(
                    number=12, title="add a timeout", head_sha="ab" * 20, draft=False, merged=False,
                    state="open", mergeable=True, mergeable_state="clean", author_login="octocat",
                    author_id=5, base_sha="cd" * 20, reviews=[], comments=[], threads=[],
                    check_runs=[{"name": "pytest", "status": "completed", "conclusion": "success"}],
                    combined_state=None,
                )):
            handle_channel_message(self.config, self.store, "321", message)
            self._join_acks()
            for thread in threading.enumerate():
                if thread.name == "pr-review":
                    thread.join(timeout=3)
        self.assertEqual(guild.fetch_calls, 1)
        self.assertEqual([item[2] for item in discord.reactions], [
            EYES, "ty:111111111111111111",
        ])
        self.assertEqual(discord.reactions[0][0:2], ("321", "654"))
        self.assertEqual(discord.threads, [("321", "654", "pr 12 review")])
        self.assertTrue(order)
        self.assertEqual(order[0][1], tuple(discord.reactions))
        self.assertTrue(any("needs work" in content or "looks good" in content for _channel, content, _roles in discord.posts))
        self.assertNotIn("run_agent_loop", inspect.getsource(review_job))

    def test_recheck_without_a_message_does_not_react_and_check_again_does(self):
        self.store.save_review_thread(
            repository="BerriAI/litellm", number=12, channel_id="321", message_id="654",
            thread_id="900", created_at="2026-11-26T00:00:00+00:00",
        )
        order = []

        def note(_config, _channel_id, message_id, **_kwargs):
            if message_id:
                order.append("react")

        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job._spawn"):
            response = handle_interaction(self.config, self.store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "900",
                "token": "tokentoken",
                "member": {"user": {"id": "111"}},
                "data": {"name": "recheck", "options": []},
                "channel": {"id": "900", "type": 11, "parent_id": "777"},
            })
        self.assertEqual(response["type"], 5)
        self.assertEqual(order, [])
        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job._spawn"):
            handle_interaction(self.config, self.store, {
                "type": 2,
                "guild_id": "333",
                "application_id": "444",
                "channel_id": "900",
                "token": "tokentoken",
                "member": {"user": {"id": "111"}},
                "message": {"id": "7777"},
                "data": {"name": "recheck", "options": []},
                "channel": {"id": "900", "type": 11, "parent_id": "777"},
            })
        self.assertEqual(order, ["react"])
        with mock.patch("contributor_bot.review_job.acknowledge_message", side_effect=note), \
                mock.patch("contributor_bot.review_job.perform_review", return_value={"status": "posted"}) as review:
            handle_thread_message(self.config, self.store, {
                "id": "8888",
                "channel_id": "900",
                "content": "check again",
                "author": {"id": "111", "bot": False},
            })
        self.assertEqual(order, ["react", "react"])
        review.assert_called_once()
        self.assertEqual(review.call_args.kwargs["source"], "recheck")

    def test_classifier_defaults_to_haiku(self):
        env = {
            "DISCORD_BOT_TOKEN": "token",
            "OSS_CONTRIBUTIONS_CHANNEL_ID": "123",
            "ANNOUNCEMENTS_CHANNEL_ID": "456",
            "PR_REVIEW_ENABLED": "true",
            "DISCORD_GUILD_ID": "333",
            "DISCORD_APPLICATION_ID": "444",
            "DISCORD_PUBLIC_KEY": "ab" * 32,
            "REVIEW_CHANNEL_IDS": "789",
            "REVIEW_MODEL": "anthropic/claude-sonnet-4-5",
        }
        config = get_config(env)
        self.assertEqual(config.review_classifier_model, "anthropic/claude-haiku-4-5")
        self.assertIsNone(config.ack_reactions)
        overridden = get_config(env | {"REVIEW_CLASSIFIER_MODEL": "openai/gpt-4o-mini", "ACK_REACTIONS": "👀,ty"})
        self.assertEqual(overridden.review_classifier_model, "openai/gpt-4o-mini")
        self.assertEqual(overridden.ack_reactions, ("👀", "ty"))


class AcknowledgeStartsBeforeTheAgentTests(unittest.TestCase):
    def test_acknowledge_returns_before_the_reaction_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def react(*args):
            started.set()
            release.wait(timeout=2)

        reset_ack_cache()
        config = _config()
        guild = _FakeGuild(333, emojis=[SimpleNamespace(name="ty", id=111111111111111111)])
        with mock.patch("contributor_bot.ack.add_reaction", side_effect=react):
            acknowledge_message(config, "321", "654", guild=guild)
            self.assertTrue(started.wait(timeout=2))
            release.set()
            for thread in threading.enumerate():
                if thread.name == "ack-react":
                    thread.join(timeout=2)
        reset_ack_cache()
