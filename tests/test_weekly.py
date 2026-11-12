from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contributor_bot import api, poller
from contributor_bot.config import Config, get_config
from contributor_bot.releases import parse_release_index, parse_new_contributors, release_url
from contributor_bot.schedule import first_friday, iso, new_window, parse_time, shift_week
from contributor_bot.state import load_state, save_state
from tests.test_bot import message

FRIDAY = parse_time('2026-11-13T14:00:00Z')
THURSDAY = FRIDAY - timedelta(days=1)
RECENT = '2026-11-10T15:00:00Z'
NOTE = 'https://docs.litellm.ai/release_notes/v1.110.0rc1/v1-110-0-rc-1'
OLD_NOTE = 'https://docs.litellm.ai/release_notes/v1.100.0/v1-100-0'


def pull(n, login='octocat', merged_at=RECENT, user_id=1, bot=False):
    return {'number': n, 'merged': merged_at is not None, 'merged_at': merged_at,
            'user': {'id': user_id, 'login': login, 'type': 'Bot' if bot else 'User'}}


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = Config('token', '123', '456')

    def test_first_start_does_not_immediately_backfill_midweek(self):
        self.assertEqual(first_friday(self.config, THURSDAY), FRIDAY)
        self.assertEqual(first_friday(self.config, FRIDAY + timedelta(hours=2)), FRIDAY)
        self.assertEqual(first_friday(self.config, FRIDAY + timedelta(days=1)), FRIDAY + timedelta(days=7))

    def test_dst_keeps_nine_am_toronto_and_contiguous_boundaries(self):
        spring = parse_time('2026-03-06T14:00:00Z')
        fall = parse_time('2026-10-30T13:00:00Z')
        self.assertEqual(iso(shift_week(self.config, spring)), '2026-03-13T13:00:00+00:00')
        self.assertEqual(iso(shift_week(self.config, fall)), '2026-11-06T14:00:00+00:00')
        self.assertEqual(shift_week(self.config, shift_week(self.config, spring), -1), spring)

    def test_bad_schedule_rejected(self):
        env = {'DISCORD_BOT_TOKEN': 'token', 'OSS_CONTRIBUTIONS_CHANNEL_ID': '123', 'ANNOUNCEMENTS_CHANNEL_ID': '456'}
        for key, value in [('ROUNDUP_TIME', '25:00'), ('ROUNDUP_TIME', '9 AM'), ('ROUNDUP_TIMEZONE', 'Not/AZone')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                get_config(env | {key: value})


class ReleaseParserTests(unittest.TestCase):
    def test_dated_articles_only_and_multiple_versions(self):
        html = '<nav><a href="/release_notes/ancient/ancient">old</a></nav><article><time datetime="2026-11-11T00:06:06.000Z">Nov 11</time><a href="/release_notes/v1.110.0rc1/v1-110-0-rc-1">release</a><a href="/release_notes/v1.110.0rc1/v1-110-0-rc-1">again</a><a href="/release_notes/v1.109.0/v1-109-0">stable</a><a href="https://evil.test/release_notes/a/b">bad</a></article>'
        entries = parse_release_index(html)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[NOTE], parse_time('2026-11-11T00:06:06Z'))

    def test_only_new_contributors_not_full_changelog_or_other_repos(self):
        html = '<h1>Release</h1><a href="https://github.com/BerriAI/litellm/pull/1">other</a><h2 id="new-contributors">New Contributors</h2><ul><li><a href="https://github.com/octocat">@octocat</a><a href="https://github.com/BerriAI/litellm/pull/2">PR</a></li><li><a href="https://github.com/other/repo/pull/3">bad</a></li></ul><h2>Full changelog</h2><a href="https://github.com/BerriAI/litellm/pull/4">other</a>'
        self.assertEqual(parse_new_contributors(html, 'BerriAI/litellm'), [2])
        self.assertEqual(parse_new_contributors(html.replace(' id="new-contributors"', ''), 'BerriAI/litellm'), [2])
        self.assertEqual(parse_new_contributors('<h1>Patch</h1><p>No new people</p>', 'BerriAI/litellm'), [])

    def test_invalid_documents_fail_closed(self):
        with self.assertRaises(RuntimeError):
            parse_release_index('<h1>Unavailable</h1>')
        with self.assertRaises(RuntimeError):
            parse_new_contributors('<p>Upstream failure</p>', 'BerriAI/litellm')
        for href in ['https://docs.litellm.ai.evil.test/release_notes/a/b', 'https://user@docs.litellm.ai/release_notes/a/b', '//evil.test/release_notes/a/b', '/release_notes/../admin', '/release_notes/a/b?token=x']:
            self.assertIsNone(release_url(href))


class WeeklyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.config = Config('secret', '123', '456', state_file=str(Path(temp.name) / 'state.json'))
        self.messages = self.patch('fetch_channel_messages', return_value=[])
        self.index = self.patch('fetch_release_entries', return_value={OLD_NOTE: parse_time('2026-10-01T00:00:00Z')})
        self.notes = self.patch('fetch_release_prs', return_value=[])
        self.search = self.patch('fetch_merged_numbers', return_value={1})
        self.github = self.patch('fetch_pull_request', side_effect=lambda _repo, n, _token: pull(n))
        self.post = self.patch('post_message', return_value={'id': '999', 'channel_id': '456'})

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(poller, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def poll(self, now=FRIDAY):
        poller.poll_once(self.config, now)

    def add_release(self, numbers):
        self.index.return_value[NOTE] = parse_time('2026-11-11T00:06:06Z')
        self.notes.return_value = numbers

    def test_union_dedupes_and_credits_author_not_person_sharing(self):
        self.add_release([1, 2])
        self.messages.return_value = [message(10, '111', (1,)), message(11, '222', (1,))]
        self.github.side_effect = lambda _r, n, _t: pull(n, merged_at=RECENT if n == 1 else '2026-11-02T01:00:00Z')
        self.poll()
        payload = self.post.call_args.args[2]
        self.assertTrue(payload['content'].startswith(poller.OPENING + '\n\n'))
        self.assertTrue(payload['content'].endswith(poller.CLOSING))
        self.assertEqual(payload['content'].count('- [octocat]'), 1)
        self.assertIn(' - merged [#1]', payload['content'])
        self.assertIn('[#2]', payload['content'])
        self.assertNotIn('<@111>', payload['content'])
        self.assertEqual(payload['allowed_mentions'], {'parse': [], 'users': []})
        state = load_state(self.config)
        self.assertEqual(len(state['roundups'][0]['contributors'][0]['candidates']), 2)
        self.assertTrue(state['pull_requests']['2']['announced'])

    def test_friday_only_restart_dedup_and_same_author_next_week(self):
        self.messages.return_value = [message(10)]
        self.poll(THURSDAY)
        self.post.assert_not_called()
        self.search.assert_not_called()
        self.messages.return_value = []
        self.poll()
        self.poll(FRIDAY + timedelta(minutes=5))
        self.post.assert_called_once()
        self.messages.return_value = [message(11, numbers=(2,))]
        self.search.return_value = {2}
        self.github.side_effect = lambda _r, n, _t: pull(n, merged_at='2026-11-18T01:00:00Z')
        self.poll(FRIDAY + timedelta(days=7))
        self.assertEqual(self.post.call_count, 2)
        self.assertIn('[#2]', self.post.call_args.args[2]['content'])

    def test_exact_cutoffs_and_old_messages_are_supported(self):
        self.messages.return_value = [message(10, numbers=(1, 2, 3, 4, 5))]
        self.search.return_value = {1, 2, 3, 4, 5}
        dates = {1: '2026-11-06T13:59:59Z', 2: '2026-11-06T14:00:00Z', 3: '2026-11-13T13:59:59Z', 4: '2026-11-13T14:00:00Z', 5: None}
        self.github.side_effect = lambda _r, n, _t: pull(n, merged_at=dates[n])
        self.poll()
        self.assertEqual(load_state(self.config)['roundups'][0]['contributors'][0]['pr_numbers'], ['2', '3'])

    def test_only_this_weeks_releases_and_no_bot_authors(self):
        self.add_release([1, 2])
        self.github.side_effect = lambda _r, n, _t: pull(n, login='helper[bot]', bot=True)
        self.poll()
        self.post.assert_not_called()
        self.notes.assert_called_once_with(NOTE, 'BerriAI/litellm')

    def test_manual_mapping_tags_correct_user_and_canonical_account_dedup(self):
        self.config = replace(self.config, github_discord_map={'octocat': '789'})
        self.messages.return_value = [message(10, '111', (1, 2))]
        self.search.return_value = {1, 2}
        self.github.side_effect = lambda _r, n, _t: pull(n, login='OctoCat' if n == 1 else 'octocat')
        self.poll()
        payload = self.post.call_args.args[2]
        self.assertEqual(payload['content'].count('<@789>'), 1)
        self.assertEqual(payload['allowed_mentions']['users'], ['789'])
        self.assertNotIn('<@111>', payload['content'])

    def test_verification_failure_keeps_progress_and_prevents_partial_post(self):
        self.add_release([1, 2])
        self.github.side_effect = [pull(1), api.TransientAPIError('rate limit')]
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        self.post.assert_not_called()
        self.assertEqual(load_state(self.config)['weekly']['end'], iso(FRIDAY))
        self.github.reset_mock(side_effect=True)
        self.github.return_value = pull(2)
        self.poll(FRIDAY + timedelta(minutes=5))
        self.github.assert_called_once_with('BerriAI/litellm', 2, None)
        self.post.assert_called_once()

    def test_source_failure_does_not_publish_or_advance_window(self):
        self.messages.return_value = [message(10)]
        self.index.side_effect = api.TransientAPIError('docs unavailable')
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        self.assertEqual(load_state(self.config)['last_message_id'], '10')
        self.assertEqual(load_state(self.config)['weekly']['end'], iso(FRIDAY))
        self.post.assert_not_called()

    def test_partial_delivery_resumes_saved_outbox_without_repeating_success(self):
        self.messages.return_value = [message(n, numbers=(n,)) for n in range(1, 31)]
        self.search.return_value = set(range(1, 31))
        self.github.side_effect = lambda _r, n, _t: pull(n, login=f'contributor-{n}', user_id=n)
        self.post.side_effect = [{'id': '100'}, api.TransientAPIError('timeout')]
        with self.assertRaises(api.TransientAPIError):
            self.poll()
        first = self.post.call_args_list[0].args[2]
        second = self.post.call_args_list[1].args[2]
        self.post.reset_mock(side_effect=True)
        self.post.return_value = {'id': '101'}
        self.poll(FRIDAY + timedelta(minutes=2))
        self.assertEqual(self.post.call_args_list[0].args[2], second)
        self.assertNotIn(first, [call.args[2] for call in self.post.call_args_list])
        self.assertEqual(len(load_state(self.config)['roundups']), 1)
        self.assertEqual(load_state(self.config)['outbox'], [])

    def test_preview_never_changes_production_state_or_pings_candidates(self):
        self.config = replace(self.config, github_discord_map={'octocat': '789'})
        state = load_state(self.config)
        save_state(self.config.state_file, state)
        before = Path(self.config.state_file).read_bytes()
        self.messages.return_value = [message(10)]
        preview = poller.preview_roundup(self.config, THURSDAY)
        self.assertEqual(Path(self.config.state_file).read_bytes(), before)
        self.assertEqual(preview['messages'][0]['allowed_mentions'], {'parse': [], 'users': []})
        self.assertEqual(preview['contributors'][0]['candidates'][0]['discord_id'], '111')
        self.assertIn('<@789>', preview['messages'][0]['content'])
        self.post.assert_not_called()

    def test_check_budget_resumes_next_poll(self):
        self.add_release([1, 2])
        with mock.patch.object(poller, 'MAX_CHECKS_PER_POLL', 1):
            self.poll()
            self.post.assert_not_called()
            self.poll(FRIDAY + timedelta(minutes=5))
        self.post.assert_called_once()
        self.assertEqual(self.github.call_count, 2)

    def test_empty_and_paused_runs_never_send(self):
        self.poll()
        self.post.assert_not_called()
        self.assertEqual(load_state(self.config)['weekly']['end'], iso(FRIDAY + timedelta(days=7)))
        self.config = replace(self.config, announcements_enabled=False)
        self.messages.reset_mock()
        self.poll()
        self.messages.assert_not_called()

    def test_large_messages_are_bounded_and_a_prolific_author_appears_once(self):
        state = load_state(self.config)
        state['weekly'] = new_window(self.config, FRIDAY)
        people = [{'github_login': f'author-{n}', 'discord_id': str(100000000000000000 + n),
                   'pr_numbers': [str(i) for i in range(n * 100 + 1, n * 100 + 90)]} for n in range(1, 120)]
        batches = poller.build_announcements(self.config, state, people)
        text = '\n'.join(b['payload']['content'] for b in batches)
        for person in people:
            self.assertEqual(text.count(f"<@{person['discord_id']}>"), 1)
        for batch in batches:
            self.assertLessEqual(len(batch['payload']['content'].encode('utf-16-le')) // 2, 2000)
            self.assertLessEqual(len(batch['payload']['allowed_mentions']['users']), 100)
        self.assertEqual(sum(len(b['pr_numbers']) for b in batches), 119 * 89)

    def test_migration_retains_dedupe_but_rejects_legacy_outbox(self):
        state = load_state(self.config)
        state.update(version=1, pull_requests={'1': {'announced': True}})
        for key in ('weekly', 'releases', 'roundups'):
            state.pop(key)
        save_state(self.config.state_file, state)
        migrated = load_state(self.config)
        self.assertEqual(migrated['version'], 2)
        self.assertTrue(migrated['pull_requests']['1']['announced'])
        state['outbox'] = [{'payload': {'content': 'old all-time announcement'}}]
        save_state(self.config.state_file, state)
        with self.assertRaisesRegex(RuntimeError, 'Legacy'):
            load_state(self.config)


class SearchTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(api.time, 'sleep')
        self.addCleanup(patcher.stop)
        patcher.start()

    def result(self, n):
        return {'number': n, 'pull_request': {'url': f'https://api.github.com/repos/BerriAI/litellm/pulls/{n}'}}

    def test_pagination_and_range_syntax(self):
        pages = [{'total_count': 101, 'incomplete_results': False, 'items': [self.result(n) for n in range(1, 101)]},
                 {'total_count': 101, 'incomplete_results': False, 'items': [self.result(101)]}]
        with mock.patch.object(api, '_request_json', side_effect=pages) as request:
            numbers = api.fetch_merged_numbers('BerriAI/litellm', THURSDAY, FRIDAY, None)
        self.assertEqual(numbers, set(range(1, 102)))
        self.assertIn('..', request.call_args_list[0].args[0].full_url)
        self.assertIn('page=2', request.call_args_list[1].args[0].full_url)

    def test_search_cap_splits_instead_of_truncating(self):
        pages = [{'total_count': 1001, 'incomplete_results': False, 'items': []},
                 {'total_count': 1, 'incomplete_results': False, 'items': [self.result(1)]},
                 {'total_count': 1, 'incomplete_results': False, 'items': [self.result(2)]}]
        with mock.patch.object(api, '_request_json', side_effect=pages):
            self.assertEqual(api.fetch_merged_numbers('BerriAI/litellm', THURSDAY, FRIDAY, None), {1, 2})

    def test_incomplete_or_wrong_repo_search_fails(self):
        for payload in [{'total_count': 1, 'incomplete_results': True, 'items': []},
                        {'total_count': 1, 'incomplete_results': False, 'items': [{'number': 1, 'pull_request': {'url': 'https://api.github.com/repos/other/repo/pulls/1'}}]}]:
            with mock.patch.object(api, '_request_json', return_value=payload), self.assertRaises(RuntimeError):
                api.fetch_merged_numbers('BerriAI/litellm', THURSDAY, FRIDAY, None)


if __name__ == '__main__':
    unittest.main()
