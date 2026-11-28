# LiteLLM weekly contributor thank-you bot

Every **Friday at 9:00 AM America/Toronto**, combine contributors from the
[official release notes](https://docs.litellm.ai/release_notes) and
**#oss-contributions**, verify their PRs on GitHub, and post in **#announcements**:

> Thank you to this week's contributors!
>
> - mubashir1osmani - merged #1234
> - @discord-contributor (github-login) - merged #2345, #3456
>
> We appreciate everything you do to help build LiteLLM!

Names and PR numbers are clickable. Confirmed Discord accounts are tagged;
otherwise the GitHub username is credited. The weekly roundup does not post
release announcements, open support threads, or call an LLM. Contributor roles
and the PR review agent are separate and stay off until their own flags are set.
The process uses discord.py for a gateway connection so the bot shows as online.

## Exactly who is included?

Each window runs **previous Friday 9:00 AM through this Friday 9:00 AM**,
start inclusive, end exclusive, in the configured timezone (DST-aware).

- **Discord:** human-shared LiteLLM PR links whose actual GitHub `merged_at`
  falls in the window. Old channel messages can qualify when their PR finally
  merges. All accessible text-channel history is paginated initially; new
  messages are read incrementally. Bots, webhooks and other repositories are ignored.
- **Release notes:** the **New Contributors** section of official docs releases
  **published in that window**. Their PRs must actually have merged before the
  cutoff, but can have merged earlier: this recognizes first-time contributors
  when their work ships. This deliberately uses the docs, not GitHub's shorter
  auto-generated release body. Stable, RC and patch note pages are supported.
- Deduplicate by PR number across sources and releases; group by the GitHub
  author's numeric account ID (username fallback). Never reannounce a PR that
  was already thanked. A contributor can appear next week for a different PR.
- Credit human PR authors, not merge operators or people who merely shared links.
  Bot authors are excluded. Empty weeks produce no thank-you post.

The bot collects and verifies release contributors throughout the week, then
publishes at Friday's boundary. Discord candidates are discovered through GitHub
merge-date search and individually verified through the PR API. Large searches
are split instead of silently losing results above GitHub's 1,000-result limit.
A source outage or incomplete verification **defers the entire roundup**; checks
are checkpointed and resumed. API outages/rate limits can delay delivery.

A new installation starts with the current weekly window, **not all-time history**.
If first started on Friday after the scheduled hour, it catches up that Friday.
After downtime, saved outstanding windows are processed in order. The public
release index has finite history; if it no longer covers a missed period, the
bot stops with a manual-catch-up error rather than silently omitting releases.

## Discord mentions and identity

Use `GITHUB_DISCORD_MAP` for moderator-confirmed GitHub → Discord links:

```dotenv
GITHUB_DISCORD_MAP={"github-login":"123456789012345678"}
```

Keys are case-insensitive; IDs must be strings. Unknown users are still thanked
by GitHub username. Only mapped IDs may be pinged; roles and @everyone never are.
**Sharing a PR or having a similar display name is not verified identity.**
Preview reports list possible matches with the Discord sharer ID, source message
ID and PR number for moderator review. Confirm mappings once, then reuse them in
future weeks. No LLM is needed to guess identities or process credentials.
The old `CONTRIBUTOR_ATTRIBUTION=message_author` setting is no longer used.

## Render configuration

Use the existing **litellm-discord Background Worker**, one instance:

- **Build Command:** `python3 -I -m compileall -q bot.py contributor_bot`
- **Start Command:** `python3 bot.py`
- **Persistent disk:** `/opt/render/project/src/.data` (1 GB; configured directly on the worker — `render.yaml` only manages LiteLLM Lens).
- Keep **Auto-Deploy**, **PR Previews**, and Blueprint **Auto Sync** off.

| Environment variable | Value / purpose |
| --- | --- |
| `DISCORD_BOT_TOKEN` | Required; bot token, never committed. |
| `OSS_CONTRIBUTIONS_CHANNEL_ID` | `1391835163780186154` for this server. |
| `ANNOUNCEMENTS_CHANNEL_ID` | Required; the real #announcements channel ID. |
| `ANNOUNCEMENTS_ENABLED` | `false` during testing; `true` enables collection and Friday posting. Default without the variable is `true`. |
| `ROUNDUP_TIMEZONE` | `America/Toronto` (default). Any installed IANA timezone. |
| `ROUNDUP_TIME` | `09:00` (default, 24-hour local time). Day is always Friday. |
| `GITHUB_DISCORD_MAP` | Optional JSON object of confirmed account links. |
| `STATE_FILE` | Default `.data/contributors.json`; must be on persistent storage. |
| `POLL_INTERVAL_MINUTES` | Default `5`; retry/collection interval, **not** posting frequency. The worker wakes at the Friday boundary. |
| `GITHUB_TOKEN` | Prefer **unset**; only public GitHub reads are needed. Anonymous rate limits are resumable. |
| `GITHUB_REPOSITORY` | Default `BerriAI/litellm`. Release-note links are restricted to that repository. |
| `CONTRIBUTOR_ROLES_ENABLED` | `false` (default). Set `true` only after the contributor-role setup below. |

Requires Python 3.11+ with system timezone data (included in Render's Linux
runtime). No package install is needed. The bot's only credential needs to be its
Discord token. Keep deployment credentials and unrelated environment secrets out
of the worker; keep manual deployments, repository protections and network isolation enabled.

Grant the bot View Channel / Read Message History in #oss-contributions and
View Channel / Send Messages in #announcements and #moderator-only. Enable
Message Content Intent in the Discord Developer Portal. Preview reconciliation
is not automatic identity verification. The weekly roundup does not need
Administrator, Manage Roles, Manage Webhooks, or Mention Everyone. Contributor
roles, when enabled, do need Manage Roles; see below. Threads, forum posts and
edits to old messages are not scanned; post new links in the source text channel.

## Preview before enabling

Keep `ANNOUNCEMENTS_ENABLED=false`. In Render Shell:

```sh
python3 bot.py --preview-channel-id 1123361089082630245
```

This checks both sources for the current week to date, posts the exact thank-you
format to **#moderator-only**, and disables **all pings**. Candidate identity
matches are separately labeled for moderator review. It does **not** mark PRs
as thanked, advance production checkpoints, or send to #announcements.

The separate `.data/contributors.json.preview` cache makes retries resumable.
At most 40 new PRs are verified per attempt; rerun after a budget/rate-limit
message to finish verification. The report is saved as
`.data/contributors.json.preview-report.json`, including every PR and candidate
Discord ID. `python3 bot.py --preview` only prints that report and never posts.
Do not use preview/test commands as the service Start Command.

After reviewing the preview, add confirmed mappings, set
`ANNOUNCEMENTS_ENABLED=true`, and restart the single worker. It will collect
without posting until the scheduled Friday. No cron job or second service is
needed. `python3 bot.py --once` is a one-shot collection/due-date check, not a
force-send command. `--test-channel-id ID` remains a simple delivery-only test.

## Persistent history and upgrades

Keep the state file across deploys. Version-1 contributor state migrates while
retaining announced PRs and Discord IDs. A pending legacy all-time outbox is
rejected for manual review, never automatically sent or silently discarded.
Old release/feed/thread state is not used. Corrupt or mismatched state is rejected.
Changing schedule settings with an existing window also requires explicit state
review so windows are not silently shifted or skipped.

Messages split at Discord's length/mention limits. A very prolific author's row
shows the first 15 PR links and a remaining count; the report retains all PRs.
A saved outbox, per-message receipts and deterministic Discord nonces make
ordinary retries/restarts idempotent. Discord only deduplicates nonces for a few
minutes: a crash after acceptance but before saving a receipt can still duplicate
that batch after a long downtime. Run only one worker/`--once` writer per state
file; previews use their own cache and can run alongside the worker.

## Contributor roles

The same process can grant Discord roles for merged pull requests on
`BerriAI/litellm` **without requiring a GitHub OAuth app, account linking, or a
repository webhook**. It stays off until `CONTRIBUTOR_ROLES_ENABLED=true`. Role
state lives in SQLite (`ROLES_DB_PATH`, default `.data/contributor_roles.db` on
the persistent disk), separate from the weekly JSON ledger.

Thresholds and path rules are in `contributor_bot/roles.py`. Defaults:

| Role | Earned when |
| --- | --- |
| Contributor | 1+ merged human PR |
| OSS Champion | 10+ merged total, or 3+ in one UTC calendar month |
| Agent Builder | a merged PR whose paths contain `mcp`, `agent`, or `a2a`, or whose labels include `agents` |
| Provider Wizard | 3+ merged PRs that touch `litellm/llms/` |

Bots, `MAINTAINER_GITHUB_LOGINS`, GitHub org members, and BerriAI staff handles
are ignored. Once a day at `ROLES_DAILY_TIME` (default `10:00`
`America/Toronto`; on Fridays right after the weekly merge summary, and still
running independently if the summary fails or skips), the bot checks PRs merged
in the **rolling past 7 days** (not full history), stores their metadata
(`number`, author, `merged_at`, labels, and changed file paths — fetching file
lists only when still needed for Agent Builder or Provider Wizard), caches each
author's lifetime merged PR `total_count` via one GitHub search
(`repo:BerriAI/litellm is:pr is:merged author:<login>`) with incremental
updates, resolves only those 7-day community PR authors to Discord members using
the same automatic matcher as the weekly summary:

- **Complete per-day PR fetch**: merged PR searches split into per-day UTC
  `merged:D..D` windows (up to 10 pages / 1,000 rows per day, deduplicated by
  PR number, logging a warning if any single day reaches 1,000). GitHub is the
  sole source of truth for contributions — Discord-side signals never create
  contributions.
- **No-skip GraphQL batch enrichment**: every merged-PR author in the window is
  fully enriched before matching (50 logins per GraphQL query: `name`, `login`,
  `socialAccounts` incl. Discord/X, `bio`, `company`, `websiteUrl`, profile
  `HEAD:README.md` capped at 4 KB, and public email domain, plus commit author
  names/emails), persisted per batch in SQLite (`github_users_cache`) for 7 days.
  Rate limits are handled with exponential backoff/retry and resume rather than
  deferring; if enrichment ultimately fails for an author, matching is skipped
  for that author this run and retried next run (never guessed).
- **Ordered 4-tier matching (each producing concrete evidence)**:
  1. **Direct link**: Discord user ID or handle listed in GitHub
     `socialAccounts` / `bio` / `blog` / `README` matched to a present guild
     member (`username` / `global_name`, never `nick`).
  2. **Exact identity**: GitHub `login` == Discord `username` or `global_name`
     (case/separator-insensitive, never `nick`).
  3. **PR-link poster**: sole eligible non-staff guild member who posted a link
     to that author's PR in Discord AND has at least one corroborating signal
     (name similarity `>= 0.50` to GitHub name/login, ownership language like
     `my PR` / `I opened`, or same X/website). Posters who linked PRs by 2+
     different GitHub authors are rejected.
  4. **Message evidence (`member_identity_notes`)**: when a Discord member's
     `username` / `global_name` does not match a GitHub login and no GitHub
     profile is linked to them, scans that member's guild messages for PR/issue
     links, explicit GitHub profile/fork URLs or handles, ownership statements
     (`my PR`, `I opened`, etc.), and discussed code paths/PR topics against
     real merged-PR authors. Findings are persisted per member in SQLite
     (`member_identity_notes`: `discord_id`, `inferred_github_login` or `NULL`,
     evidence snippets, message links, `confidence`, `updated_at`) so future
     runs reuse them without re-scanning and refresh only when new messages
     arrive. Message evidence still requires a real merged LiteLLM PR author
     (GitHub stays the source of truth) and respects strict 1-to-1 matching.
  5. **List-wise LLM** (`MATCH_MODEL`, falling back to `REVIEW_MODEL` or
     `anthropic/claude-sonnet-4-5`): only when full enrichment is present, over
     the top-5 guild candidates plus explicit `"none of these"` (including
     corroborating message-evidence signals), requiring `>= 2` independent
     signals AND confidence `>= 0.85` (no single-signal accepts).
  Strict 1-to-1 bijection is enforced between Discord members and GitHub logins
  (conflicts grant nothing and log a warning). Confirmed matches are cached in
  SQLite for 30 days, and definitive non-matches are negatively cached in
  `match_attempts` for 7 days (expiring so new evidence is picked up).

It verifies the matched Discord member is currently in the guild before
granting newly earned roles, logging `discord_id`, `github_login`, `method`,
`confidence`, and `evidence` on every grant. By default,
**`ROLES_ANNOUNCE_ENABLED=false`** so roles are granted silently and marked
`announced_at` immediately (turning it on later never posts a backlog), logging
`Role announcements disabled (ROLES_ANNOUNCE_ENABLED=false); granted N roles
silently`. When `ROLES_ANNOUNCE_ENABLED=true`, newly earned roles are **batched
into a single message** (chunked only if needed, posting nothing when there are
zero new grants) to `ROLE_ANNOUNCEMENTS_CHANNEL_ID`. Each completed local date
is recorded in `roles_daily_runs` and concurrent runs across instances are
guarded by a DB-backed lease lock (`job_locks`), so restarts and redeploys
never overlap or run twice on the same day, while catching up automatically if
the bot was offline at `10:00`. A one-time boot re-validation job re-checks all
existing grants under these rules and silently revokes roles that no longer
pass. Docs Hero is no longer granted; anyone who already has that Discord role
keeps it.

### Minimal setup (no GitHub OAuth or webhook required)

1. **Copy the 4 existing Discord role IDs** (`Contributor`, `OSS Champion`,
   `Agent Builder`, and `Provider Wizard`) from Server Settings → Roles into
   `ROLE_CONTRIBUTOR_ID`, `ROLE_OSS_CHAMPION_ID`, `ROLE_AGENT_BUILDER_ID`, and
   `ROLE_PROVIDER_WIZARD_ID`.
2. **Place the bot's role above those 4 roles** in Server Settings → Roles, and
   grant the bot **Manage Roles** plus View Channel and Send Messages in the
   announcements channel (for example `#shipped`).
3. **Set Render environment variables**:
   - `CONTRIBUTOR_ROLES_ENABLED=true`
   - `DISCORD_GUILD_ID`
   - `ROLE_CONTRIBUTOR_ID`, `ROLE_OSS_CHAMPION_ID`, `ROLE_AGENT_BUILDER_ID`,
     `ROLE_PROVIDER_WIZARD_ID` (optional `ROLE_*_NAME` overrides change
     announcement copy only)
   - `ROLE_ANNOUNCEMENTS_CHANNEL_ID`
   - Optional: `ROLES_DAILY_TIME=10:00` (24-hour `HH:MM` in
     `MERGED_POST_TIMEZONE`, default `America/Toronto`),
     `ROLES_BACKFILL_ON_BOOT=true` (one-time historical backfill on startup),
     `BACKFILL_SINCE=2023-01-01T00:00:00Z`, `GITHUB_TOKEN` (or
     `GITHUB_ACTOR_TOKEN`) for higher GitHub read limits.
4. **Historical backfill**: either set `ROLES_BACKFILL_ON_BOOT=true` for a
   one-time boot run, or have a moderator run `/roles-backfill` in Discord
   (ephemeral, newest-first so recent contributors get roles first, rate-limit
   aware, time-bounded, and resumable across `/roles-backfill` runs via SQLite
   `backfill_windows` + cursor until all windows complete — full history is
   never run by the daily job). Confirmed
   matches from the weekly merge summary (`github_discord_matches`) are reused
   directly without re-running the LLM. Backfill assigns roles silently and
   posts **at most a single summary line** to `ROLE_ANNOUNCEMENTS_CHANNEL_ID`
   (never one message per contributor).

`DISCORD_PUBLIC_KEY`, `GITHUB_WEBHOOK_SECRET`, `GITHUB_OAUTH_*`, and the
Interactions Endpoint URL are all **optional** and not required. `/link-github`
and `/unlink-github` stay hidden unless GitHub OAuth is configured. `/my-prs` shows an ephemeral PR snapshot for any auto-matched (or manually
mapped) contributor, `/roles-backfill` is mod-only (`Manage Messages`), and `/links-backfill` is mod-only (`Manage Messages`) to backfill `pr_link_posters`
from stored `review_threads` and Discord guild message search (paging up to
offset 9975 with HTTP 202 retry handling).

### Optional webhook / GitHub OAuth

If you also want real-time webhook delivery or optional `/link-github` OAuth:

- Set `GITHUB_WEBHOOK_SECRET` and point a `BerriAI/litellm` **Pull requests**
  webhook at `https://<public-host>/github/webhook`.
- Set `GITHUB_OAUTH_CLIENT_ID`, `GITHUB_OAUTH_CLIENT_SECRET`, and
  `GITHUB_OAUTH_REDIRECT_URI` (`https://<public-host>/github/oauth/callback`)
  to enable `/link-github` and `/unlink-github`.
- Set `DISCORD_APPLICATION_ID` and `DISCORD_PUBLIC_KEY` only if you want HTTP
  interactions (`https://<public-host>/discord/interactions`) instead of gateway
  slash commands.
- You can also run `python3 bot.py --backfill-merged-prs` in Render Shell
  (`GITHUB_TOKEN` or `GITHUB_ACTOR_TOKEN` required; `BACKFILL_SINCE` defaults to
  `2023-01-01T00:00:00Z`).

### Optional invite comment

`PR_COMMENT_INVITES_ENABLED` defaults to `false`. Set it to `true` and set
`DISCORD_INVITE_URL` to let the bot comment once on a merged pull request whose
author has not linked Discord, inviting them to run `/link-github`. The GitHub
token then needs permission to write issue comments. Leave this off unless you
want that comment on the repository.

## Discord gateway

The bot connects to the Discord gateway with discord.py, so it shows as online
with a custom status (`discord.CustomActivity`) of **skedaddling 🏃‍♂️**. Set
`BOT_STATUS_TEXT` to override the status text, and optionally set
`BOT_STATUS_ROTATE=true` to rotate through a handful of fun statuses every 15
minutes.

Required intents in the Developer Portal:

- **Guilds**
- **Guild Messages**
- **Message Content** (privileged; without it the gateway will not stay connected)

**Server Members** is requested only when contributor roles are enabled
(`DISCORD_MEMBERS_INTENT` defaults to that). Turn that privileged intent on in
the portal too, or set `DISCORD_MEMBERS_INTENT=false` so the gateway can still
connect. Role assignment itself uses REST and does not need the members intent.

Slash commands are delivered over the gateway when the Interactions Endpoint
URL is empty. Leave it empty unless you want HTTP interactions. On connect the
bot syncs its slash commands to the guild with discord.py's command tree, which
is what makes `/recheck` and `/review-pr` show up in channels and in threads.
An empty tree is not synced, so a failed sync cannot wipe commands that were
already registered. The HTTP server still handles GitHub webhooks and GitHub
OAuth. If the endpoint URL is set, Discord sends interactions there and the
existing HTTP handler answers them. `/recheck` only works inside a review
thread. The parent channel does not have to be in `REVIEW_CHANNEL_IDS`. The
bot defers the interaction before the agent run and edits that reply when the
run finishes, including the cooldown and daily-budget notes. A plain message
such as `check again` in the thread starts the same recheck.

If discord.py is not installed, the process logs a warning, stays offline, and
keeps the Friday roundup on REST. Channel review messages are polled only while
the gateway is disconnected.

## PR review agent

People ask for reviews on `BerriAI/litellm` faster than one maintainer can read
them. With `PR_REVIEW_ENABLED=true` an agent pre-reviews a pull request in a
Discord thread. It stays off by default, and turning it on does not change who
the Friday roundup thanks.

The agent runs on the OpenAI Agents SDK (`openai-agents`) through
`LitellmModel`, so `REVIEW_MODEL=anthropic/claude-sonnet-4-5` still uses
`ANTHROPIC_API_KEY` (or whichever provider key LiteLLM expects). Message
classification stays a small `litellm.completion` call and does not need the
Agents SDK. `REVIEW_AGENT_MAX_TOKENS`, `REVIEW_AGENT_MAX_STEPS`, and
`REVIEW_DAILY_RUN_BUDGET` are unlimited when unset, empty, or `0`. A positive
`REVIEW_AGENT_MAX_TOKENS` caps **output** tokens only. Each turn re-sends the
prompt, so adding those input totals together used to stop a large pull
request (including one with the cap set to `1000000`) before `submit_verdict`.
Unset `REVIEW_AGENT_MAX_TOKENS` on Render if a review still says it ran out of
budget. Steps still stop at 50 (`max_turns`) so a tool loop cannot run
forever. Each model call may use up to 8192 output tokens, which Anthropic
needs in order to finish a tool call. Tool results are truncated to valid JSON
per call. If the provider still reports a context-window error, the run
retries once with shorter tool results. `REVIEW_AGENT_TIMEOUT_SECONDS`
defaults to 600. `0` means no wall-time cap. Discord handlers call
`run_review_session` (the Agents SDK). The older in-process loop is not on
that path. OpenAI tracing is off unless `OPENAI_API_KEY` is set or
`REVIEW_AGENT_TRACING=true`. Set `REVIEW_AGENT_TRACING=false` to force it off
even when that key is present.

The review starts with one `review_bundle` call (bots, unresolved threads, CI,
file names). It does not download the full diff. A file patch is read only
when a bot finding or a failing check names that file. Classifying channel
messages uses `anthropic/claude-haiku-4-5` unless `REVIEW_CLASSIFIER_MODEL` is
set. A non-command message is scored by TypeSafe Jev (`POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`) before anyone is answered. `should_reply` is a noul; the bot stays silent below `DECISION_REPLY_THRESHOLD` (default 0.7) or when the route is `maintainer_only`. Simple questions then go to `REVIEW_FAST_MODEL` and pull-request or bug-report work goes to `REVIEW_MODEL`, through LiteLLM's Autorouter (`auto_router/complexity_router`) with a custom classifier that reuses that one Jev decision. Without `TYPESAFE_API_KEY` the same decision is a Haiku JSON call. Explicit review requests, `/recheck`, and `check again` skip the gate.

As soon as a request is accepted, the bot reacts with 👀 and then the guild's
custom `ty` emoji before the agent runs. `ACK_REACTIONS` is a comma-separated
list of unicode emoji, `<:name:id>`, `name:id`, or a guild emoji name (for
example `ACK_REACTIONS=👀,ty`). Unset, it uses 👀 plus the first guild emoji
matching `ty`, `litellm_ty`, `thankyou`, or `thank_you` (case-insensitive). If
none of those names exist, it falls back to 🙏 and logs the fallback. Reactions
are resolved lazily at reaction time from `message.guild.emojis` (the gateway
enables `Intents.emojis_and_stickers` so discord.py caches guild emojis), and
fall back to `await guild.fetch_emojis()` cached per guild with a 5-minute TTL
if the guild cache is empty. On the first resolution for a guild, the bot logs
the guild id, emoji count, all visible emoji names, and the resolved reactions.
Slash
commands with no source message are not reacted to. Reactions are best-effort
and need **Add Reactions** in addition to **Read Message History**.

`bugbot run` ([Bugbot](https://cursor.com/docs/bugbot)) and `@greptileai` ([Greptile](https://www.greptile.com/docs/code-review/developer-essentials)) are the only default trigger comments. Veria publishes no re-run comment, so it is not triggered. `GITHUB_ACTOR_TOKEN` posts them as that user, once per commit (and also authenticates GitHub reads when `GITHUB_TOKEN` is unset). `REVIEW_DRY_RUN=true` only logs them. On Fridays at 10:00 America/Toronto the bot posts community merges since the previous Friday post to `ANNOUNCEMENTS_CHANNEL_ID` (falling back to `#announcements`). Bots and `MAINTAINER_GITHUB_LOGINS` are left out. `/merged-this-week` previews that text.

Tools are read-only on GitHub except those allowlisted trigger comments: pull request metadata, the file list, one file's
diff, a full file at the pull request head or base, code search in this
repository only, Bugbot/Greptile/Veria threads (including resolved state), CI
check output and a few annotations, and the contributor's linked Discord
account plus merged history from this bot's store. On Discord it can post or
edit a message in the review thread. It cannot approve, merge, or comment on
GitHub. `ping_maintainer` refuses unless `submit_verdict` has already decided
the pull request is ready. Diffs, comments, and Discord messages are untrusted:
the system prompt tells the model to ignore instructions inside them, and tool
output is redacted for token-shaped strings.

It opens one public thread per pull request when:

- a human message in `REVIEW_CHANNEL_IDS` is classified as a review request
  (`REVIEW_CLASSIFY_MESSAGES` defaults to true; a message with no review hint
  never calls the model). Set `REVIEW_WATCH_ALL_CHANNELS=true` to classify
  messages in every channel the gateway can see
- someone runs `/review-pr <url>`

Set `REVIEW_CLASSIFY_MESSAGES=false` to go back to URL detection, plus
`REVIEW_REQUIRE_ASK=true` if the message must also ask for a review.

Inside that thread the agent answers follow-ups ("why is this a bug?").
`check again`, `recheck`, `fixed it`, `pushed a fix`, or `updated the pr`
runs another review, still under the per-PR cooldown. `/recheck` does the
same thing from the slash-command menu inside the thread.

### How a thread goes

```
contributor: can someone review https://github.com/BerriAI/litellm/pull/54722
bot: needs work
     - cursor[bot]: unresolved on litellm/proxy/server.py - timeout returns early
     - litellm/proxy/server.py:120 - the stream path never applies the timeout
     checklist:
     - cover a hung upstream
contributor: why is this a bug?
bot: the stream branch returns before the wrapper, so a stuck upstream never gets cancelled.
contributor: i fixed it, check again
bot: ready for maintainer ✅
     <@contributor> <@&maintainers>
```

The verdict is **ready for maintainer ✅** only when every configured bot has
re-reviewed the latest commit, none of them have an unresolved current thread,
CI is green, the pull request is mergeable and not a draft, and the agent did
not report a blocking bug. Otherwise it is **needs work** with file and line
items when the agent has them. Messages are split at Discord's 2000-character
limit. The maintainer role is pinged only on a ready verdict. Make that role
mentionable.

Default bot logins, seen on recent `BerriAI/litellm` reviews such as #54722:

| Bot | GitHub login |
| --- | --- |
| Cursor Bugbot | `cursor[bot]` (`BUGBOT_REVIEW`, stale passes include `BUGBOT_REVIEW_STALE`) |
| Greptile | `greptile-apps[bot]` |
| Veria | `veria-ai[bot]` |

`REVIEW_WEBHOOK_RECHECK=true` re-runs the agent for a pull request that already
has a thread when GitHub sends `pull_request` `synchronize` or
`ready_for_review`, `pull_request_review` `submitted`, or `check_run` /
`check_suite` `completed`. New GitHub events do not open threads by themselves.

`REVIEW_COOLDOWN_SECONDS` (default 600) limits re-reviews. Follow-up answers use
`REVIEW_FOLLOWUP_COOLDOWN_SECONDS` (default 45). Daily run limits stay off
unless `REVIEW_DAILY_RUN_BUDGET` is a positive number.

### Setup

The deploy is a Render **web service**. Build command
`pip install -r requirements.txt`. Start command `python3 bot.py`. That
installs discord.py, litellm, and the OpenAI Agents SDK. Do not run two pollers on one state file.
Set the provider key the model expects (`OPENAI_API_KEY`, and so on).

1. Enable **Guilds**, **Guild Messages**, and **Message Content** on the bot.
   Enable **Server Members** too if contributor roles are on, or set
   `DISCORD_MEMBERS_INTENT=false`.
2. Invite the bot with the `bot` and `applications.commands` scopes. In the
   review channels grant **View Channel**, **Read Message History**,
   **Send Messages**, **Add Reactions**, **Create Public Threads**, and **Send
   Messages in Threads**. Add Reactions is what lets the bot put 👀 and ty on
   the request before the review finishes. Read Message History is required
   for that reaction and for thread history.
3. Set `PR_REVIEW_ENABLED=true`, `REVIEW_MODEL`, and `REVIEW_CHANNEL_IDS`.
   Set `DISCORD_GUILD_ID` and `DISCORD_APPLICATION_ID`. `DISCORD_PUBLIC_KEY`
   is only required if you still set an Interactions Endpoint URL. Leave that
   URL empty to take slash commands on the gateway. `/review-pr` and
   `/recheck` are registered for the guild, plus the role commands when those
   are enabled.
4. `GITHUB_TOKEN` should be able to read pull requests, contents, reviews, and
   checks on `BerriAI/litellm`. It must not be able to write, approve, or merge.
   Code search needs a token. Anonymous access still works for the other reads
   until the rate limit.
5. Restart. The bot should show online. Paste a pull request URL or run
   `/review-pr`. The thread should show `ready for maintainer ✅` or
   `needs work`.
6. For webhook re-checks, set `REVIEW_WEBHOOK_RECHECK=true` and
   `GITHUB_WEBHOOK_SECRET` (already required when contributor roles are on).
   On the GitHub webhook, send **Pull requests**, **Pull request reviews**,
   and **Check runs** (or check suites). The HTTP server still verifies
   `X-Hub-Signature-256`.

`PR_REVIEW_ENABLED=false` leaves the Friday roundup and role commands as they
were. The gateway still connects so the bot shows online.

## Contributor PR backlog board

The bot can keep a GitHub Projects (v2) board synced with community contributor
PRs on `BerriAI/litellm` (excluding bots, org members/staff, and
`MAINTAINER_GITHUB_LOGINS`). It stays off with a single `INFO` log until you set
`PR_BOARD_PROJECT_ID` (or `PR_BOARD_OWNER` + `PR_BOARD_NUMBER`).

### 1. Create the GitHub Project (v2)

1. Create a new GitHub Project (v2) under your user or organization (for
   example, `BerriAI` or your personal GitHub account). Note either:
   - the owner login + project number from the URL
     (`https://github.com/orgs/<owner>/projects/<number>` or
     `https://github.com/users/<owner>/projects/<number>`), **or**
   - the GraphQL Project node ID (`PVT_...`).
2. Configure the built-in single-select **`Status`** field (or a custom
   single-select field named in `PR_BOARD_STATUS_FIELD`) with these exact
   options:
   - `Open`
   - `In review` (has a review or a Bugbot / Greptile / Veria comment)
   - `Changes requested`
   - `Approved`
   - `Merged`
   - `Closed`

### 2. Token scope (`GITHUB_ACTOR_TOKEN`)

The sync uses the GitHub GraphQL API with `GITHUB_ACTOR_TOKEN`:

- **Classic personal access token:** enable the **`project`** scope (plus
  `public_repo` if you also use `GITHUB_ACTOR_TOKEN` for review trigger
  comments).
- **Fine-grained personal access token:** grant **Projects: Read and write** on
  the project's owner (user or organization) and public repository read access.

### 3. Render environment variables

- `GITHUB_ACTOR_TOKEN` — token with `project` scope.
- `PR_BOARD_PROJECT_ID` (e.g. `PVT_kwDO...`) **or** both `PR_BOARD_OWNER` and
  `PR_BOARD_NUMBER` (e.g. `BerriAI` and `12`).
- Optional: `PR_BOARD_SYNC_SECONDS` (default `600`, ~10 minutes),
  `PR_BOARD_STATUS_FIELD` (default `Status`), `MAINTAINER_GITHUB_LOGINS`
  (comma-separated logins to exclude alongside org members).

Every ~10 minutes, the bot fetches open and recently updated community PRs,
adds missing ones (`addProjectV2ItemById`), caches their project item IDs and
last-set status in SQLite (`ROLES_DB_PATH`) so repeat runs make zero mutations
for unchanged PRs, and updates `Status` when a PR changes state. Moderators can
run `/pr-board` in Discord for an ephemeral reply with the board link and PR
counts per status.

## LiteLLM Lens tracing (OpenTelemetry)

The repo root includes a Render Blueprint (`render.yaml`) that deploys
[LiteLLM Lens](https://github.com/BerriAI/lens) in `standalone` mode (`lens`
web service on port `4318` + `lens-clickhouse` private service) inside the
`aria-litellm-discord` project (`production` environment). `render.yaml`
**only** declares Lens resources and does **not** define the existing `lol`
background worker, so syncing the Blueprint never modifies the bot service.

On the bot worker, tracing via OpenTelemetry + OpenInference
(`openinference-instrumentation-litellm`) stays **off** (full no-op, zero
third-party imports) until `OTEL_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) is
set. When set, the bot configures an OTel `TracerProvider` with an OTLP HTTP
span exporter, calls `LiteLLMInstrumentor().instrument(tracer_provider=provider)`
(without adding `"otel"` to `litellm.callbacks`, which Lens v0.6.0 hides as
gateway-internal), and wraps each logical task (`role_matcher`, `pr_review`,
`pr_review_followup`, `discord_qa`, `weekly_summary`, `message_router`,
`pr_review_classifier`) in an OpenInference `AGENT` span so Lens shows one run
per task with its nested LLM calls.

### Bot environment variables

`contributor_bot/tracing.py` reads the following variables
(`OTEL_EXPORTER_OTLP_*` aliases take precedence if both are set):

| Environment variable | Value / notes |
| --- | --- |
| `OTEL_ENDPOINT` | `https://<lens-host>/v1/traces` (or base URL `https://<lens-host>`, auto-normalized to `/v1/traces`; alias: `OTEL_EXPORTER_OTLP_ENDPOINT`). Leaving both unset keeps tracing completely off. |
| `OTEL_HEADERS` | `Authorization=Bearer <tracing-key>` (comma-separated `k=v,k2=v2` with literal values; alias: `OTEL_EXPORTER_OTLP_HEADERS`). |
| `OTEL_SERVICE_NAME` | `litellm-contributor-bot` (optional; defaults to `litellm-contributor-bot`). |

**Header formatting note:** `parse_otel_headers()` splits `OTEL_HEADERS` on `,`
and then on the first `=` **without** URL-decoding. Set the value literally as
`Authorization=Bearer <tracing-key>` (using `=` and a literal space; do **not**
use `Authorization: Bearer ...` and do **not** percent-encode the space as
`%20`).

## Checks

Offline tests never post or use real credentials:

```sh
python3 -m unittest discover -v
```
