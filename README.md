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
otherwise the GitHub username is credited. No release announcements, support
threads, LLM calls, or third-party Python dependencies.

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
- **Persistent disk:** `/opt/render/project/src/.data` (1 GB in `render.yaml`).
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

Requires Python 3.11+ with system timezone data (included in Render's Linux
runtime). No package install is needed. The bot's only credential needs to be its
Discord token. Keep deployment credentials and unrelated environment secrets out
of the worker; keep manual deployments, repository protections and network isolation enabled.

Grant the bot View Channel / Read Message History in #oss-contributions and
View Channel / Send Messages in #announcements and #moderator-only. Enable
Message Content Intent in the Discord Developer Portal. Preview reconciliation
is not automatic identity verification. No Administrator, Manage Roles, Manage
Webhooks, or Mention Everyone permission is needed. Threads, forum posts and
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

## Checks

Offline tests never post or use real credentials:

```sh
python3 -m unittest discover -v
```
