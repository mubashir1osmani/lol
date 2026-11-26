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
release announcements, open support threads, or call an LLM, and it has no
third-party dependencies. Contributor roles and PR pre-review are separate
features and stay off until their own flags are set.

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
`BerriAI/litellm`. It stays off until `CONTRIBUTOR_ROLES_ENABLED=true`, so an
existing weekly-roundup worker keeps its current behavior. Role state lives in
SQLite (`ROLES_DB_PATH`, default `.data/contributor_roles.db` on the persistent
disk), separate from the weekly JSON ledger. The SQL is isolated in
`contributor_bot/store.py` if you later move it to Postgres.

Thresholds and path rules are in `contributor_bot/roles.py`. Defaults:

| Role | Earned when |
| --- | --- |
| Contributor | 1+ merged human PR |
| OSS Champion | 10+ merged total, or 3+ in one UTC calendar month |
| Agent Builder | a merged PR whose paths contain `mcp`, `agent`, or `a2a`, or whose labels include `agents` |
| Provider Wizard | 3+ merged PRs that touch `litellm/llms/` |
| Docs Hero | 3+ merged PRs that touch `docs/` |

Bot authors are ignored. Roles are sticky: the bot adds them and does not remove
them. A person keeps OSS Champion after the month they earned it. `/unlink-github`
drops the account link and leaves roles already granted.

### Discord setup

1. Invite the bot with the `bot` and `applications.commands` scopes. For roles,
   grant **Manage Roles**, plus View Channel and Send Messages in the
   announcement channel (for example #shipped). The weekly roundup still needs
   its existing channel permissions and the Message Content Intent.
2. In the server role list, put the bot's role **above** every role it grants.
   Discord rejects assignments otherwise; the log says so and the bot keeps running.
3. Copy each role's snowflake into `ROLE_CONTRIBUTOR_ID`, `ROLE_OSS_CHAMPION_ID`,
   `ROLE_AGENT_BUILDER_ID`, `ROLE_PROVIDER_WIZARD_ID`, and `ROLE_DOCS_HERO_ID`.
   Optional `ROLE_*_NAME` overrides change the announcement text only.
4. Set `DISCORD_GUILD_ID`, `DISCORD_APPLICATION_ID`, and `DISCORD_PUBLIC_KEY`
   (the 64-character hex key from the application). Set
   `ROLE_ANNOUNCEMENTS_CHANNEL_ID` to the channel that should receive posts like
   `<@user> just became an oss champion 🎉`.
5. Start the bot, then set the application's Interactions Endpoint URL to
   `https://<public-host>/discord/interactions`. Discord sends a ping; the
   endpoint must already be up. Slash commands are registered for that guild:
   `/link-github`, `/unlink-github`, and `/my-prs`.

`/link-github` replies with a GitHub OAuth link that expires in 10 minutes.
`/my-prs` is ephemeral and shows the caller's open LiteLLM pull requests (with
review status), the merged count this bot has recorded, and the roles those
pull requests earn.

### GitHub setup

1. Create a GitHub OAuth app. Callback URL:
   `https://<public-host>/github/oauth/callback`, the same value as
   `GITHUB_OAUTH_REDIRECT_URI`. Homepage URL can be the Discord invite.
   The app only requests the `read:user` scope. Put the client id and secret in
   `GITHUB_OAUTH_CLIENT_ID` and `GITHUB_OAUTH_CLIENT_SECRET`.
2. On `BerriAI/litellm`, add a webhook for **Pull requests** only.
   Payload URL `https://<public-host>/github/webhook`, content type
   `application/json`, secret `GITHUB_WEBHOOK_SECRET`. The bot checks
   `X-Hub-Signature-256` and ignores anything that is not a merged pull request
   from a human on the configured repository.
3. `GITHUB_TOKEN` should be a fine-grained token that can read pull requests
   (and metadata) on `BerriAI/litellm`. It is used for changed-file lists,
   historical backfill, and `/my-prs`. Anonymous access still works until the
   rate limit; file paths are what the path-based roles need.
4. The process listens on `HTTP_HOST`:`HTTP_PORT` (default `0.0.0.0` and `8080`,
   or Render's `PORT` when `HTTP_PORT` is unset). GitHub and Discord need
   public HTTPS. A Render **web service** with the same start command
   (`python3 bot.py`) provides that. Leave the existing background worker as-is
   until you are ready to receive webhooks; do not run two pollers on one state file.

On each merged pull request the bot stores the number, author, `merged_at`,
labels, and changed paths, then re-evaluates that author. If their Discord
account is linked, new roles are assigned and announced. A linked author can
also get a short `<@user> just shipped #1234 🎉` line
(`MERGE_SHOUTOUTS_ENABLED`, default `true`). Linking an account evaluates roles
immediately and catches up that person's older merged pull requests in the
background.

Backfill everyone else, rate-limit aware, after people have linked (or before;
linking catches a person up either way):

```sh
python3 bot.py --backfill-merged-prs
```

`GITHUB_TOKEN` is required. `BACKFILL_SINCE` defaults to `2023-01-01T00:00:00Z`.
Completed search windows are checkpointed, so a rerun skips work it finished.
The command does not post a shoutout per historical pull request; it announces
a role once when a linked account newly qualifies.

### Optional invite comment

`PR_COMMENT_INVITES_ENABLED` defaults to `false`. Set it to `true` and set
`DISCORD_INVITE_URL` to let the bot comment once on a merged pull request whose
author has not linked Discord, inviting them to run `/link-github`. The GitHub
token then needs permission to write issue comments. Leave this off unless you
want that comment on the repository.

## PR pre-review

People ask for reviews on `BerriAI/litellm` faster than one maintainer can read
them. With `PR_REVIEW_ENABLED=true` the bot pre-reviews a pull request in a
Discord thread so the maintainer sees it after the automated reviewers are
clean. It stays off by default, and turning it on does not change the Friday
roundup.

The bot only posts in Discord. It does not approve, merge, or comment on GitHub.

It starts a public thread when either of these happens:

- a human message in `REVIEW_CHANNEL_IDS` contains a `BerriAI/litellm` pull
  request URL (set `REVIEW_REQUIRE_ASK=true` to also require a review ask such
  as "please review" or "ptal")
- someone runs `/review-pr <url>` in one of those channels

One thread per pull request. A later link or `/review-pr` reuses it.
`/recheck` inside that thread runs the gate again. The first time the bot
watches a channel it does not backfill old messages.

Optional `REVIEW_WEBHOOK_RECHECK=true` also re-checks a pull request the bot
already has a thread for, when GitHub sends `pull_request` `synchronize` or
`ready_for_review`, or `pull_request_review` `submitted`. New GitHub events do
not open threads by themselves.

Each run is rate-limited per pull request (`REVIEW_COOLDOWN_SECONDS`, default
600, minimum 30). That applies to `/recheck` and to webhook updates.

### What the thread says

The gate uses the GitHub API:

- review comments, reviews, and (via GraphQL) whether a review thread is
  resolved or outdated
- check runs and commit statuses
- merge conflicts and draft state

Configured bot logins default to the accounts that currently review
`BerriAI/litellm` (seen on recent pull requests such as #54722):

| Bot | GitHub login | How it shows up |
| --- | --- | --- |
| Cursor Bugbot | `cursor[bot]` | review body contains `<!-- BUGBOT_REVIEW -->`; stale passes contain `BUGBOT_REVIEW_STALE` |
| Greptile | `greptile-apps[bot]` | inline comments and a summary |
| Veria | `veria-ai[bot]` | inline comments and a summary |

Override the list with `REVIEW_BOT_LOGINS` if a login changes. A pull request
is **ready for maintainer ✅** only when every configured bot has reviewed the
latest commit, none of them have an unresolved current thread, CI is green,
the pull request is not a draft, and GitHub says it is mergeable. Anything
else is **needs work** with the specific items. The linked Discord account
for the GitHub author (from `/link-github`) is tagged. `REVIEW_MAINTAINER_ROLE_ID`
is pinged only on a ready verdict. Make that role mentionable, or the ping is
silent.

A written review uses `litellm.completion` and `REVIEW_MODEL` (for example
`openai/gpt-4o-mini`). Large diffs are split on file boundaries, capped, and
the model is told what it did not see. The note covers what the pull request
does, risks, missing tests, and a short checklist. If the SDK or the model
call fails, the gate still posts. Provider credentials are the normal LiteLLM
environment variables (`OPENAI_API_KEY`, and so on). They are not given new
names here.

Messages stay within Discord's 2000-character limit by splitting on line
breaks.

### Setup

1. Install the optional dependency and set the model key the provider expects:
   `pip install -r requirements.txt`. The Friday worker's build does not do
   this. A process that serves reviews needs that install, plus
   `python3 bot.py` on a public HTTPS port (the same web service as
   contributor roles is enough). Do not run two pollers on one state file.
2. Set `PR_REVIEW_ENABLED=true`, `REVIEW_MODEL`, and `REVIEW_CHANNEL_IDS`
   (comma-separated channel snowflakes). Set `DISCORD_GUILD_ID`,
   `DISCORD_APPLICATION_ID`, and `DISCORD_PUBLIC_KEY` if contributor roles
   are not already configured. The interactions endpoint is the same URL:
   `https://<public-host>/discord/interactions`.
3. Invite the bot with the `bot` and `applications.commands` scopes. In the
   review channels grant **View Channel**, **Read Message History**,
   **Send Messages**, **Create Public Threads**, and **Send Messages in
   Threads**. In the Developer Portal enable the **Message Content** intent.
   Slash commands registered for the guild are `/review-pr` and `/recheck`
   (plus the role commands, when those are enabled).
4. `GITHUB_TOKEN` should be able to read pull requests, reviews, and checks
   on `BerriAI/litellm`. Anonymous access works until the rate limit.
   GraphQL is used only to read review-thread resolution. The token does not
   need permission to write, approve, or merge.
5. Restart, then paste a pull request URL in a review channel or run
   `/review-pr`. The thread should show either `ready for maintainer ✅` or
   `needs work`.
6. For webhook re-checks, set `REVIEW_WEBHOOK_RECHECK=true` and
   `GITHUB_WEBHOOK_SECRET` (already required when contributor roles are on).
   On the existing GitHub webhook, keep **Pull requests** and also send
   **Pull request reviews**. Payload URL and HMAC check are unchanged.

`PR_REVIEW_ENABLED=false` leaves channels, slash commands, and webhooks
exactly as they were.

## Checks

Offline tests never post or use real credentials:

```sh
python3 -m unittest discover -v
```
