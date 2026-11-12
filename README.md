# LiteLLM contributor thank-you bot

One feature: read PR links from **#oss-contributions**, check whether those PRs
have **actually merged on GitHub**, and thank their contributors by Discord
mention in **#announcements**. No release announcements, feeds, support-thread
automation, or LLM calls.

## How it works

1. On first run, read all accessible message history in #oss-contributions,
   paginating beyond Discord's 100-message limit. Later runs read new messages.
2. Extract links to PRs in `BerriAI/litellm` (including Markdown links and link
   previews). Ignore other repositories, issues, bot messages, and webhooks.
3. Check each PR using GitHub's pull-request API. Only `merged: true` qualifies;
   closed-but-unmerged PRs do not. Keep rechecking unmerged PRs even when their
   original Discord messages are old, so later merges are picked up.
4. Group newly merged PRs by contributor and post a thank-you list mentioning
   each Discord ID once per batch. Large lists are split to respect Discord's
   message and mention limits. Only those users can be pinged, never roles or
   @everyone. The first run **does thank contributors for historical merged PRs**.
5. Save all tracked PRs, an announcement outbox, and the cumulative unique
   `contributor_ids` list in `.data/contributors.json`. Previously announced PRs
   stay deduplicated across polls and restarts. A contributor can be thanked
   again when another of their PRs merges.

By default, the **first human Discord member to share a PR** is credited. This is
an attribution assumption, not proof that the Discord and GitHub accounts belong
to the same person. If people share others' PRs, use `GITHUB_DISCORD_MAP` to map
the actual GitHub PR author to their Discord ID. Explicit mappings take
precedence over the sharer. For verified/manual attribution only, set
`CONTRIBUTOR_ATTRIBUTION=github_mapping`; unmapped merged PRs remain pending until
you add the mapping and restart the bot. Display names are never used for matching.

The source must be a **text channel**. This bot reads its messages, not messages
inside separate threads or forum posts. Edits to already-scanned messages are
not rescanned; share added PR links in a new message.

## Set up Discord

1. In the [Discord Developer Portal](https://discord.com/developers/applications),
   open your application, go to **Bot**, and copy/reset the bot token.
2. Enable **Message Content Intent** under **Privileged Gateway Intents**.
   Discord also applies this restriction to REST history requests: without it,
   ordinary messages' content and embeds may be empty. Verified bots may need
   Discord approval for this intent.
3. Add the bot to your server with the `bot` OAuth2 scope. Grant:
   - **View Channel** and **Read Message History** in #oss-contributions.
   - **View Channel** and **Send Messages** in #announcements.
4. Enable **User Settings → Advanced → Developer Mode** in Discord. Right-click
   #oss-contributions and #announcements and choose **Copy Channel ID** for each.
   Channel names alone are not enough. The same menu on a user provides their ID.

## Configure

Copy `.env.example` to `.env` and fill in the three required values:

```dotenv
DISCORD_BOT_TOKEN=your-bot-token
OSS_CONTRIBUTIONS_CHANNEL_ID=123456789012345678
ANNOUNCEMENTS_CHANNEL_ID=234567890123456789
```

| Variable | Required | Meaning |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | yes | Discord bot token. |
| `OSS_CONTRIBUTIONS_CHANNEL_ID` | yes | Source #oss-contributions text-channel ID. |
| `ANNOUNCEMENTS_CHANNEL_ID` | yes | Destination #announcements channel ID. |
| `ANNOUNCEMENTS_ENABLED` | no | `false` keeps the worker idle without scanning or changing state. Default without the variable: `true`; the Render blueprint and `.env.example` explicitly set `false` for initial testing. |
| `GITHUB_TOKEN` | recommended | GitHub token for API rate limits. No write access needed for the public LiteLLM repository. |
| `GITHUB_REPOSITORY` | no | Default: `BerriAI/litellm`. |
| `POLL_INTERVAL_MINUTES` | no | Check cadence. Default: `5`. |
| `STATE_FILE` | no | Default: `.data/contributors.json`. Must live on persistent storage. |
| `GITHUB_DISCORD_MAP` | no | JSON object mapping GitHub usernames to Discord ID **strings**, e.g. `{"octocat":"345678901234567890"}`. Username matching is case-insensitive. |
| `CONTRIBUTOR_ATTRIBUTION` | no | `message_author` (default) or `github_mapping` (requires a nonempty map). |

Without a GitHub token, GitHub's anonymous API limit can make a large initial
backfill slow. Successfully checked merges are retained, and pending PRs are
retried after API failures. No GitHub username or Discord user ID is inferred
from a display name.

## Run

Requires Python 3.10+; no third-party dependencies:

After testing, set `ANNOUNCEMENTS_ENABLED=true` to enable automatic announcements.

```sh
python3 bot.py
```

Example announcement:

```text
🎉 Thank you for contributing to LiteLLM!

These contributors had 3 PRs shared in #oss-contributions merged on GitHub:

- @contributor-one
- @contributor-two

Your contributions make LiteLLM better for everyone. 💚
```

The real post uses `<@DISCORD_ID>` mentions. The machine-readable cumulative
list of thanked Discord IDs is the `contributor_ids` array in the state file.

Run **one worker** per state file and keep that file across redeploys. The outbox
is saved before sending; successful batches are saved individually. Discord's
`nonce`/`enforce_nonce` deduplication also protects immediate send retries.
Discord only remembers nonces for a few minutes, so a crash after a successful
send but before saving its receipt can still duplicate that batch after a long
downtime. Deleting the state intentionally causes a full historical reannouncement.

### Render / upgrading the old bot

The included `render.yaml` runs a background worker with a 1 GB persistent disk
at `.data/`. Its existing service name is retained so a Blueprint update targets
the old worker rather than creating a second bot. Set the new source and
destination channel variables and redeploy that worker; no separate worker is
needed. The bot does not run until those new variables are configured.

Old release/feed/thread state files are unused and can be left in place. Old
settings such as `DISCORD_CHANNEL_ID`, `SUPPORT_USER_ID`, `BLOG_FEED_URL`,
`MENTION_EVERYONE`, and LLM credentials are no longer read and can be removed
from the deployment. If `STATE_FILE` was customized for release state, unset it
or point it to a new contributor state file; incompatible state is rejected,
never silently reset. Keep the contributor state when redeploying.

### First deployment: test in #moderators-only

1. Deploy the current code as a Render **Background Worker** using the included
   Blueprint. Fill in the bot token, source/destination channel IDs, and GitHub
   token. Keep `ANNOUNCEMENTS_ENABLED=false`; the worker will log that it is
   paused and will not post historical thank-yous to #announcements.
2. Grant the bot **View Channel** and **Send Messages** in #moderators-only.
   Copy that channel's ID from Discord's Developer Mode menu.
3. In the service's Render **Shell**, run the following, replacing the placeholder
   with that channel ID:

   ```sh
   python3 bot.py --test-channel-id MODERATORS_ONLY_CHANNEL_ID
   ```

   This command needs only `DISCORD_BOT_TOKEN`. It checks the channel name, sends
   one clearly labeled delivery test with all mentions disabled, prints the
   Discord message link, and exits. It does not scan GitHub, run the worker, or
   read/write contributor state. Do not change the worker's Start Command to
   this one-shot command, or Render may restart it and send repeated tests.
4. When ready for real announcements, change `ANNOUNCEMENTS_ENABLED` to `true`
   in Render and in `render.yaml` so future Blueprint syncs keep it enabled,
   then redeploy. The first active poll includes historical merged PRs.

The normal Start Command remains `python3 bot.py`. Changes must be committed
and pushed to Render's connected branch before Render can deploy them.

## Test

Tests use fake Discord/GitHub responses and never send real messages:

```sh
python3 -m unittest discover -v
```
