# LiteLLM Discord agent bot

An AI-agent Discord bot for the LiteLLM community. It:

- **Posts GitHub releases** from [BerriAI/litellm](https://github.com/BerriAI/litellm/releases) as Discord embeds, with a Claude-written digest of the release notes.
- **Shares new blog posts** from [docs.litellm.ai/blog](https://docs.litellm.ai/blog) (via its RSS feed), with a Claude-written blurb.
- **Shares social media updates** from any RSS/Atom feeds you configure.
- **Adds you to new support threads** the moment they're created, and (optionally) posts a Claude-written first reply acknowledging the question.
- **Pings a role (or @everyone)** on announcements so updates reach the right people.

Claude is optional: without an LLM key, the bot posts raw release notes and feed summaries instead of AI-written digests, and skips thread greetings. If a Claude request ever fails, the bot falls back to the raw content rather than dropping the update. LLM calls go through a LiteLLM proxy when `LITELLM_PROXY_BASE_URL` is set, or straight to Anthropic otherwise.

## Set up Discord

1. In the [Discord Developer Portal](https://discord.com/developers/applications), open your application and go to **Bot**.
2. Copy or reset the bot token. Never post or commit this token.
3. Go to **OAuth2 > URL Generator**, select the `bot` scope, then select these permissions:
   - View Channels
   - Send Messages
   - Send Messages in Threads
   - Embed Links
   - Read Message History (for thread greetings)
   - Mention Everyone (only if you use `MENTION_EVERYONE` or a non-mentionable role)
4. Open the generated URL and add the bot to your server.
5. In Discord, enable **User Settings > Advanced > Developer Mode**. Right-click the destination channel and choose **Copy Channel ID**. The same right-click menu gives you server, role, and user IDs.

## Configure

Create a `.env` file. Only the first two variables are required.

| Variable | Required | What it does |
| --- | --- | --- |
| `DISCORD_BOT_TOKEN` | yes | Bot token from the developer portal. |
| `DISCORD_CHANNEL_ID` | yes | Channel for release announcements (and updates, unless overridden). |
| `LITELLM_PROXY_BASE_URL` | no | LiteLLM proxy URL for LLM calls (e.g. `https://proxy.example.com`). |
| `LITELLM_PROXY_API_KEY` | no | API key for the proxy. Enables the Claude agent (digests, blurbs, thread greetings). |
| `ANTHROPIC_API_KEY` | no | Alternative to the proxy key: call Anthropic directly. |
| `CLAUDE_MODEL` | no | Model name to request. Default: `claude-sonnet-4-6`. |
| `DISCORD_GUILD_ID` | for threads | Your server ID; needed to watch for new threads. |
| `SUPPORT_USER_ID` | for threads | User to add to every new thread (you). |
| `SUPPORT_CHANNEL_IDS` | no | Comma-separated channel IDs; only threads under these parents count. Default: all threads. |
| `AGENT_REPLIES_IN_THREADS` | no | Set `false` to add you silently without a Claude greeting. Default: `true`. |
| `UPDATES_CHANNEL_ID` | no | Separate channel for blog/social posts. Default: `DISCORD_CHANNEL_ID`. |
| `BLOG_FEED_URL` | no | Default: `https://docs.litellm.ai/blog/rss.xml`. Set empty to disable. |
| `SOCIAL_FEED_URLS` | no | Comma-separated RSS/Atom feeds for social updates (e.g. an X/Twitter RSS bridge). |
| `MENTION_ROLE_ID` | no | Role ID to ping on release/blog/social announcements. |
| `MENTION_EVERYONE` | no | Set `true` to ping `@everyone` instead. Prefer a role — @everyone fatigue is real. |
| `GITHUB_TOKEN` | no | Raises GitHub's rate limit. |
| `POLL_INTERVAL_MINUTES` | no | Release check cadence. Default: 5. |
| `FEED_POLL_INTERVAL_MINUTES` | no | Blog/social check cadence. Default: 30. |
| `THREAD_POLL_INTERVAL_MINUTES` | no | New-thread check cadence. Default: 1. |
| `INCLUDE_PRERELEASES` | no | Default: `true`. |
| `POST_LATEST_ON_STARTUP` | no | Post the latest release on first run. Default: `true`. |

> **Role vs. @everyone:** prefer a dedicated, self-assignable role like `@litellm-updates` (`MENTION_ROLE_ID`) — people opt in and you never ping the uninterested. Use `MENTION_EVERYONE=true` only for small servers.

## Run it

Requires Python 3.10 or newer. The Claude agent needs the `anthropic` package:

```sh
pip install -r requirements.txt
python3 bot.py
```

On its first run it posts the latest release, then seeds state files under `.data/` (release IDs, seen feed entries, joined threads) so it only reacts to new activity afterwards.

### Render

The included `render.yaml` runs the bot as a background worker with a 1 GB persistent disk mounted at `.data/`, so state survives redeploys and nothing gets re-posted. Create a new Blueprint from this repository and fill in the environment variables when Render asks. Render background workers may require a paid instance.

## Test

```sh
python3 -m unittest discover -v
```
