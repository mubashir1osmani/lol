# LiteLLM release Discord bot

This Python bot checks [BerriAI/litellm releases](https://github.com/BerriAI/litellm/releases) every five minutes and posts new releases as Discord embeds. It uses only Python's standard library, so there are no packages to install and no privileged gateway intents to enable.

## Set up Discord

1. In the [Discord Developer Portal](https://discord.com/developers/applications), open your application and go to **Bot**.
2. Copy or reset the bot token. Never post or commit this token.
3. Go to **OAuth2 > URL Generator**, select the `bot` scope, then select these permissions:
   - View Channels
   - Send Messages
   - Embed Links
4. Open the generated URL and add the bot to your server.
5. In Discord, enable **User Settings > Advanced > Developer Mode**. Right-click the destination channel and choose **Copy Channel ID**.

## Run it

Requires Python 3.10 or newer.

Create a `.env` file containing `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID`, then run:

```sh
python3 bot.py
```

On its first run, it posts the latest release as a connection check. It then records fetched release IDs in `.data/releases.json` and only posts newly published releases. Set `POST_LATEST_ON_STARTUP=false` if the initial post is unwanted. Set `INCLUDE_PRERELEASES=false` to post stable releases only.

Keep the process running with your preferred process manager or hosting service. No inbound port is needed.

### Render

The included `render.yaml` runs the bot as a background worker. Create a new Blueprint from this repository, then enter `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` when Render asks for them. Render background workers may require a paid instance.

## Test

```sh
python3 -m unittest discover -v
```
