# Security and deployment boundaries

## Threat model

A compromised repository must not automatically turn into a production deploy.
The primary boundary is **outside the repository**, in Render's control plane.
CI, CODEOWNERS, and request validation help catch mistakes; none of them can
contain arbitrary malicious Python code after an administrator deploys it.
An attacker controlling both GitHub administration and Render administration
can remove these controls. Use distinct, MFA-protected identities and narrowly
scoped access; add an independent reviewer where practical.

## Render controls

- Keep **Auto-Deploy: Off** on the worker. Manually deploy a reviewed **full
  commit SHA**, not an unreviewed moving branch head. A passing CI result is
  not a malware verdict.
- Keep **Pull Request Previews: Off**. Never expose production secrets to PR
  previews, build jobs, or untrusted code.
- If a Blueprint is ever connected, set **Blueprint → Settings → Auto Sync:
  No**. `autoDeployTrigger: off` in `render.yaml` is not enough on its own:
  someone who changes that file could otherwise overwrite dashboard settings.
- Use a dedicated environment containing only this bot, with **Block
  cross-environment connections** enabled. Restart the worker after enabling
  isolation to close old connections. This blocks private-network access to
  other environments, not arbitrary public internet access.
- Do not share environment groups, database credentials, cloud credentials,
  organization tokens, Render API keys, SSH keys, or deploy hooks with the bot.
  Do not put deployment credentials or hooks in GitHub Actions secrets either.
- Keep the deploy-hook URL private and unused. If it has been exposed or used
  by an old automation, rotate it from Render and remove the old integration.
- Keep **ANNOUNCEMENTS_ENABLED=false** until the private delivery test passes.
  This is a posting pause, not a security sandbox: it does not stop malicious
  replacement code from running during startup or a build.

The live worker is named `litellm-discord`. Do not deploy a second worker from
the Blueprint. Match the existing service name and keep its state disk.

## Credentials and upstream write access

The deployed bot does not need GitHub credentials to read public LiteLLM PRs.
**Prefer no `GITHUB_TOKEN` at all.** Anonymous GitHub API rate limits make
backfills slower; the poller rotates pending work to avoid starving later PRs.

If a token becomes necessary, use a dedicated, expiring fine-grained token
with only the minimum public-repository read access. No repository write,
workflow, administration, organization, or account permissions. Do not reuse
your personal development PAT or Render's Git integration credentials. A
variable named `GITHUB_TOKEN` does **not** make its contents read-only; its
permissions must be configured at GitHub. The bot cannot prove a fine-grained
token's permissions just by inspecting the token string.

The Discord bot should have View Channel / Read Message History only in the
source channel, and View Channel / Send Messages only in #announcements and
#moderators-only. Do not grant Administrator, Manage Server, Manage Roles,
Manage Webhooks, or Mention Everyone. A stolen Discord token can still act
within those granted Discord permissions.

## GitHub controls

- Protect `main`: require pull requests and the two CI test checks, require
  the branch to be up to date, resolve review conversations, disallow force
  pushes and deletion, and enforce the rules for administrators.
- The repository currently has one collaborator. Requiring an independent
  approval would prevent that owner from merging their own PRs. Add another
  trusted collaborator and then require at least one approval, code-owner
  review, stale-review dismissal, and approval after the last push. CODEOWNERS
  alone is **not** an enforced approval rule.
- Enable secret scanning and push protection; review alerts without copying
  secret values into issues, logs, or chat. Require approval for workflow runs
  from all outside contributors.
- Restrict Actions to the two official actions used by CI and require full
  commit-SHA pins. CI uses GitHub-hosted ephemeral runners, read-only token
  permissions, no persisted checkout credentials, and no deployment job.
- No `pull_request_target`, privileged untrusted checkout, self-hosted runner,
  production environment, or deployment secrets are needed for tests.

Repository administrators can alter repository protections. Render's manual
deployment gate remains necessary even with all of these controls enabled.

## Application and build controls

- Standard library only. The Render build compiles Python syntax without
  installing packages or running repository tests with production secrets.
- Only HTTPS GitHub pull-request GETs and the necessary Discord channel/message
  requests are allowed by the HTTP client. Redirects and environment-proxy
  routing are disabled; API response bodies are bounded and not logged on errors.
- The bot never checks out a contributor's branch, downloads PR files, executes
  message content, or writes to GitHub. Mention allowlists never permit mass pings.
- State is JSON, not executable serialization, and is written using exclusive
  temporary files with owner-only permissions and atomic replacement.
- Local secret files are ignored by Git. Ignore rules do not protect files
  already committed; GitHub push protection is an additional independent layer.

These in-process restrictions prevent accidental or input-driven misuse;
malicious replacement code can bypass them. For a stronger future boundary,
build vetted images in a separate trusted pipeline and deploy immutable image
digests with provenance verification outside this repository.

## Deployment procedure

1. Review the full code and configuration diff; pay particular attention to
   workflows, `render.yaml`, dependencies, and outbound requests.
2. Merge via a PR after the required checks pass. Verify the exact commit SHA.
3. Verify Render auto-deploy and PR previews remain off, and Blueprint auto-sync
   is off if one is configured. Confirm no upstream-write/deployment tokens
   or unrelated environment groups have been added.
4. Manually deploy the reviewed SHA. Keep automatic contributor posts paused
   until the existing `--test-channel-id` delivery test passes.

## Reporting and response

Use the repository's private GitHub vulnerability-reporting channel when
available, or contact the owner privately. Do not put credentials in a public
issue. For an actual compromise: suspend the worker, disable deploys/syncs,
revoke exposed credentials at their providers, review GitHub/Render audit
events and state, and redeploy a verified clean commit with fresh credentials.

References: [Render deployment controls](https://render.com/docs/deploys),
[Blueprint auto-sync](https://render.com/docs/infrastructure-as-code#disabling-automatic-sync),
[private-network isolation](https://render.com/docs/private-network),
[GitHub branch protections](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
