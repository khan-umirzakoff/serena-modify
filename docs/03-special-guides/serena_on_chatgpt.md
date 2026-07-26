
# Connecting Serena MCP Server to ChatGPT via Remote MCP

This guide explains how to expose Serena's native streamable-HTTP MCP endpoint through a Cloudflare Tunnel and connect it to ChatGPT.

Once configured, ChatGPT becomes a powerful **coding agent** with direct access to your codebase, shell, and file system — so **read the security notes carefully**.

---
## Prerequisites

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) 
and [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) installed.

## 1. Start the Serena MCP Server

For one ChatGPT chat at a time, start the server in switchable single-workspace mode:

```bash
uv run serena start-mcp-server \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 9121 \
  --project "$(pwd)" \
  --context chatgpt
```

`--project` selects only the initial project. `activate_project` may switch the server to another project later.

For independent parallel ChatGPT chats, use multi-workspace mode:

```bash
uv run serena start-mcp-server \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 9121 \
  --context chatgpt \
  --workspace-mode multi
```

Each chat calls `open_workspace` and keeps its returned `workspace_id`. Parallel edits to the same repository require separate Git worktree paths; sharing one working tree is rejected by default.

Harness runtime state is stored outside repositories under the user cache. Use `--state-dir PATH` to override it. Multi-workspace resource controls are available through `--workspace-ttl-seconds` and `--max-workspaces`.

## 2. Expose the Server Using Cloudflare Tunnel

Run:

```bash
cloudflared tunnel --url http://localhost:9121
```

This will give you a **public HTTPS URL** like:

```
https://serena-agent-tunnel.trycloudflare.com
```

Your server is now publicly reachable through HTTPS; authentication remains a separate deployment concern.

---

## 3. Connect It to ChatGPT

Add the public MCP endpoint as a remote MCP-backed ChatGPT plugin action:

```text
https://your-domain.example/mcp
```

After restarting or updating Serena, refresh the plugin actions so ChatGPT reloads the tool schemas. `get_current_config` reports an MCP schema fingerprint that helps confirm whether the client is using the current schema.

## Security Warning — Read Carefully

Depending on your configuration and enabled tools, Serena's MCP server may:
- Execute **arbitrary shell commands**
- Read, write, and modify **files in your codebase**

This gives ChatGPT the same powers as a remote developer on your machine.

### Key Rules

- Protect the public endpoint with the authentication layer used by your deployment.
- Only expose the server when needed, and monitor its use.

In your project’s `.serena/project.yml` or global config, you can disable tools like:

```yaml
excluded_tools:
  - execute_shell_command
  - ...
read_only: true
```

This is strongly recommended if you want a read-only or safer agent.


---

## Final Thoughts

With this setup, ChatGPT becomes a coding assistant **running on your local code** — able to index, search, edit, and even run shell commands depending on your configuration.

Use responsibly, and keep security in mind.
