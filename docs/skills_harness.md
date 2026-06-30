# Serena Skills Harness

Serena supports Codex-compatible skills as local instruction packages.

## Skill layout

A skill is a directory containing `SKILL.md`. The instruction file must start with YAML frontmatter containing at least:

```yaml
---
name: example-skill
description: Short task-matching description.
---
```

Optional metadata can be declared in `agents/openai.yaml` next to `SKILL.md`:

```yaml
short-description: Compact display text
policy:
  allow_implicit_invocation: true
  allow_scripts: false
  allowed_tools:
    - read_file
dependencies:
  tools:
    - read_file
  mcp_servers:
    - SERENA_MCP_local
resources:
  references:
    - references/workflow.md
  scripts: []
  assets: []
```

## Discovery roots

Serena scans these roots:

- `<project>/.serena/skills`
- `<project>/.agents/skills`
- scoped `.agents/skills` directories from the project root to the active focus path
- `~/.agents/skills`

## Progressive disclosure

`discover_skills` and `prepare_coding_task` expose only skill metadata, dependency status, resources, policy, and the path to the instruction file. They do not inline full instructions.

Before using a skill, an agent must call `read_skill` for the skill name or instruction path. This mirrors Codex progressive disclosure and prevents skill bodies from flooding the initial context.

## Safety

Skill scripts and resources are discoverable but not executed automatically. Agents should inspect local resources before relying on them and should not run scripts unless the skill policy allows scripts and the script has been inspected.

Missing tool dependencies are surfaced in `dependency_report`. Missing MCP server dependencies are reported as declared-only metadata until explicit MCP server availability integration is added.

## Safety-friendly file operations

For workflows that would otherwise use shell commands such as directory creation or ad hoc file writes, Serena exposes dedicated tools:

- `create_directory`
- `create_or_replace_file`

These tools reduce shell-script payloads and make file operations easier for clients with strict safety filters.
