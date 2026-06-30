# Skills harness

Serena now has Codex-compatible skill discovery and progressive disclosure.

Key contracts:
- Skills live under `.serena/skills`, scoped `.agents/skills`, or `~/.agents/skills`.
- A skill directory contains `SKILL.md` with YAML frontmatter `name` and `description`.
- Optional `agents/openai.yaml` can declare `short-description`, `policy`, `dependencies`, and `resources`.
- `discover_skills` and `prepare_coding_task` return metadata only; agents must call `read_skill` before following a skill body.
- Skill scripts/resources are never executed automatically.
- `dependency_report` surfaces missing Serena tools and declared MCP server dependencies.
- `create_directory` and `create_or_replace_file` provide safety-friendly file ops to avoid shell `mkdir`/heredoc workflows.

Dogfooding skill pack:
- `.agents/skills/serena-coding-harness/SKILL.md`
- `.agents/skills/serena-coding-harness/agents/openai.yaml`
- `.agents/skills/serena-coding-harness/references/workflow.md`
