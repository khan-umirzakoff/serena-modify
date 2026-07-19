---
name: serena-coding-harness
description: Build and verify Serena MCP coding-harness features with Codex-style discipline.
---

Use this skill when modifying Serena MCP tools, workflow contracts, validation helpers, or coding-harness behavior.

Workflow:
1. Start with `prepare_coding_task` for the relevant module or test path.
2. Inspect current code and tests before editing.
3. Prefer small typed helper functions and dataclasses for durable contracts.
4. Preserve existing tool schemas unless the task explicitly requires new tools.
5. Add focused tests for every new contract field or tool behavior.
6. Run the smallest relevant validation command first, then broader checks when practical.

Safety rules:
- Do not execute skill scripts unless the skill policy allows it and the script was inspected.
- Do not treat skill metadata as full instructions; this file is the authoritative instruction body.
- If required tools are missing, report the missing dependency rather than guessing.
