# Serena Coding Harness Skill Reference

This reference documents the intended use of Serena skills for coding-harness work.

- `discover_skills` lists only metadata, policy, resources, dependencies, and the instruction file path.
- `read_skill` loads the full instruction file before the assistant follows the skill.
- `prepare_coding_task` includes available skills and dependency reports so agents can decide whether a skill is relevant.
- Scripts and assets are discoverable resources; they are not executed automatically.
- Missing tool dependencies should be reported explicitly.
