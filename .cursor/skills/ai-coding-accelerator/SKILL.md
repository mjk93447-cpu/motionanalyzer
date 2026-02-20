---
name: ai-coding-accelerator
description: Optimizes agent behavior with three tools: Skills, shell tools, and server-side compaction workflow. Use when implementing features, debugging, or running multi-step coding tasks in this repository.
---

# AI Coding Accelerator

## When to Use

Use this skill for normal coding sessions in this project, especially when tasks involve code edits, test execution, and iterative debugging.

## Instructions

1. Load relevant project rules first, then plan changes in short steps.
2. Prefer deterministic execution:
   - Run commands through shell tools in PowerShell.
   - Keep commands explicit and reproducible.
3. Use shell tools for verification checkpoints:
   - dependency install
   - tests
   - lints or static checks
4. Keep context compact during long sessions:
   - after major milestone completion
   - after large error logs are resolved
   - before starting a new sub-task
5. Maintain stable processing standards for this repo:
   - sort frame filenames before parsing
   - document units for vector metrics (pixel, pixel/s, pixel/s^2)
   - prefer typed functions and deterministic numeric logic

## Shell Tool Defaults (Windows)

- Shell: `powershell`
- Workspace root: `c:\motionanalyzer`
- Use quoted paths when containing spaces.
- Record key command outputs in concise notes for traceability.

## Server-Side Compaction Playbook

Apply compaction-aware workflow in long chats:

- Keep prompts task-scoped (one target per request).
- Summarize resolved branches before continuing.
- Start a fresh thread when context becomes noisy.
- Preserve durable instructions in project rules/skills instead of repeated chat history.
