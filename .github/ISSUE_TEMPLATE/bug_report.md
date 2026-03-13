---
name: Bug report
about: Create a report to help us reproduce and fix a bug
title: "[bug] "
labels: bug, needs-triage
assignees: ''
---

<!--
Thank you for filing a bug report. Use this template to provide the minimal,
reproducible information required for maintainers to triage and fix the issue.

Before filing:
- Search existing issues to avoid duplicates.
- If the bug is a build failure, try `cargo clean && cargo check -v` and include the full output.
- If the bug involves a third-party native dependency, note your OS/toolchain and whether you used MSVC or GNU toolchain on Windows.

Be concise but precise — include commands you ran, exact error output, and the smallest reproduction you can provide.
-->

### Summary
A short one-line summary of the problem (what happened).

### What I expected
Describe the expected behavior (what you wanted to happen).

### What actually happened
Describe the actual behavior. Include error messages, stack traces, or screenshots (if helpful).

### Reproduction steps
Provide a minimal, step-by-step set of commands or code required to reproduce the bug.

1. Clone the repo at commit: `<commit-or-branch>`
2. Run: `...`
3. Observe: `...`

If the bug only occurs under specific features, include the exact cargo command and flags:
- Example: `cargo check --features dynamic-linking,importance-decay -v`

### Platform & toolchain
Provide OS and toolchain specifics so maintainers can reproduce the environment:

- OS: e.g. Windows 11, Ubuntu 22.04, macOS 14
- Toolchain: `rustc --version` and `cargo --version`
- On Windows: indicate whether you used MSVC or GNU toolchain (e.g., `stable-x86_64-pc-windows-msvc`)
- Any other relevant installed tools (e.g., Visual Studio Build Tools, MSYS2, MSVC version)

### Logs & error output
Paste full, verbatim error output (use triple backticks). Include `-v`/`--verbose` outputs where applicable.

```
<paste full error build/test/log output here>
```

If the output is long, include the most relevant excerpt and attach full logs if possible.

### Minimal repro (recommended)
If possible, attach a minimal repro project or a Gist with the smallest code example that reproduces the bug. Steps:

- Create a tiny reproduction (preferably a GitHub repo or gist).
- Include the exact commands to run.
- Attach or link the reproduction here.

### Debugging you tried
List the steps you already tried to fix or further isolate the issue (helps avoid duplicate efforts):

- `cargo clean` and re-built
- toggled features X and Y
- tried GNU toolchain vs MSVC
- checked for similar issues in upstream dependency Z

### Impact / Priority
How severe is this bug for you?
- P0 — blocks development or CI for this repo
- P1 — important but has a workaround
- P2 — minor annoyance
Please pick one and explain briefly.

### Suggested diagnosis / potential causes (optional)
If you have an idea of the root cause or where to look, mention it. This speeds up triage.

### Attachments
- Add any relevant files, screenshots, or logs as attachments to the issue.

---

Developer checklist (for triage)
- [ ] Issue has a descriptive title
- [ ] Reproduction steps provided
- [ ] Environment and toolchain provided
- [ ] Full verbose logs attached or pasted
- [ ] Minimal repro attached or linked (if feasible)

If this is a regression, include:
- The last known-good commit or version, and what changed since then.

Thank you — a maintainer will triage this soon. If the bug is security-sensitive, do NOT post secrets here; instead contact maintainers privately as described in the repository's SECURITY policy (if present).