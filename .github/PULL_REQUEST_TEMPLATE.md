# Pull Request Template

Use this template for all pull requests. Fill in every section that applies — PRs missing required information may be requested for changes before review.

---
## Title
(Please use the commit-style header format)
`<type>(<scope>): <short description>`

Examples:
- `feat(vector): vendor usearch v2.24.0 and add MSVC shim`
- `fix(graph): guard table.len() import for redb`
- `docs!: add API surface section to PRD`

Supported commit types: `feat`, `fix`, `refactor`, `chore`, `test`, `docs`. Use `!` for breaking changes (e.g., `feat!`).

---

## What
One-sentence summary of what this PR does.

## Why
Why this change is needed. Explain the user-visible problem, CI/build requirements, or an architectural reason.

## Changes
- Short bulleted list of code/config/docs changes made by this PR.
- Include file-level highlights for non-obvious edits.

## Subsystems affected
List the subsystems touched (choose from the project subsystem list). Example:
- `vector` — usearch vendor + patch
- `graph` — redb table metadata
- `build/ci` — Cargo patching, vendored dependency

If many subsystems are affected, create a `PLAN.md` in the repo root before implementation and link it here.

## PLAN.md
Is this PR multi-file or touching >1 subsystem? If yes, a `PLAN.md` describing what/why/assumptions/out-of-scope must be present in the PR.
- [ ] I created `PLAN.md` with What / Why / Subsystems affected / Assumptions / Out of scope

(If you did not create `PLAN.md` and the change touches multiple files/subsystems, add it now and update the PR.)

## Assumptions
List any assumptions you made while implementing this change. Be explicit about environment, platform, or dependency assumptions.

## Out of scope
What this PR intentionally does not change.

---

## How to verify / Testing
Describe manual and automated steps reviewers should run to validate the change locally and in CI.

- Commands to run locally:
  - `cargo check -v`
  - `cargo test -v` (integration tests if relevant)
- Any environment variables, toolchain or platform requirements (e.g., MSVC vs GNU toolchain)
- If the PR vendors or patches a native dependency, include a short smoke test that exercises the native functionality.

## Backwards compatibility / Migration
If this PR introduces breaking changes, explain migration steps and how callers should migrate.

---

## Release Notes (1 sentence)
A single sentence suitable for a changelog entry that explains the user-facing change.

---

## Risks and rollback plan
- Describe potential risks (build, runtime, security) and how to mitigate them.
- If something goes wrong, how to revert safely (e.g., revert commit, remove `[patch.crates-io]`).

---

## CI / Observability
- [ ] CI job(s) updated (if build/test matrix changed)
- [ ] Tracing spans added for any cross-module operation (per repo observability rules)
- [ ] No hardcoded timeouts/thresholds — values must come from config

---

## Tests included
- Unit tests (list)
- Integration tests (list)
- Manual test steps
If no tests were added, explain why.

---

## Docs
- [ ] Documentation updated (README, PRD, API docs) where applicable
- Location of docs changes: `docs/`, `README.md`, `PRD.md`, or inline comments.

---

## Security / Privacy
- Any sensitive data handled? If yes, describe how it is protected.
- Confirm that no user data is logged in plaintext (per repository logging rules).

---

## Suggested rules additions (optional)
If this work exposed a gap in project rules, propose a short rule addition (one sentence) that would prevent the problem in future PRs.

---

## Reviewer checklist
- [ ] Does the PR have a clear What / Why?
- [ ] Does `cargo check` and `cargo test` pass locally?
- [ ] If vendoring/patching native deps: is an upstream PR planned and tracked?
- [ ] Did the author add/confirm `PLAN.md` if change touched >1 subsystem?
- [ ] Is the PR title and commit style correct?

---

Thank you — please wait for at least one approval before merging. If this is a release-blocking or CI-failing fix, tag a maintainer directly in the PR comments.