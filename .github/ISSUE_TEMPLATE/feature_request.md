---
name: Feature request
about: Propose a new feature, enhancement, or improvement to ech0
title: "[feat] "
labels: enhancement, needs-triage
assignees: ''
---

<!--
Thank you for proposing a feature. Use this template to explain the request clearly:
- Keep it focused: one feature request per issue.
- Provide motivation, design ideas, and the acceptance criteria.
- If the change touches >1 subsystem or is non-trivial, create a `PLAN.md` before implementation and link it here.
-->

## Summary
One-line summary of the feature or enhancement.

## Motivation / Why
Why is this feature needed? Describe the user-facing problem, developer pain point, or CI/build need this solves. Explain who benefits and in what scenarios.

## Goals
List the concrete objectives for this feature (what success looks like). Example:
- Provide a stable API for X
- Improve performance of Y by N%
- Make ech0 work on platform Z

## Non-Goals
Explicitly state what this feature will not do. This prevents scope creep.

## Proposal / What
Describe the proposed solution at a high level:
- Key design decisions
- API changes (public types/functions) — include signatures when possible
- Feature flags that should gate the work (if applicable)
- Data model changes (if any)

If this is primarily a documentation change, describe the doc updates.

## Detailed Design (optional)
For larger features include one or more of:
- Sequence diagrams or step-by-step flows
- Data structures and fields
- Error handling behavior
- Observability (spans/events) to add
- Config knobs (toml) and default values

If you expect multiple implementation phases, outline them.

## Alternatives considered
List other approaches you evaluated and why they were rejected.

## Drawbacks / Risks
List potential downsides, trade-offs, or risks (compatibility, security, performance, maintenance). Suggest mitigation strategies.

## Backwards compatibility / Migration
If this changes public APIs, storage formats, or on-disk artifacts, describe:
- Whether it is backward-compatible
- Migration steps (if required)
- Upgrade plan for users/CI

## Dependencies
List any external dependencies or pre-reqs (other PRs, library updates, toolchain changes).

## Testing & Verification
Describe tests that will demonstrate correctness:
- Unit tests
- Integration tests (paths, example inputs)
- Manual test steps
- Performance benchmarks or metrics (if applicable)
- CI changes needed

Example commands:
- `cargo test -p ech0`
- `cargo check --features dynamic-linking,importance-decay -v`

## Docs
Describe documentation updates required (README, PRD, API docs). Where should they live (docs/, README.md, PRD.md)?

## Rollout Plan
If this requires staged rollout or feature flags, describe the plan:
- How to enable/disable
- Migration timing
- How to roll back

## Acceptance criteria
Concrete list of checks the reviewer can use to accept the feature:
- [ ] Design implemented per Proposal
- [ ] Tests added & passing
- [ ] No regressions in core test suite
- [ ] Documentation updated
- [ ] CI matrix updated (if needed)

## Implementation notes (for implementer)
- Suggested subsystems to touch (pick from repo docs): e.g., `vector`, `graph`, `search`, `store`, `linking`, `decay`, `conflict`, `provenance`
- If the change touches >1 subsystem, create `PLAN.md` with What/Why/Subsystems/Assumptions/Out-of-scope and link it here.

## Reference / Links
- Link to related issues, RFCs, design docs, or upstream PRs.

---

Developer checklist (for triage)
- [ ] Does the issue have a clear Summary, Motivation, and Proposal?
- [ ] Is scope limited to a single feature or well-scoped collection of changes?
- [ ] If multi-subsystem: Is `PLAN.md` present and referenced?
- [ ] Can reviewers reproduce the problem or verify the feature locally?
- [ ] Are risks and compatibility impacts documented?

Thank you — a maintainer will triage this soon. If this feature is security-sensitive or involves secret handling, do not post secrets here; contact maintainers directly per repository security guidance.