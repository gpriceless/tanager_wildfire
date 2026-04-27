# OpenSpec Process Guide

> How to use OpenSpec for spec-driven development in the Tanager Competition project.

---

## Overview

OpenSpec is the spec-driven development workflow for this project. It maintains separation between:

- **specs/** - TRUTH: What IS built (current capabilities)
- **changes/** - PROPOSALS: What SHOULD change (pending work)
- **archive/** - COMPLETED: Past changes that have been deployed
- **deferred/** - PARKED: Changes postponed for later consideration

---

## Quick Reference

| Task | Command |
|------|---------|
| List active changes | `openspec list` |
| List specs | `openspec list --specs` |
| View a change | `openspec show <change-id>` |
| Validate before handoff | `openspec validate <change-id> --strict` |
| Archive after deployment | `openspec archive <change-id> --yes` |

---

## Lifecycle: Proposal to Archive

```
1. DISCOVER
   |-- Read openspec/project.md for context
   |-- Run openspec list --specs to see capabilities
   |-- Run openspec list to see active changes
   |
2. SCAFFOLD
   |-- Choose unique verb-led change-id
   |-- Create openspec/changes/<change-id>/
   |-- Write proposal.md (Why, What, Impact)
   |
3. SPECIFY
   |-- Write spec deltas in specs/<capability>/spec.md
   |-- Use ADDED/MODIFIED/REMOVED headers
   |-- Include #### Scenario: for each requirement
   |
4. DESIGN (if needed)
   |-- Write design.md for technical decisions
   |-- Include context, goals, alternatives, risks
   |
5. TASK
   |-- Write tasks.md with implementation checklist
   |-- Group by phase or track
   |
6. VALIDATE
   |-- Run openspec validate <change-id> --strict
   |-- Fix any issues before proceeding
   |
7. IMPLEMENT
   |-- Follow tasks.md sequentially
   |-- Mark tasks complete as you go
   |-- Use /run-phase for execution
   |
8. REVIEW
   |-- QA validates implementation matches spec
   |-- Wave gates at section boundaries
   |
9. ARCHIVE
   |-- Apply deltas to canonical specs
   |-- Move to openspec/archive/
```

---

## Integration with Plane

- One Plane issue per OpenSpec change
- Updated at section boundaries, not per-task
- Link Plane issue ID in proposal.md frontmatter

## Integration with Product Memory

- Product Queen updates `docs/product-memory.md` after each change
- Engineering Manager updates `docs/engineering-memory.md` for architecture changes
- Research findings tracked in `docs/research-memory.md`
