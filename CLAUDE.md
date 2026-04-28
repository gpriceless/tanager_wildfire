<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Use `@/openspec/PROCESS.md` to understand:
- End-to-end lifecycle from proposal to archive
- Integration with Plane issue tracking
- Integration with product-memory.md updates

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Tanager Competition — Project Context

## Overview

Research and development project for the Planet Tanager Open Data Competition. Focus: wildfire analysis using 426-band hyperspectral imagery from Tanager-1.

**Competition deadline:** August 31, 2026
**Submission focus:** FireSpec — burn severity mapping + live fuel moisture estimation

## Delegation Rules (NON-NEGOTIABLE)

- **Engineering Manager is a planner, not an executor.** EM validates OpenSpec, enriches tasks.md, and returns READY/GAPS. EM does not spawn agents or write source code.
- **No Project Manager.** PM role is absorbed by EM (enriches tasks.md) + MCP workflow server (task management).
- **All phase work tracked through Plane at phase level.** One Plane issue per OpenSpec change, updated at section boundaries — not per-task tickets.
- **Full protocol:** `~/.claude/docs/AGENT_TEAMS_PROTOCOL.md`

## OpenSpec Execution Rules (NON-NEGOTIABLE)

When executing an OpenSpec change (the Research → PQ → EM → Execute pipeline), the implementation phase MUST use `/run-phase <change-id>`. No exceptions.

**The pipeline:**
1. PQ creates OpenSpec (proposal + spec + tasks.md)
2. EM enriches tasks.md (execution markers, file refs, gotchas) → returns READY
3. CTO invokes `/run-phase <change-id>` — this is the ONLY way to execute
4. MCP workflow drives: `next_task` → coder → QA → `complete_task` → repeat
5. Wave gate (qa-master) at every section boundary
6. Phase gate (qa-master) before completion
7. EM audit of all changes before merge to main

**What `/run-phase` provides that raw coder agents do not:**
- Task-level state tracking (in-progress, completed, blocked)
- QA validation after each task — coder does not test its own work
- Wave gates that catch problems before they compound across sections
- Auditable progress visible mid-execution
- Smaller, focused coder sessions instead of long marathons

**What is NOT covered by this rule:**
- Ad-hoc bug fixes, QA fixes, quick patches
- Research tasks (Tobler operates independently for research)
- One-off file changes outside OpenSpec

## Agent Working Memories

**IMPORTANT:** Before building features or refactoring, check these documents:

| Agent | Memory File | Purpose |
|-------|-------------|---------|
| Product Queen | `docs/product-memory.md` | Features, specs, roadmap |
| Engineering Manager | `docs/engineering-memory.md` | Architecture, tech debt, patterns |
| Tobler / Researchers | `docs/research-memory.md` | Research findings, literature, experiments |

## Key Agents

| Agent | Role | Responsibility |
|-------|------|----------------|
| Tobler | Researcher | Hyperspectral analysis, literature review, spectral science |
| Product Queen | Lead | Synthesize research into specs, product direction |
| Crenshaw (EM) | Planner | Validate specs, enrich tasks.md, return READY/GAPS — does NOT spawn coders or write code |
| CTO | Orchestrator | Invokes `/run-phase`, spawns coders/QA/qa-master, drives the workflow loop |

## Tech Stack

- **Python 3.10+** — primary language
- **spectral (SPy)** — spectral analysis (MESMA, SAM, band math)
- **HyperCoast** — Tanager data I/O
- **rasterio** — raster I/O
- **xarray** — N-dimensional arrays
- **geopandas** — vector operations
- **Jupyter** — competition deliverables

## Code Quality Rules

- **Research-first:** Prioritize correctness and reproducibility over performance
- **Notebook-friendly:** All core functions should be importable and usable in Jupyter
- **Type hints:** Required on all public functions
- **Docstrings:** Google style, include parameter units (e.g., wavelength in nm)
- **Tests:** Validate against known spectral signatures

## Plane Update Protocol

**When working on any feature or task for Tanager, you MUST keep Plane in sync.**
**This is NON-NEGOTIABLE for coder and QA agents.** Update Plane as you go, not at session end.

### Plane Access
- **Workspace:** gabriel-dev
- **Project ID:** `9e01aeb2-23af-4a22-8e33-4b9443a85d9b`
- **Identifier:** TANAGER
- **State IDs:**
  - Backlog: `f4de3802-9e8d-46db-bc03-24e0f7af0a29`
  - Todo: `93791415-061e-4582-97ae-96f26120b193`
  - In Progress: `8e90b4ee-1df1-4e82-8d38-ea384c541693`
  - Structured: `58225f69-1f69-495e-b43e-764557ef2b49` (scaffolded)
  - Wired: `03815f7f-f284-4c3f-8c82-a95113d69270` (data hooked)
  - Functional: `a6c35e31-bbc9-4dea-bc97-47bb1c04bc96` (works end-to-end)
  - Verified: `aec29d32-1f99-48eb-8233-02d9b2f71ff0` (qa-master gate passed)
  - Done: `61fcc20e-b337-4aaf-8ab4-0e7aa3d1ae12`
  - Cancelled: `6227dbab-8d45-40aa-8f03-619a7579b0b2`

### Updates
- **One Plane issue per OpenSpec change** — phase level, not per-task
- **`/run-phase` creates the Plane issue automatically** via `workflow__start_phase`
- **MCP workflow posts section summaries** at wave-gate boundaries
- **Status progression:** Backlog → Todo → Structured → Wired → Functional → Verified → Done
- **For ad-hoc work** (bug fixes, research): create Plane issue manually via Plane MCP

### MCP Tools
Load via `ToolSearch("plane")`:
- `mcp__plane__list_work_items` — view current Tanager issues
- `mcp__plane__retrieve_work_item` — issue details
- `mcp__plane__create_work_item` — new issue
- `mcp__plane__update_work_item` — status, labels, assignee
- `mcp__plane__create_work_item_comment` — work logs

**Plane should ALWAYS reflect reality. If code is done, Plane should say Done.**
