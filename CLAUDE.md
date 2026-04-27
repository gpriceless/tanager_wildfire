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
- **All phase work tracked through Plane at phase level.** One Plane issue per OpenSpec change.
- **Full protocol:** `~/.claude/docs/AGENT_TEAMS_PROTOCOL.md`

## OpenSpec Execution Rules (NON-NEGOTIABLE)

When executing an OpenSpec change, use `/run-phase <change-id>`. No exceptions.

**The pipeline:**
1. PQ creates OpenSpec (proposal + spec + tasks.md)
2. EM enriches tasks.md → returns READY
3. CTO invokes `/run-phase <change-id>`
4. MCP workflow drives: `next_task` → coder → QA → `complete_task` → repeat
5. Wave gate (qa-master) at every section boundary

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
| Crenshaw (EM) | PM | Validate specs, enrich tasks, manage execution |

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

## Plane Integration

- **Project:** Tanager Competition
- **Workspace:** Managed via Paperclip API
- **Updates:** Phase-level, not per-task
