# Tanager Open Data Competition — T&C Compliance Review

**Reviewed:** 2026-04-27
**Document:** Planet-TermsConditions-TanagerCompetition.pdf

---

## Summary

Our FireSpec submission (burn severity + LFMC) is **fully compliant** with the competition T&C. No structural changes to our approach are needed. Several tie-breaker opportunities are available that align with work we already planned.

---

## Key Terms

| Term | Detail | Our Status |
|------|--------|------------|
| **Deadline** | August 31, 2026 at 11:59 PM PST | Tracked (4 months remaining) |
| **Registration** | Online at competition website (opened April 14) | **ACTION: Must register** |
| **Submission portal** | surveymonkey.com/r/tanager-competition | Noted |
| **Max winners** | 3 | — |
| **Prize** | Seat on Planet's Open Committee; direct selection of 30 Tanager images for Open STAC catalog | Non-monetary; prestige + data influence |
| **Submissions per participant** | 1 (team members may also submit individually) | Plan: 1 team submission |
| **Language** | English only | Compliant |
| **Contact** | hyperspectral@planet.com | Noted |

---

## Eligibility Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Age 18+ | **Verify** | Participant must confirm |
| Not Planet employee/former (12 months) | **Verify** | Participant must confirm |
| Not immediate family of Planet employee | **Verify** | Participant must confirm |
| Employer/institution permits participation | **Verify** | Participant must confirm |
| Compliance with applicable laws (FCPA, sanctions, export controls) | **OK** | Standard U.S. research activity |

---

## Submission Requirements

### Required Components (must include at least one)

| Format | Max Length | Our Plan | Status |
|--------|-----------|----------|--------|
| Short technical memo / project summary | 1-3 pages | **Yes** — Project Summary & Impact Statement | Planned |
| Well-annotated Jupyter notebooks or GitHub repos | — | **Yes** — primary deliverable format | Planned |
| Slide decks or infographics | — | Optional stretch goal | — |
| Brief video walkthrough | < 3 minutes | Optional stretch goal | — |

### Optional Components (recommended for tie-breaker points)

| Component | Our Plan | Status |
|-----------|----------|--------|
| Public repository with documentation + reproducibility | **Yes** — GitHub repo | Planned |
| Data derivatives on Zenodo or open data platform | Consider hosting MESMA fraction maps | **Recommended** |
| Figures, maps, plots, or interactive maps | **Yes** — burn severity maps, time-series, LFMC maps | Planned |

### Third-Party Code/Libraries

> "You are allowed to use third-party code, libraries, SDKs, and APIs as long as these technologies comply with applicable law and the Participants have a license to use and distribute such third-party materials."

| Library | License | Distributable? | Status |
|---------|---------|---------------|--------|
| SPy (spectral) | MIT | Yes | OK |
| HyperCoast | MIT | Yes | OK |
| rasterio | BSD-3 | Yes | OK |
| xarray | Apache 2.0 | Yes | OK |
| geopandas | BSD-3 | Yes | OK |
| mesma | MIT | Yes | OK |
| spectral-libraries | MIT | Yes | OK |
| spyndex | MIT | Yes | OK |
| FRAMES SoCal Library | **Unknown** | **Verify** | **ACTION NEEDED** |
| USGS Spectral Library v7 | Public domain (USGS) | Yes | OK |
| ECOSTRESS | Public (NASA/JPL) | Yes | OK |
| Globe-LFMC 2.0 | CC BY 4.0 | Yes | OK |

**Risk:** FRAMES SoCal endmember library redistribution rights need verification. If restricted, we can reference it without redistributing the raw spectra, or use only USGS + ECOSTRESS + image-derived endmembers.

---

## Judging Criteria — Alignment Analysis

### Core Score (100 points)

| Category | Max Points | Our Alignment | Expected Score Range |
|----------|-----------|---------------|---------------------|
| **1. Scientific Integrity & Innovation** | 30 | **Strong.** MESMA methodology well-established (Quintano 2023). First satellite hyperspectral LFMC product = high novelty. Clear limitations assessment (GSD vs airborne, MESMA scalability). | 22-28 |
| **2. Application or Use Case** | 30 | **Strong.** Wildfire is high-value, clearly defined. LA fires = compelling case study. LFMC for fire risk = actionable for land managers. | 24-28 |
| **3. Workflow & Tool Development** | 20 | **Good.** Clean Python pipeline, STAC-native, reproducible Jupyter notebooks. Open-source potential. | 14-18 |
| **4. Visualization & Storytelling** | 20 | **Needs attention.** Must invest in compelling maps, before/after comparisons, time-series animation, narrative arc. This is 20% of the score. | 12-18 |
| **Estimated core range** | **100** | | **72-92** |

### Tie-Breaker Points (up to 15 bonus)

| Tie-Breaker | Points | Our Alignment | Status |
|-------------|--------|---------------|--------|
| Aligns with Planet's commercial verticals or strategic impact areas | +5 | **Yes.** Wildfire directly relates to biodiversity and environmental monitoring. | Eligible |
| Quantitative comparison of Tanager vs public alternatives (EMIT, PRISMA) | +5 | **Yes.** Already planned in research-memory.md — Tanager vs EMIT vs PRISMA vs Sentinel-2 accuracy comparison. | Planned |
| Open-source contribution or cutting-edge AI/ML application | +5 | **Possible.** Could package MESMA/LFMC pipeline as reusable library. Or include ML component (Random Forest for CBI regression). | **Recommended** |

**All 15 tie-breaker points are achievable.** The quantitative comparison is already part of our research plan. Open-source packaging requires intentional effort during Phase 4.

---

## IP & Licensing — What to Know

### What we keep
- **Full IP ownership** of our submission. Planet does not claim ownership beyond the license below.

### What Planet gets
- **Irrevocable, royalty-free, perpetual license** to:
  - Use, review, assess, test, and analyze the submission
  - Feature, display, and describe it in promotional materials (ads, press, trade shows)
- **No compensation** from Planet for promotional use
- Submission may be publicized in Planet's marketing materials

### What this means for us
- We can publish our research independently (papers, preprints)
- We can open-source our code
- We can use the work in other contexts
- Planet can showcase our submission without paying us
- **Patent risk:** Public disclosure of the submission may preclude patent rights — not relevant for our research-focused submission

### Tanager Data
- Planet retains all title and rights to Tanager Open STAC data
- We use it under the applicable Creative Commons license (CC BY 4.0)
- Must comply with CC license terms in our submission

---

## Withdrawal Rights

- Can withdraw at any time by emailing Planet
- If withdrawn before Final Evaluation: submission deleted, license terminates
- If withdrawn after Final Evaluation: license continues in perpetuity
- We should not withdraw if we've invested significant effort

---

## Action Items

| Priority | Action | Owner | Deadline |
|----------|--------|-------|----------|
| **HIGH** | Register for the competition online | Participant (human) | ASAP (registration open since April 14) |
| **HIGH** | Verify FRAMES SoCal library redistribution rights | Team | Before Phase 3 |
| **HIGH** | Plan visualization & storytelling strategy (20% of score) | Author | Phase 4 planning |
| **MEDIUM** | Plan Zenodo data derivative hosting | Team | Phase 4 |
| **MEDIUM** | Design open-source library packaging for +5 tie-breaker | Author | Phase 4 |
| **MEDIUM** | Include Tanager vs EMIT/PRISMA quantitative comparison | Team | Phase 3 |
| **LOW** | Consider 3-minute video walkthrough as optional deliverable | Author | Phase 4 |
| **LOW** | Confirm all eligibility requirements with participant | Participant (human) | Before submission |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| FRAMES library not redistributable | Medium | Fall back to USGS + ECOSTRESS + image-derived endmembers only |
| Visualization score low (20% of total) | Medium | Invest in Phase 4; leafmap interactive maps, matplotlib publication-quality figures |
| Late submission (deadline is hard) | High | Phase 4 packaging starts no later than mid-August |
| Planet modifies T&C before submission | Low | Check published version at submission time |

---

## Conclusion

Our FireSpec approach is well-positioned competitively. The judging criteria strongly favor our strengths (scientific methodology, real-world application, novel contribution). The main areas requiring deliberate investment are:

1. **Visualization & storytelling** (20% of score) — must be excellent, not an afterthought
2. **Tie-breaker points** — all 15 are achievable with intentional planning
3. **Registration** — must register before submitting

No compliance blockers identified. Proceed with Phase 2 build as planned.
