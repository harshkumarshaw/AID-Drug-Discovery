# Research Analysis: AI Drug Discovery & Personalized Medicine
## Literature Review + SOTA Verification — April 2026

> This file is auto-referenced by CLAUDE.md. See also @docs/architecture.md for bug catalogue.
> **Last updated**: April 2026

---

## SOTA Per Phase (Quick Reference)

| Phase | 2026 Best Method | Key Tool |
|---|---|---|
| Disease Subtyping | Subtype-GAN + Transformer fusion | PyTorch custom |
| Biomarker Discovery | Per-omics AEs + LMF + Integrated Gradients | shap, deepchem |
| Drug Response | MMDRP-style multimodal | TensorFlow/PyTorch |
| Drug Repurposing | PyKEEN KG + BioGPT path reasoning | pykeen, transformers |
| Molecular Simulation | AlphaFold3 co-folding + ADMET-AI | alphafold3, admet-ai |
| Personalized Medicine | Digital Twin simulation + LLM explanation | All phases combined |
| De Novo Generation | DiffMol diffusion / MolGAN | deepchem, torch |

---

## 3 Critical 2026 Upgrades

### Upgrade 1: AlphaFold3 (Released Open-Source January 2026)
- **Replaces**: AlphaFold2 (colabfold) + DiffDock (two-step process)
- **Why**: AF3 handles protein-ligand co-folding in a single model
- **Source**: `github.com/google-deepmind/alphafold3` (weights need registration)
- **Impact**: Phase 5 molecular simulation is now more accurate and streamlined

### Upgrade 2: LLM + Knowledge Graph for Drug Repurposing
- **Replaces**: PyKEEN-only TransE embedding
- **Why**: 2025–2026 SOTA (CLEAR, LLaDR, DrugCORpath) shows LLM path reasoning
  gives mechanism-aware repurposing and human-readable explanations
- **Tools**: PyKEEN (KG base) + BioGPT/BioBERT (path reasoning)
- **Impact**: Phase 4 now produces "biological path sentences" explaining each candidate

### Upgrade 3: Digital Twin Patient Simulation Framing
- **Replaces**: "KNN recommendation" language
- **Why**: 2026 clinical AI standard — simulate drug responses virtually before administering
- **Impact**: Phase 6 is now framed as a Digital Twin engine (same computation, stronger clinical relevance)

---

## Research Gaps Confirmed Novel (April 2026)

1. **LIHC Subtype-GAN**: Never applied to liver cancer
2. **LINCS + GDSC fusion**: No paper fuses LINCS signatures as drug features with GDSC IC50 labels
3. **LLM+KG for LIHC**: CLEAR/LLaDR target ADRD — LIHC is uncovered
4. **AF3 for LIHC candidates**: No paper uses open-source AF3 for LIHC drug candidates
5. **End-to-end XAI pipeline**: No unified SHAP+LLM chain from omics to recommendation exists

---

## Full Analysis
See the full document in the artifacts directory.
