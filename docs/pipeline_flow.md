## Phase Flow (Updated April 2026)

```
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 1: DISEASE SUBTYPING                [2026 UPGRADED] ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    cleaned_clinical_data.csv                                          ║
║    cleaned_biospecimen_data.csv                                        ║
║    TCGA-LIHC mRNA expression matrix (RNA-seq counts)                  ║
║    TCGA-LIHC CNV segments                                             ║
║  MODEL: Subtype-GAN (adversarial multi-omics AE + GMM)               ║
║    Optional: Transformer fusion encoder (comparison model)            ║
║  PROCESS:                                                             ║
║    → Independent encoder per omics (mRNA, CNV, clinical)             ║
║    → Shared adversarial latent space (GAN discriminator)             ║
║    → GMM clustering on shared latent vectors                          ║
║    → Kaplan-Meier survival significance per cluster                   ║
║  OUTPUT:                                                              ║
║    data/processed/patient_clusters.csv  [case_id, z_dim..., Cluster]  ║
║    plots/survival_curves_per_cluster.png                              ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 2: BIOMARKER DISCOVERY              [2026 CONFIRMED] ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    patient_clusters.csv                                               ║
║    E-MTAB-5902 variant features (chunked, per-chromosome counts)      ║
║    TCGA-LIHC RNA-seq expression matrix                                ║
║  MODEL: Per-omics autoencoders → LMF fusion → classifier             ║
║  PROCESS:                                                             ║
║    → Expression AE: gene_counts → z_expr(64)                        ║
║    → Variant AE: variant_counts → z_var(32)                         ║
║    → LMF fusion of z_expr + z_var                                    ║
║    → Classify subtype labels                                          ║
║    → SHAP DeepExplainer + Integrated Gradients → top gene list       ║
║  OUTPUT:                                                              ║
║    data/processed/biomarkers.csv  [gene_id, importance, shap_mean]   ║
║    plots/shap_summary_plot.png                                        ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 3: DRUG RESPONSE PREDICTION         [2026 CONFIRMED] ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    LINCS modzs.gctx (full L1000 via cmapPy)                          ║
║    GDSC2 IC50 data (data/external/gdsc2_drug_sensitivity.csv)         ║
║    CCLE expression (data/external/ccle_expression.csv)                ║
║  MODEL: MMDRP-style multimodal DNN                                   ║
║  PROCESS:                                                             ║
║    → Cell line: expression AE + mutation AE + CNV AE                ║
║    → Drug: Morgan fingerprints + LINCS perturbation z-scores → drug AE ║
║    → LMF fusion → MLP → LN_IC50 prediction                          ║
║    → Evaluation: Trailing Moving Average (TMA) for skewed data        ║
║  OUTPUT:                                                              ║
║    models/drug_response/model.pt                                      ║
║    data/processed/drug_response_predictions.csv                       ║
║    plots/tma_performance_curves.png                                   ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 4: DRUG REPURPOSING                 [2026 UPGRADED]  ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    ChEMBL-35 SQLite (extracted)                                       ║
║    DRKG (Drug Repurposing Knowledge Graph — pre-built)                ║
║    biomarkers.csv (LIHC-specific gene signatures)                     ║
║    DeepDRA model (Drug Repurposing/DeepDRA-main/)                     ║
║  MODEL: PyKEEN (KG embedding) + BioGPT (path reasoning) + DeepDRA   ║
║  PROCESS:                                                             ║
║    → Build augmented KG: DRKG + ChEMBL drug-target + KEGG pathways  ║
║    → PyKEEN TransE/RotatE: train link prediction embeddings           ║
║    → Score drug→LIHC disease edges not in original graph             ║
║    → BioGPT: read drug→gene→disease→pathway multi-hop paths          ║
║       → "biological path sentences" → path-aware score               ║
║    → LLM generate explanation text per candidate                      ║
║    → DeepDRA: predict drug-cell line interaction score                ║
║    → Final: 0.4*KG + 0.3*LLM_path + 0.3*DeepDRA                    ║
║  OUTPUT:                                                              ║
║    data/processed/repurposing_candidates.csv                          ║
║    [drug, chembl_id, kg_score, llm_score, deepdra_score, explanation] ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 5: MOLECULAR SIMULATION             [2026 UPGRADED]  ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    repurposing_candidates.csv (top 50)                                ║
║    LIHC target proteins (ChEMBL target_dictionary + UniProt)          ║
║  MODEL: AlphaFold3 (open-source Jan 2026) + ADMET-AI                 ║
║  PROCESS:                                                             ║
║    → Retrieve protein sequences from ChEMBL/UniProt                  ║
║    → AlphaFold3: protein + ligand SMILES → co-folded complex          ║
║    → Extract binding confidence: pTM, ipTM scores                    ║
║    → ADMET-AI: 42 ADMET endpoints from SMILES                        ║
║    → Filter: binding confidence > threshold + ADMET: safe            ║
║  OUTPUT:                                                              ║
║    data/processed/interaction_scores.csv                              ║
║    [drug, target, af3_ptm, af3_iptm, admet_flags, pass/fail]         ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 6: DIGITAL TWIN PATIENT SIMULATION  [2026 UPGRADED]  ║
╠══════════════════════════════════════════════════════════════════════╣
║  INPUT:                                                               ║
║    New patient profile {age, gender, markers, tumor_stage, diagnosis} ║
║    patient_clusters.csv + drug_response_predictions.csv               ║
║    interaction_scores.csv + repurposing_candidates.csv                ║
║  MODEL: Subtype-GAN inference + MMDRP inference + SHAP + LLM        ║
║  PROCESS:                                                             ║
║    → Run Subtype-GAN encoder on patient features → assign cluster    ║
║    → For each repurposing candidate: predict IC50 (MMDRP inference)  ║
║    → Filter by predicted binding (AF3 scores)                         ║
║    → Filter by ADMET flags                                            ║
║    → Rank remaining drugs                                             ║
║    → Compute SHAP values per recommendation                           ║
║    → LLM generates clinical narrative: "Drug X recommended because.."║
║  OUTPUT:                                                              ║
║    JSON: { patient_id, cluster, cluster_description,                  ║
║            recommended_drugs: [{drug, ic50, binding, explanation}] }  ║
╚══════════════════════════════════════════════════════════════════════╝
```
