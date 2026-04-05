# /run-pipeline
# Runs the full end-to-end pipeline in order.

## Steps

1. Confirm `config.py` is present and all paths resolve
2. Run `python -m pipeline.data_loader --validate` — check all datasets accessible
3. Run Phase 1: `python -m pipeline.disease_subtyping`
4. Run Phase 2: `python -m pipeline.biomarker_discovery`
5. Run Phase 3: `python -m pipeline.drug_response`
6. Run Phase 4: `python -m pipeline.drug_repurposing`
7. Run Phase 5: `python -m pipeline.molecular_simulation`
8. Run Phase 6: `python -m pipeline.personalized_medicine`
9. Check all files exist in `data/processed/`
10. Print summary: number of candidates, model metrics, top recommendations
