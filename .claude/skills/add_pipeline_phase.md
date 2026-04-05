---
name: add-pipeline-phase
description: Add a new phase to the drug discovery pipeline with module, notebook, and test
---

# Skill: Add New Pipeline Phase

## Steps

### 1. Create the Pipeline Module
Create `pipeline/<phase_name>.py` with:
- A main function: `def run(config: dict) -> pd.DataFrame`
- Input validation using `.claude/rules/data_loading.md` patterns
- Model training using `.claude/rules/ml_training.md` patterns
- Output saved to `data/processed/<phase_name>_output.csv`
- Structured logging: `logger = logging.getLogger(__name__)`

### 2. Create the Notebook
Copy the closest existing notebook from `notebooks/` as a template.
Update:
- Cell 1: Import from `config.py` and new module
- Cell 2: Load and validate data (show `.shape` and `.head()`)
- Cell 3: Run the pipeline phase
- Cell 4: Visualize outputs (scatter plot or bar chart)
- Cell 5: Save outputs

### 3. Register in Master Orchestrator
In `pipeline/__init__.py`, import and add to `PIPELINE_STAGES`:
```python
from pipeline import <phase_name>
PIPELINE_STAGES = [..., <phase_name>.run]
```

### 4. Add API Endpoint
In `api/app.py`, add:
```python
@app.route('/predict/<phase_name>', methods=['POST'])
def predict_phase():
    ...
```

### 5. Write Tests
Create `tests/test_<phase_name>.py`

### 6. Update docs
- Add dataset info to `docs/datasets.md`
- Add flow arrows to `docs/pipeline_flow.md`
- Log decision in `docs/architecture.md`
