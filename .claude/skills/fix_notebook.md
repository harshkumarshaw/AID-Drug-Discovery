---
name: fix-notebook
description: Repair and refactor a broken Jupyter notebook into a clean pipeline module
---

# Skill: Fix & Refactor Jupyter Notebook

## When to Use
Use this skill when asked to fix a broken `.ipynb` notebook in this project.

## Steps

### 1. Diagnose
Read the notebook and identify:
- [ ] Hardcoded absolute paths (look for `D:\`, `C:\`, `E:\`)
- [ ] Empty DataFrames (look for `.dropna()` after `.concat()`)
- [ ] Missing columns (look for `KeyError` comments or missing derived labels)
- [ ] Large files loaded without chunking (look for files > 500 MB)
- [ ] Unextracted archives passed to connectors (`sqlite3.connect`, etc.)

### 2. Fix the Notebook
Apply fixes inline in the notebook cells:
- Replace all paths with `config.py` imports
- Fix data loading issues per `.claude/rules/data_loading.md`
- Fix ML training issues per `.claude/rules/ml_training.md`
- Add `print(f"Step complete: shape={df.shape}")` after each major step

### 3. Extract to Pipeline Module
For each major notebook section, create a corresponding `pipeline/*.py` function:
```python
def phase_name(config: dict) -> pd.DataFrame:
    """
    One-line summary.
    
    Args:
        config: Dictionary from config.py
    
    Returns:
        Processed DataFrame saved to data/processed/
    """
    ...
```

### 4. Write a Smoke Test
Create `tests/test_phase_name.py` with:
- Load first 500 rows only
- Assert output shape > 0
- Assert output columns exist
- Assert no NaN in key columns

### 5. Update task.md
Mark the notebook as fixed in the current task checklist.
