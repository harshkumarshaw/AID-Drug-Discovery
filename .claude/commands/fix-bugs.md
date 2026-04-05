# /fix-bugs
# Runs the full bug triage and fix workflow for this project.

## Steps

1. Read `docs/architecture.md` Bug Catalogue
2. For each OPEN bug, identify the affected file
3. Apply fix following `.claude/rules/data_loading.md` and `.claude/rules/ml_training.md`
4. Mark bug as FIXED in `docs/architecture.md` with today's date
5. Run smoke test: `pytest tests/ -v -x`
6. Update `tasks.md` to mark that bug as done
