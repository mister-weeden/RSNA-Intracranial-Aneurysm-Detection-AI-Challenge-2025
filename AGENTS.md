# Repository Guidelines

## Project Structure & Module Organization
- `aneurysm_preprocessing.py`: Image preprocessing and vessel enhancement.
- `train_medicalnet_aneurysm.py`: Model, trainer, deterministic seeding, performance monitor.
- `evaluation_comprehensive.py`: Competition scoring and statistical analysis.
- `experiment_generator.py`: Baseline and ablation experiment configs.
- `demo_complete_pipeline.py`: End‑to‑end demo of the full pipeline.
- `test_suite_comprehensive.py`: Unittest suite for determinism, complexity, accuracy.
- `models/`: Model code (e.g., `models/MedicalNet`) and checkpoints.
- `data/`: Local datasets and intermediate artifacts (not versioned).
- `MONITORING_SYSTEM.md`, `monitor_training.py`, `install_scheduler.sh`: Optional training monitor and scheduler.

## Build, Test, and Development Commands
- Create env (example): `python3 -m venv .venv && source .venv/bin/activate`
- Install core deps (example): `pip install numpy torch nibabel scikit-image numba psutil scikit-learn`
- Run demo: `python3 demo_complete_pipeline.py`
- Run tests (verbose): `python3 -m unittest -v test_suite_comprehensive.py`
- Monitor training (manual): `python3 monitor_training.py`
- Install scheduler (macOS): `chmod +x install_scheduler.sh && ./install_scheduler.sh`

## Coding Style & Naming Conventions
- Python 3.x, 4‑space indentation, PEP 8. Use type hints and docstrings on public APIs.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, module files in `snake_case.py`.
- Keep functions cohesive; favor vectorized NumPy/PyTorch ops. Avoid committing data or PHI.

## Testing Guidelines
- Framework: `unittest` in `test_suite_comprehensive.py`.
- Naming: tests start with `test_` and target deterministic behavior, complexity, memory, and accuracy.
- Run all tests: `python3 -m unittest -v test_suite_comprehensive.py`.
- Add tests alongside new modules or extend the comprehensive suite when modifying core behavior.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits (e.g., `feat: add vesselness filter`, `fix: stabilize seed handling`).
- Commits should be scoped and descriptive; avoid “Training”/“Update” without context.
- PRs must include: clear description, rationale, affected files, how to test (commands/inputs), and any relevant metrics/plots.
- Link issues, attach logs/screenshots (e.g., from `training_monitor.log`), and update docs when interfaces change.

## Security & Configuration Tips
- Store datasets in `data/` and exclude from VCS. Do not commit PHI.
- Ensure determinism: call `set_deterministic_seed(42)` when adding training/inference entry points.
- GPU optional; code paths should run on CPU by default.
