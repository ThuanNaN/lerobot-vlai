This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
uv run pytest tests/test_foo.py -svv                  # Single test file
uv run pytest tests/test_foo.py::test_bar -svv        # Single test
LEROBOT_TEST_DEVICE=cuda uv run pytest tests -svv     # Run tests on a specific device (default: auto-detected)
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                            # Lint + format (ruff, typos, bandit, etc.)
uv run ruff check . && uv run ruff format .            # Lint + format directly
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`), CI (`Dockerfile.internal`), and per-benchmark images (`Dockerfile.benchmark.<name>`, e.g. `libero`, `metaworld`, `robotwin`) run by `.github/workflows/benchmark_tests.yml`.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information), `AI_POLICY.md` (disclosure/review expectations for AI-assisted contributions).

## This fork: SmolVLA + LIBERO Vietnamese-instructions project

This is a customized fork whose active work adapts the SmolVLA + LIBERO training pipeline so the policy follows **Vietnamese** task instructions (language is input-only — no Vietnamese generation). Design and staged plan live in `docs/superpowers/specs/2026-07-02-smolvla-vietnamese-instructions-design.md` and `docs/superpowers/plans/2026-07-02-smolvla-vietnamese-instructions.md` — read these first before touching the pipeline.

- **Run wrappers** (root, all source `.env` for `HF_USER` / `WANDB_API_KEY` / `MUJOCO_GL=egl`; copy `.env.example`):
  - `run_train.sh` — Stage 1 baseline: train `smolvla` on `HuggingFaceVLA/libero` (English) → `./outputs/train/`. Multi-GPU via `NUM_GPUS>1` → `accelerate launch`. Tunables: `TASK_SUITE`/`STEPS`/`BATCH_SIZE`/`NUM_GPUS`.
  - `run_eval.sh <checkpoint> <output_dir>` — Stage 1 benchmark: eval a checkpoint across all 4 LIBERO suites (English instructions).
  - `run_vi.sh` — Stage 2: LoRA finetune on the Vietnamese-forked dataset → `./outputs/train_vi/`. Its `TARGET_MODULES` regex widens PEFT beyond SmolVLA's default (action-expert q/v only) to also adapt the VLM **text-model** attention layers — required for language adaptation.
  - `run_eval_vi.sh <checkpoint> <output_dir>` — Stage 3: eval with Vietnamese instructions injected via `--env.task_language_overrides_path`.
- **Key source modification**: eval instruction override. `lerobot-eval` on LIBERO sources the task string from the upstream `libero` package (English), **not** from the dataset — so translating the dataset alone does nothing at eval time. The knob added here: `EnvConfig.task_language_overrides_path` (`src/lerobot/envs/configs.py:327`) loads a per-suite/per-task JSON that `LiberoEnv` applies via `task_language_override` (`src/lerobot/envs/libero.py`).
- **`vlai-experiments/vi-instructions/`** — the offline data/eval tooling (standalone scripts, not `lerobot.*` modules):
  - `extract_tasks.py` / `validate_translations.py` — pull LIBERO task strings, validate `data/tasks_en.csv` → `data/tasks_vi.csv`.
  - `build_dataset.py` — fork the LIBERO dataset with translated task strings.
  - `build_eval_overrides.py` — generate `data/eval_overrides.json` consumed by `run_eval_vi.sh`.
  - `diagnostics.py` — Stage 0 tokenizer-fertility / embedding sanity checks (no training).
  - `compare_eval.py` — build a suite-vs-suite success-rate comparison table from `lerobot-eval` output.
  - Tests live in `vlai-experiments/vi-instructions/tests/` and add the parent dir to `sys.path`; run with `uv run pytest vlai-experiments/vi-instructions/tests/ -q` (separate from the main `tests/` suite).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
