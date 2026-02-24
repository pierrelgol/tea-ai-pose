set shell := ["bash", "-cu"]

config := "config.json"

default:
    @just --list

venv:
    @if [[ ! -d .venv ]]; then uv venv .venv; fi

build: venv
    @uv sync --all-packages
    @shopt -s nullglob
    @for pkg in packages/*; do \
      if [[ -f "$pkg/pyproject.toml" ]]; then \
        rm -rf "$pkg/dist" "$pkg/build"; \
        mkdir -p "$pkg/dist"; \
        uv build "$pkg" --out-dir "$pkg/dist"; \
      fi; \
    done

clean:
    @find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) -prune -exec rm -rf {} +
    @for path in artifacts dataset wandb; do \
      if [[ -e "$path" ]]; then rm -rf "$path"; fi; \
    done
    @if [[ -d dist || -d build || -f .coverage || -d htmlcov ]]; then \
      rm -rf dist build .coverage htmlcov; \
    else \
      :; \
    fi
    @shopt -s nullglob
    @for pkg in packages/*; do \
      rm -rf "$pkg/dist" "$pkg/build"; \
      find "$pkg" -type d -name "*.egg-info" -prune -exec rm -rf {} +; \
    done

fclean: clean
    @if [[ -d .venv ]]; then rm -rf .venv; fi
    @if [[ -d .pixi ]]; then rm -rf .pixi; fi
    @if [[ -d .tox ]]; then rm -rf .tox; fi
    @if [[ -d .nox ]]; then rm -rf .nox; fi
    @if [[ -d .hypothesis ]]; then rm -rf .hypothesis; fi
    @find . -maxdepth 1 -type f -name "*.pt" -delete

fetch-dataset: venv
    @uv sync --all-packages
    @uv run dataset-fetcher --config {{config}}

fetch-dinov3: venv
    @uv sync --all-packages
    @uv run dinov3-fetcher --config {{config}}

label-targets: venv
    @uv sync --all-packages
    @uv run target-labeller --config {{config}}

generate-dataset: venv
    @uv sync --all-packages
    @uv run dataset-generator --config {{config}}

check-dataset: venv
    @uv sync --all-packages
    @uv run augment-checker --config {{config}}

train: venv
    @uv sync --all-packages
    @uv run detector-train --config {{config}}

tune-gpu: venv
    @uv sync --all-packages
    @uv run gpu-auto-tuner --config {{config}}

eval: venv
    @uv sync --all-packages
    @uv run detector-infer --config {{config}}
    @uv run detector-grader --config {{config}}

optimize: venv
    @uv sync --all-packages
    @uv run detector-optimize --config {{config}}

review: venv
    @uv sync --all-packages
    @uv run detector-reviewer --config {{config}}

profile-pipeline: venv
    @uv sync --all-packages
    @uv run pipeline-profile --config {{config}}
