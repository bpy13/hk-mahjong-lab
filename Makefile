# Run inside the `mahjong` conda env: `conda activate mahjong`.
.PHONY: dev dev-windows test test-rust test-python test-web test-e2e lint fmt docs web serve web-dev bench

dev:            ## Build the Rust extension, install the Python package (editable) and UI deps
	pip install -e ".[dev]"
	npm install --prefix web

# Windows: the env's conda-forge cargo (GNU) cannot link, and `make` is often missing.
# This target prints the PowerShell steps; see CLAUDE.md "Windows (this PC)".
dev-windows:    ## Print the Windows setup steps (rustup MSVC cargo, PYO3_PYTHON, CUDA torch)
	@echo '$$env:PATH = "$$env:USERPROFILE\.cargo\bin;" + $$env:PATH'
	@echo '$$env:PYO3_PYTHON = "$$env:CONDA_PREFIX\python.exe"'
	@echo 'pip install -e ".[dev]"'
	@echo 'pip install torch --index-url https://download.pytorch.org/whl/cu130'
	@echo 'Push-Location web; npm install; Pop-Location'

test: test-rust test-python test-web

test-rust:
	cargo test --workspace

test-python:
	pytest

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	ruff check python experiments
	ruff format --check python experiments
	mypy python/mahjong
	cd web && npx tsc -b && npx oxlint

fmt:
	cargo fmt --all
	ruff format python experiments
	ruff check --fix python experiments

docs:           ## Build Rust API docs into target/doc
	cargo doc --workspace --no-deps

test-web:       ## UI unit tests
	cd web && npx vitest run

test-e2e: web   ## Plays a full hand in a real browser (needs `npx playwright install chromium` once)
	cd web && npx playwright test

web:            ## Build the UI into web/dist
	npm run build --prefix web

serve: web      ## UI + API on http://127.0.0.1:8000
	mahjong serve

web-dev:        ## UI with hot reload on :5173 (run `uvicorn mahjong.server.app:app --reload` alongside)
	npm run dev --prefix web

bench:          ## Engine and bot speeds
	cargo run --release -p mahjong-search --example speed
