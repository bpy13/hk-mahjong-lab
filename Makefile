# Run inside the `mahjong` conda env: `conda activate mahjong`.
.PHONY: dev test test-rust test-python lint fmt docs

dev:            ## Build the Rust extension and install the Python package (editable)
	pip install -e ".[dev]"

test: test-rust test-python

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

fmt:
	cargo fmt --all
	ruff format python experiments
	ruff check --fix python experiments

docs:           ## Build Rust API docs into target/doc
	cargo doc --workspace --no-deps
