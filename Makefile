# Makefile for the Lex Eval project

.PHONY: install install-dev static-type-check lint lint-check test stress-test pr help

# Default target
default: help

install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv sync

install-dev:
	@echo "--- 🚀 Installing development dependencies ---"
	uv sync --dev

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	uv run mypy src

lint:
	@echo "--- 🧹 Formatting and linting codebase ---"
	uv run ruff format .
	uv run ruff check . --fix

lint-check:
	@echo "--- 🧐 Checking if project is formatted and linted ---"
	uv run ruff format . --check
	uv run ruff check .

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest src/tests/

stress-test:
	@echo "--- 🔥 Running stress test ---"
	uv run python -m lex_eval.stress_test.run_stress_test $(ARGS)

pr: static-type-check lint test
	@echo "--- ✅ All PR checks passed successfully ---"
	@echo "Ready to make a PR!"

help:
	@echo "Makefile for the Lex Eval project"
	@echo ""
	@echo "Available commands:"
	@echo "  make install             Install project dependencies using uv sync"
	@echo "  make install-dev         Install development dependencies"
	@echo "  make static-type-check   Run static type checking with mypy on the src directory"
	@echo "  make lint                Format code with Ruff and apply lint fixes"
	@echo "  make lint-check          Check formatting with Ruff and run lint checks without fixing"
	@echo "  make test                Run tests with pytest"
	@echo "  make pr                  Run all pre-PR checks: static-type-check, lint, and test"
	@echo "  make help                Show this help message"
