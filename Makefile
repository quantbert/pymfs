.DEFAULT_GOAL := help

.PHONY: help install test docs clean publish

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Create/sync the environment (incl. dev deps)
	uv sync

test: ## Run the test suite
	uv run pytest -q

docs: ## Placeholder for future documentation tooling
	@echo "Documentation tooling is not configured yet."

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache dist build
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +

publish: test ## Runs test first. Then bump patch version, build and publish to PyPI
	uv version --bump patch
	rm -rf dist
	uv build
	uv publish
