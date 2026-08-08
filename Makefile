.PHONY: install run debug clean lint lint-strict test help

FUNCTIONS_DEF ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calling_results.json
MODEL ?= Qwen/Qwen3-0.6B
install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model  $(MODEL)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)
		--model  $(MODEL)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
