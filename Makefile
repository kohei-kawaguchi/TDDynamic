PWD              := $(shell pwd)
PYTHON           := poetry run python
PYTEST           := poetry run pytest
PYSEN            := poetry run pysen
MODULE           := td_dynamic



.PHONY: all format test clean clean-pyc clean-build setup

all: format lint test

lint:
	$(PYSEN) run lint

format:
	$(PYSEN) run format

test:
	PYTHONPATH=$(PWD) $(PYTEST)

clean: clean-pyc clean-build

clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf $(MODULE).egg-info/
	rm -rf pip-wheel-metadata/

setup:
	poetry install --with dev

setup-gpu:
	poetry install --with dev
	poetry run pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
