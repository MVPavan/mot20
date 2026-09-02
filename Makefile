.DEFAULT_GOAL := test

PYTHON := .venv/bin/python

.PHONY: acceptance test test-backend test-frontend test-cvat lint lint-backend lint-frontend build build-frontend e2e e2e-real smoke-local pip-check run

test: test-backend test-frontend test-cvat

test-backend:
	$(PYTHON) -m pytest -m "not local_data"

test-frontend:
	npm --prefix web run test
	npm --prefix web run typecheck

test-cvat:
	$(PYTHON) -m unittest discover -s cvat/tests -v

lint: lint-backend lint-frontend

lint-backend:
	$(PYTHON) -m ruff check src/mot20 tests/viewer
	$(PYTHON) -m mypy

lint-frontend:
	npm --prefix web run typecheck

build:
	$(MAKE) build-frontend

build-frontend:
	npm --prefix web run build

e2e:
	PLAYWRIGHT_BROWSERS_PATH=$(CURDIR)/web/.playwright npm --prefix web run e2e

e2e-real: build
	MOT20_REAL_E2E=1 PLAYWRIGHT_BROWSERS_PATH=$(CURDIR)/web/.playwright npm --prefix web run e2e -- --config=playwright.real.config.ts

smoke-local:
	$(PYTHON) -m pytest -m local_data tests/viewer/test_local_smoke.py

pip-check:
	$(PYTHON) -m pip check

acceptance: test lint build e2e smoke-local pip-check

run:
	$(PYTHON) scripts/run_viewer.py