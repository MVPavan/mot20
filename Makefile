.DEFAULT_GOAL := test

PYTHON := .venv/bin/python

.PHONY: acceptance test test-viewer test-cvat lint build e2e e2e-real smoke-local pip-check run

test: test-viewer test-cvat

test-viewer:
	$(MAKE) -C track-viz test

test-cvat:
	$(PYTHON) -m unittest discover -s cvat/tests -v

lint build e2e e2e-real smoke-local pip-check run:
	$(MAKE) -C track-viz $@

acceptance: test lint build e2e smoke-local pip-check