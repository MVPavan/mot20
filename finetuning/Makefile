PYTHON ?= ../.venv/bin/python

.PHONY: test compile

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/detection -p "test_*.py" -v

compile:
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests scripts