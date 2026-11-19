PYTHON ?= python3

.PHONY: test test-alphacrafter test-machine-learn

test: test-alphacrafter test-machine-learn

test-alphacrafter:
	cd quant-method/alphacrafter && $(PYTHON) -m unittest discover -v

test-machine-learn:
	cd quant-method/machine-learn && $(PYTHON) -m unittest discover -v
