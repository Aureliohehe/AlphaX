PYTHON ?= python3

.PHONY: test test-alphacrafter test-machine-learn test-smoke

test: test-alphacrafter test-machine-learn

test-alphacrafter:
	cd quant-method/alphacrafter && $(PYTHON) -m unittest discover -v

test-machine-learn:
	cd quant-method/machine-learn && $(PYTHON) -m unittest discover -v

test-smoke:
	cd quant-method/alphacrafter && $(PYTHON) -m unittest tests.test_pipeline.PipelineIntegrationTests -v
