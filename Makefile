PYTHON ?= python3

.PHONY: test test-alphacrafter test-machine-learn test-smoke lint test-llm test-qlib

test: test-alphacrafter test-machine-learn

test-alphacrafter:
	cd quant-method/alphacrafter && $(PYTHON) -m unittest discover -v

test-machine-learn:
	cd quant-method/machine-learn && $(PYTHON) -m unittest discover -v

test-smoke:
	cd quant-method/alphacrafter && $(PYTHON) -m unittest tests.test_pipeline.PipelineIntegrationTests -v

lint:
	$(PYTHON) -m ruff check quant-method integration

test-llm:
	LITELLM_LOCAL_MODEL_COST_MAP=True LANGSMITH_TRACING=false $(PYTHON) -m unittest discover -s integration -p test_llm.py -v

test-qlib:
	$(PYTHON) -m unittest discover -s integration -p test_qlib.py -v
