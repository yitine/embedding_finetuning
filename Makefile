.PHONY: install train evaluate clean

install:
	python -m pip install -r requirements.txt

train:
	python -m src.training.train

evaluate:
	python -m src.evaluation.evaluate

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
