.PHONY: setup clean lint test train predict deploy

# Setup environment
setup:
	pip install -r requirements.txt

# Clean artifacts
clean:
	rm -rf logs/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf .pytest_cache

# Run linting
lint:
	flake8 src tests app.py

# Run tests
test:
	pytest tests/

# Train model
train:
	python scripts/train.py

# Make predictions
predict:
	python scripts/predict.py

# Deploy to Hugging Face Spaces
deploy:
	huggingface-cli login
	git add .
	git commit -m "Update model and app"
	git push

