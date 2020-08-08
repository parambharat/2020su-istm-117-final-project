.PHONY: dataset install deploy

install:
	pip install pipenv
	pipenv install -v


deploy:
	pipenv run uvicorn app:app --reload --host=0.0.0.0

