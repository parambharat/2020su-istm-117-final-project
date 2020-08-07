.PHONY: dataset install deploy

install:
	pip install pipenv
	pipenv install -v --ignore-pipfile --deploy


deploy:
	pipenv run uvicorn app.main:app

