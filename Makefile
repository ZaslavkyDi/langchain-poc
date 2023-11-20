.PHONY: ruff
ruff:
	poetry run ruff . --fix

.PHONY: black
black:
	poetry run black .

.PHONE: lint
lint: black ruff