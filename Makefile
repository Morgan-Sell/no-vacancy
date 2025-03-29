# Use to run linters and tests before committing code.
.PHONY: precommit-local
precommit-local:
	pre-commit autoupdate --config .pre-commit-config.yaml
	pre-commit run --all-files --config .pre-commit-config.yaml
	pre-commit clean