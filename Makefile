.PHONY: setup-hooks test check-privacy

setup-hooks:
	cp hooks/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Git hooks installed."

check-privacy:
	python3 hooks/check_privacy.py

test:
	python3 -m pytest tests/ -v
