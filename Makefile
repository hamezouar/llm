PYTHON = python3

SCRIPT = -m src

# Run main script
run:
	@${PYTHON} ${SCRIPT}

# Install dependencies
install:
	${PYTHON} -m pip install --upgrade pip 
	${PYTHON} -m pip install flake8 mypy
	${PYTHON} -m pip install pydantic mypy 


# Debug mode with pdb
debug:
	${PYTHON} -m pdb ${SCRIPT}

# Clean caches
clean:
	@rm -rf __pycache__ .mypy_cache .pytest_cache  */__pycache__  llm_sdk/llm_sdk/*cache*


# Lint (standard)
lint:
	@${PYTHON} -m flake8 ${PWD} --exclude=llm_sdk/llm_sdk/__init__.py
	@${PYTHON} -m mypy ${PWD} --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

# Lint strict - optional
lint-strict:
	${PYTHON} -m flake8 ${PWD}
	${PYTHON} -m mypy ${PWD} --strict