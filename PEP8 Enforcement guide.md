# PEP8 Enforcement in GitLab

I have tried to implement the enforcement procedure inside this repository as well, so you can find my way of 
implementation as example if any concerns arise.

## Step 1: Install Ruff locally

```bash
pip install ruff
```
Ruff is a better and much faster version of autopep8 library 
## Step 2: Create a pyproject.toml file
The file should containt the following:
```bash
[tool.ruff]
# default is 79
line-length = 79

target-version = "py312"

# (E = pycodestyle, F = pyflakes, W = warnings)
select = ["E", "F", "W"]

ignore = []

# Directories to exclude
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "*.egg-info",
]
 ```
this will set the rules to follow, state the excluded files and ignored rules
## Step 3: Create the GitLab CI/CD Pipeline
Create a file at .gitlab-ci.yml
```bash
stages:
  - lint
  - format

# Check pep8 compliance

pep8-lint:
  stage: lint
  image: python:3.12
  before_script:
    - pip install ruff
    - chmod +x actions/run_ruff.sh
  script:
    - ./actions/run_ruff.sh
  rules:
    - if: '$CI_PIPELINE_SOURCE == "push"'

# Auto-fix
pep8-format:
  stage: format
  image: python:3.12
  before_script:
    - pip install ruff
  script:
    - echo "Running Ruff formatter..."
    - ruff format .
    - ruff check --fix .
  rules:
    - if: '$CI_PIPELINE_SOURCE == "push"'
  allow_failure: true
```
and if you want to organise your CI scripts you can also create a directory "actions" for all future scripts.

