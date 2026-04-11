"""YAML prompt loading from package resource directories."""
from functools import lru_cache

from importlib.resources import files

import yaml


@lru_cache(maxsize=None)
def load_prompts(package: str) -> dict:
    """Load and cache a prompts.yaml file from a package's resource directory."""
    prompts_path = files(package).joinpath("prompts.yaml")
    with prompts_path.open("r") as f:
        return yaml.safe_load(f)
