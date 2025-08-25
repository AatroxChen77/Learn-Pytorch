import os, random
import torch
import numpy as np
import argparse
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device='cpu'):
    return torch.load(path, map_location=device)


def parse_base_config_arg(default_config_path: str | None):
    """
    Create a base parser that only parses --config to allow early YAML loading.
    Returns (base_parser, known_args).
    """
    # Initialize a temporary parser that will only parse the --config argument.
    # We set add_help=False so this parser does not define -h/--help,
    # which avoids conflicts when we later create the full parser.
    base_parser = argparse.ArgumentParser(add_help=False)
    # Add a --config argument (short form: -c).
    # This allows the user to specify a YAML configuration file path.
    # The default is taken from default_config_path if provided.
    # metavar="" just ensures that the help message looks cleaner (it won’t display something like "CONFIG" after --config).
    base_parser.add_argument("-c", "--config", type=str, default=default_config_path, help="path to YAML config", metavar="")
    # parse_known_args() parses only the arguments known to base_parser,
    # and ignores the rest (which will later be parsed by the full parser).
    # This way, we can extract --config early, before defining all other arguments.
    known_args, _ = base_parser.parse_known_args()
    # Return both the parser itself (for possible reuse/extension)
    # and the parsed --config value.
    return base_parser, known_args


def load_yaml_defaults(config_path: str | None) -> dict:
    """Load YAML file into a flat dict of defaults. Returns {} if not found/invalid."""
    if not config_path:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}
