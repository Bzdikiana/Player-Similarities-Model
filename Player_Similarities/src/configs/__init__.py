"""
Configuration loader utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load all configuration files from the configs directory.
    
    Returns dict with keys: 'data', 'model', 'train', 'eval'
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    else:
        config_dir = Path(config_dir)
    
    configs = {}
    
    for config_name in ['data', 'model', 'train', 'eval']:
        config_path = config_dir / f'{config_name}.yaml'
        if config_path.exists():
            configs[config_name] = load_config(str(config_path))
        else:
            configs[config_name] = {}
    
    return configs


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configs, with override taking precedence.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


@dataclass
class Config:
    """Container for all configuration."""
    
    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    
    @classmethod
    def from_dir(cls, config_dir: Optional[str] = None) -> 'Config':
        """Load configuration from directory."""
        configs = load_all_configs(config_dir)
        return cls(**configs)
    
    def get_nested(self, *keys, default=None):
        """Get a nested config value."""
        result = self.__dict__
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
