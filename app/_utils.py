import os
import yaml
import langchain

def get_config(config_path) -> dict:
    """ Load configuration from config.yaml file.
    
    Args:
        config_path (str): Path to config.yaml file.
    Returns:
        config (dict): Configuration dictionary."""
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def init_langchain(config : dict) -> None:
    """ Initialize Langchain settings.

    Args:
        config (dict): Configuration dictionary.
    """

    langchain.debug = config["langchain"]["debug"]
    langchain.verbose = config["langchain"]["verbose"]
    # set LANGCHAIN_TRACING_V2 environment variable to config value
    os.environ["LANGCHAIN_TRACING_V2"] = str(config["langchain"]["tracing"])