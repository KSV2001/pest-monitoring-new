import dotenv
import os
import hydra
from omegaconf import DictConfig, open_dict
from hydra.experimental import compose, initialize
import argparse
import warnings

warnings.simplefilter('ignore')
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
OUT_DIR = '/output/'

@hydra.main(config_path="configs/")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils
    
    with open_dict(config):
        config.run_name = os.getcwd().split(OUT_DIR)[1]
    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
