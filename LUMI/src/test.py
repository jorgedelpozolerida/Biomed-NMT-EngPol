
import os
import sys
import argparse
import logging                                                                      
import os
import json

import pandas as pd                                                                
import torch
import datasets

from transformers import MBart50Tokenizer, MBartForConditionalGeneration, TrainingArguments, Trainer
from transformers.integrations import TensorBoardCallback

THISFILE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(args):

    # Load JSON configuration file for training hyperparameters
    with open(args.config_path, 'r') as file:
        hyperparams = json.load(file)
    # Log configuration
    _logger.info("Training configuration:")
    _logger.info(json.dumps(hyperparams, indent=4, sort_keys=True))



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=None, required=True,
                        help='Path to config file in json format with hyperparams')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
