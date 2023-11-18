#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to train mBART50 on filtered data


This script does the following:
- loads tokenized dataset
- performs filtering based on method and level form args
- generates train-val split for fitlered data
- fine-tunes model
- saves model

TODO:
- agree on some nice hyperparaemters for trianing
- agree on layers to freeze

Note on folder structure:
└── args.base_dir
    ├── data
    │   └── *medical_corpus_clean_preprocessed.tsv*
    │   
    ├── filtering
    │   ├── *method1_level1.tsv*
    │   ├── *method2_level1.tsv*
    │   ├── *method1_level2.tsv*
    │   ...
    │   └── *methodn_leveln.ysv*
    ├── logs
    │   ├── *method1_level1*
    │   ...
    │   └── *methodn_leveln* 
    ├── models
    │   ├── *method1_level1*
    │   ...
    │   └── *methodn_leveln*
    ├── tokenizers
    │   └── *dataset_tokenized*
    └── training

@author: jorgedelpozolerida
@date: 17/11/2023
"""

import os
import sys
import argparse
import traceback

from typing import Dict, Any
import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import os
import sys
import torch
import gc
import datasets
from sklearn.model_selection import train_test_split

from transformers import MBart50Tokenizer, MBartForConditionalGeneration, TrainingArguments, Trainer
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Mapping of column names for languages to mBART50 language codes
MAPPING_LANG: Dict[str, str] = {
    "pol": "pl_PL",
    "eng": "en_XX"
}

def filter_and_split_dataset(args, original_dataset):
    """
    Filters and splits a given dataset based on a filter file and stratifies the split.

    Args:
        args: ArgumentParser object containing necessary configurations.
        original_dataset: The original dataset to be filtered and split.
    """
    
    # Create a deep copy of the original dataset to avoid modifying it directly
    dataset = original_dataset.copy()

    # Attempt to read the filter file with comma or tab separator
    try:
        ids_selected = pd.read_csv(os.path.join(args.base_dir, "filtering", args.filter_file), sep=',')
    except pd.errors.ParserError:
        ids_selected = pd.read_csv(os.path.join(args.base_dir, "filtering", args.filter_file), sep='\t')

    # Convert 'id' column from the filter file into a set for efficient searching
    selected_ids_set = set(ids_selected['id'])

    # Filter the 'train_val' split of the dataset
    train_val_dataset = dataset['train_val'].filter(lambda example: example['id'] in selected_ids_set)

    # Stratified split of the filtered 'train_val' dataset
    train_val_split = train_val_dataset.train_test_split(test_size=0.2, stratify_by_column='src')

    # Update the dataset with new 'train' and 'val' splits
    dataset['train'] = train_val_split['train']
    dataset['val'] = train_val_split['test']

    # Remove the 'train_val' split
    del dataset['train_val']

    return dataset

def get_run_metadata(args):
    '''
    Function to extarct metadata from filtering file for run
    '''
    
    if args.filter_file is None:
        return None, None, "Baseline"
    
    filename_split = args.filter_file.split(".")[0].split("_")
    if len(filename_split) < 2:
        raise ValueError(f"Invalid format for 'filter_file': {args.filter_file}")
    
    method = filename_split[0]
    level = filename_split[-1]

    # Create a subname by combining 'method' and 'level'
    subname = f"{method}_{level}"

    return method, level, subname

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def main(args):

    assert os.path.isdir(args.base_dir)
    assert args.source_lang != args.target_lang

    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    method, level, model_subname = get_run_metadata(args)
    
    # Setup directories
    model_folder = ensure_dir(os.path.join(args.base_dir, "models", model_subname))
    tokenizer_folder = os.path.join(args.base_dir, "tokenizers")
    training_folder = ensure_dir(os.path.join(args.base_dir, "training", model_subname))
    logs_folder = ensure_dir(os.path.join(args.base_dir, "logs", model_subname))

    # Log parameters and method
    _logger.info(f"Parameters: {args}")
    _logger.info(f"Methods: {method}, level: {level}")
    
    # Use a GPU if available and specified
    if "gpu" in args.device_name.lower():
        device = torch.device(f"cuda:{args.device_name}" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    _logger.info(f"Using device: {device}")
    
    # Load tokenizer and tokenized dataset
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, 
                                                    src_lang=MAPPING_LANG[args.source_lang], 
                                                    tgt_lang=MAPPING_LANG[args.target_lang]
                                                    )  
    tokenized_datasets = datasets.load_from_disk(os.path.join(tokenizer_folder, args.dataset_name))


    # Perform filtering
    if args.filter_file is not None:
        tokenized_datasets_filt = filter_and_split_dataset(args, tokenized_datasets)
    else:
        tokenized_datasets_filt = tokenized_datasets.copy()
    
    
    # Logging
    train_size = len(tokenized_datasets_filt['train'])
    validation_size = len(tokenized_datasets_filt['val'])
    test_size = len(tokenized_datasets_filt['test'])
    total_size_after_filtering = train_size + validation_size
    total_size_before_filtering = len(tokenized_datasets['train_val'])

    _logger.info(
        f"Finished filtering dataset '{args.dataset_name}' using {args.filter_file}\n" + \
        f"\tSize before filtering: {total_size_before_filtering}\n" + \
        f"\tSize after filtering: {total_size_after_filtering}\n" + \
        f"\ttrain_size: {train_size}, valid_size: {validation_size}, test_size: {test_size}"
    )


    # Load model
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Freeze all layers except the last one. TODO: investigate best layers to freeze
    msg = 'Unfreezing layers: \n'
    for name, param in model.named_parameters():
        if 'model.encoder.layers.11.' in name or 'model.decoder.layers.11.' in name:
            msg += f"\t{name}\n"
            param.requires_grad = True
        else:
            param.requires_grad = False
    _logger.info(msg)
    
    # Define your training arguments
    training_args = TrainingArguments(
        output_dir=training_folder,          # Output directory for model checkpoints
        num_train_epochs=args.num_train_epochs,                 # Number of training epochs
        per_device_train_batch_size=args.train_batch_size,      # Batch size for training (small for GPU memory issues)
        per_device_eval_batch_size=args.eval_batch_size,       # Batch size for evaluation
        warmup_steps=500,                                  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                                 # Weight decay if applied
        logging_dir=logs_folder,   # Directory for storing logs
        logging_steps=10,                                  # Log metrics every 'logging_steps' steps
        evaluation_strategy='steps',                       # Evaluate every logging_steps
        eval_steps=10,                                     # Number of steps to run evaluation
        load_best_model_at_end=True,                       # Load the best model at the end of training
        metric_for_best_model='bleu',                      # Use BLEU score for best model selection
        greater_is_better=True                             # Higher BLEU score is better
    )

    # Initialize Trainer
    callbacks = [TensorBoardCallback()]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_filt['train'],
        eval_dataset=tokenized_datasets_filt['val'],
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    # Training
    _logger.info("Starting training")
    trainer.train()

    # Save model
    trainer.save_model(model_folder)



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--base_dir', type=str, default=None, required=True,
                        help='Path where all subfolders are present for the project')
    parser.add_argument('--device_name', type=str, default="cpu", 
                        help='Name of device to use')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized",
                        help='Name of device to use')
    parser.add_argument('--filter_file', type=str, default=None,
                        help='Name of file with ids after filtering')
    parser.add_argument('--source_lang', type=str, default="eng", choices=['eng', 'pol'],
                        help='Source language')
    parser.add_argument('--target_lang', type=str, default="pol", choices=['eng', 'pol'],
                        help='Target language')
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("-- ", type=int, default=12, help="Batch size for evaluation")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
