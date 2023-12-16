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
# Included in Python
import os
import argparse
import logging                                                                      
import json
import datetime
import random


# Installed apart in container
import pandas as pd                                                                
import torch
import datasets

from transformers import MBart50Tokenizer, MBartForConditionalGeneration, TrainingArguments, Trainer
import sentencepiece as spm # just ot make sure it is there

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

THISFILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapping of column names for languages to mBART50 language codes
MAPPING_LANG = {
    "pol": "pl_PL",
    "eng": "en_XX"
}

def read_csv_corrupt(path2file, cols_used):
    try:
        d = pd.read_csv(path2file, sep='\t', usecols=cols_used)
    except pd.errors.ParserError:
        d = pd.read_csv(path2file, sep=',', usecols=cols_used)

    return d

def subset_trainval_set(original_dataset, n_sentences=10000, seed=42):
    """
    Subsets an already filtered training dataset for computational reasons.
    Selects a random subset of 'n_sentences' samples from the training set.
    
    Args:
        original_dataset (datasets.Dataset): The original dataset to be subsetted.
        n_sentences (int): Number of sentences to select in the subset.
        seed (int): Random seed for reproducibility.

    Returns:
        datasets.Dataset: A subset of the original dataset.
    """
    
    # Set the seed for reproducibility
    random.seed(seed)

    # Get indices for the subset
    indices = random.sample(range(len(original_dataset)), n_sentences)

    # Select the subset from the original dataset
    subset_dataset = original_dataset.select(indices)

    return subset_dataset

def filter_and_sample_and_split_dataset(args, original_dataset):
    """
    Filters and splits a given dataset based on a filter file and stratifies the split.

    Args:
        args: ArgumentParser object containing necessary configurations.
        original_dataset: The original dataset to be filtered and split.
    """
    
    # Create a deep copy of the original dataset to avoid modifying it directly
    dataset = original_dataset.copy()
    _logger.info(f"Started filtering dataset '{args.dataset_name}' using {args.filter_file} file")
    _logger.info(f"\tSize before filtering: {len(dataset['train_val'])}")

    if args.filter_file is not None:
        # Attempt to read the filter file with comma or tab separator
        ids_selected = read_csv_corrupt(os.path.join(args.base_dir, "filtering", args.filter_file), cols_used=None)

        # Convert 'id' column from the filter file into a set for efficient searching
        selected_ids_set = set(ids_selected['id'])

        # Filter the 'train_val' split of the dataset
        train_val_dataset = dataset['train_val'].filter(lambda example: example['id'] in selected_ids_set)

    else:
        train_val_dataset = dataset['train_val']

    _logger.info(
        f"\tSize after filtering: {len(train_val_dataset)}\n")

    # Subset sample_size sentences only for training-val splits
    if args.sample_size is not None:
        train_val_dataset = subset_trainval_set(train_val_dataset, n_sentences=args.sample_size, seed=args.seed)
        _logger.info(f"Succesfully sampled {args.sample_size} sentences using seed={args.seed}")

    # Stratified split of the filtered 'train_val' dataset
    train_val_split = train_val_dataset.train_test_split(test_size=0.2, stratify_by_column='src')

    # Update the dataset with new 'train' and 'val' splits
    dataset['train'] = train_val_split['train']
    dataset['val'] = train_val_split['test']

    _logger.info(f"Performed stratified splits")
    _logger.info(f"\ttrain_size: {len(dataset['train'])}, valid_size: {len(dataset['val'])}, test_size: {len(dataset['test'])}")

    # Remove the 'train_val' split
    del dataset['train_val']

    return dataset




def get_run_metadata(args):
    '''
    Function to extarct metadata from filtering file for run
    '''
    
    if args.filter_file is None: 
        return None, None, f"Baseline_seed-{args.seed}_size-{args.sample_size}"
    
    filename_split = args.filter_file.split(".")[0].split("_")
    if len(filename_split) < 2:
        raise ValueError(f"Invalid format for 'filter_file': {args.filter_file}")
    
    method = filename_split[0]
    level = filename_split[-1]

    # Create a subname by combining 'method' and 'level'
    subname = f"{method}_{level}_seed-{args.seed}_size-{args.sample_size}"

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
    tokenizer_folder = os.path.join(args.base_dir, "tokenizers") # where tokenized dataset is
    training_folder = ensure_dir(os.path.join(args.base_dir, "training", model_subname))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("run_%Y-%m-%d_%H-%M-%S")
    # logs_folder = ensure_dir(os.path.join(args.base_dir, "logs", model_subname, formatted_time))
    logs_folder = ensure_dir(os.path.join(args.base_dir, "logs", model_subname))

    # Log parameters and method
    _logger.info(f"Parameters: {args}")
    _logger.info(f"Methods: {method}, level: {level}")

    
    # Load tokenizer and tokenized dataset
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, 
                                                    src_lang=MAPPING_LANG[args.source_lang], 
                                                    tgt_lang=MAPPING_LANG[args.target_lang]
                                                    )  
    tokenized_datasets = datasets.load_from_disk(os.path.join(tokenizer_folder, args.dataset_name))


    # Perform filtering
    tokenized_datasets_filt = filter_and_sample_and_split_dataset(args, tokenized_datasets)


    # MODEL TRAINING -----------------------------------------------------------
    
    # Load JSON configuration file for training hyperparameters
    # see: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

    with open(args.config_path, 'r') as file:
        hyperparams = json.load(file)
    # Log configuration
    _logger.info("Training configuration:")
    _logger.info(json.dumps(hyperparams, indent=4, sort_keys=True))

    # Load the model
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Define training arguments using hyperparameters from JSON
    training_args = TrainingArguments(
        output_dir=training_folder,
        logging_dir=logs_folder,
        **hyperparams
    )
    
    # Initialize Trainer
    # callbacks = [TensorBoardCallback()]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_filt['train'],
        eval_dataset=tokenized_datasets_filt['val'],
        tokenizer=tokenizer
        # callbacks=callbacks
    )

    # Start training
    _logger.info("Starting training")
    trainer.train()
    _logger.info("Finished training")

    # Save the model
    trainer.save_model(model_folder)
    _logger.info(f"Succesfully saved model in {model_folder}")



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--base_dir', type=str, default=None, required=True,
                        help='Path where all subfolders are present for the project')
    parser.add_argument('--config_path', type=str, default=None, required=True,
                        help='Path to config file in json format with hyperparams')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized",
                        help='Name of device to use')
    parser.add_argument('--filter_file', type=str, default=None,
                        help='Name of file with ids after filtering. If None, then no filtering is applied and saved into Baseline_seed-{seed}_size-{size}')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of sentences to sample for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for sampling')
    parser.add_argument('--source_lang', type=str, default="eng", choices=['eng', 'pol'],
                        help='Source language')
    parser.add_argument('--target_lang', type=str, default="pol", choices=['eng', 'pol'],
                        help='Target language')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
