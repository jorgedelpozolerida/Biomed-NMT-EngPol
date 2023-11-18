#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" {Short Description of Script}


This script generates tokenizeddataset to be used for training and testing. 
It creates ands saves into data/tokenizers a dataset where the 'test' split belongs
to unbiased dataset and 'train_val' is the data that is yet to be filtered before 
fine-tuning. 

TODO:
- incorporate dataset for testing

Note on folder structure:
└── args.base_dir
    ├── data
    │   
    ├── filtering
    │   
    ├── logs
    │   
    ├── models
    │   ├── *submodel_1*
    │   ├── *submodel_2*
    │   ...
    │   └── *submodel_N*
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


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


from typing import Dict, Any
import os
import torch
import pandas as pd
import datasets
from datasets import DatasetDict
from transformers import MBart50Tokenizer
from sklearn.model_selection import train_test_split
import shutil

# Mapping of column names for languages to mBART50 language codes
MAPPING_LANG: Dict[str, str] = {
    "pol": "pl_PL",
    "eng": "en_XX"
}

def preprocess_function(examples: Dict[str, Any], tokenizer: MBart50Tokenizer, 
                        src_col: str = "pol", target_col: str = "eng") -> Dict[str, torch.Tensor]:
    """
    Preprocess input data for model training.

    Args:
        examples: A batch of examples from the dataset.
        tokenizer: The tokenizer to use for encoding the text.
        src_col: The source column name.
        target_col: The target column name.

    Returns:
        A dictionary containing tokenized inputs and labels.
    """
    inputs = [doc for doc in examples[src_col]]
    targets = [doc for doc in examples[target_col]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"].squeeze()
    
    return model_inputs


def main(args: Any):

    assert os.path.isdir(args.base_dir)
    assert args.source_lang != args.target_lang

    data_folder = os.path.join(args.base_dir, "data")
    models_folder = os.path.join(args.base_dir, "models")
    tokenizer_folder = os.path.join(args.base_dir, "tokenizers")
    results_folder = os.path.join(args.base_dir, "results")
    logs_folder = os.path.join(args.base_dir, "logs")

    # Load the training dataset
    df_train = pd.read_csv(
        os.path.join(data_folder, args.train_corpus_name),
        usecols=['id', 'pol', 'eng', 'src']
    ).head(50) # TODO: remove head

    # Load the test dataset
    df_test = pd.read_csv(
        os.path.join(data_folder, args.test_corpus_name),
        usecols=['id', 'pol', 'eng']
    ).head(50) # TODO: remove head

    # Use a GPU if available and specified
    if "gpu" in args.device_name.lower():
        device = torch.device(f"cuda:{args.device_name}" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    _logger.info(f"Using device: {device}")

    # Load the training dataset into Hugging Face datasets
    train_dataset = datasets.Dataset.from_pandas(df_train, preserve_index=False)

    # Load the test dataset into Hugging Face datasets
    test_dataset = datasets.Dataset.from_pandas(df_test, preserve_index=False)

    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name, 
                                                    src_lang=MAPPING_LANG[args.source_lang], 
                                                    tgt_lang=MAPPING_LANG[args.target_lang]
                                                    )  

    # Mapping function with additional arguments
    preprocess_args = {
        "tokenizer": tokenizer,
        "src_col": args.source_lang,
        "target_col": args.target_lang
    }

    # Training dataset ----------
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, **preprocess_args), batched=True)
    # handle src column to allow future stratified splits
    unique_classes = sorted(set(tokenized_train['src']))
    class_label_feature = datasets.ClassLabel(names=unique_classes)
    tokenized_train = tokenized_train.cast_column('src', class_label_feature)

    # Test dataset --------------
    tokenized_test = test_dataset.map(lambda x: preprocess_function(x, **preprocess_args), batched=True)

    # Combine the splits to form the final dataset
    final_dataset = datasets.DatasetDict({
        'train_val': tokenized_train,
        'test': tokenized_test
    })

    dataset_dir = os.path.join(tokenizer_folder, args.dataset_name)
    if os.path.exists(dataset_dir) and args.overwrite:
        shutil.rmtree(dataset_dir)
        _logger.info(f"Succesfully removed dir before overwriting: {dataset_dir}")
        final_dataset.save_to_disk(dataset_dir)
    elif not os.path.exists(dataset_dir): 
        final_dataset.save_to_disk(dataset_dir)
    else:
        _logger.warning(f"Could not save dataset since directory already exists. Add flag --overwrite if intended")
        

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--base_dir', type=str, default=None, required=True,
                        help='Path where all subfolders are present for the project')
    parser.add_argument('--device_name', type=str, default=None, required=True,
                        help='Name of device to use')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized",
                        help='Name of dataset to create')
    parser.add_argument('--source_lang', type=str, default="pol", choices=['eng', 'pol'],
                        help='Source language')
    parser.add_argument('--target_lang', type=str, default="eng", choices=['eng', 'pol'],
                        help='Target language')
    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-50-one-to-many-mmt",
                        choices=["facebook/mbart-large-50-one-to-many-mmt", "facebook/mbart-large-50-many-to-many-mmt"],
                        help='Path to the input directory')
    parser.add_argument('--train_corpus_name', type=str, default="medical_corpus_clean_preprocessed.tsv",
                        help='Name of training corpus. Train-val splits taken from here.')
    parser.add_argument('--test_corpus_name', type=str, default="medical_corpus_clean_preprocessed.tsv", #TODO: change this
                        help='Name of test corpus')
    parser.add_argument('--overwrite',  default=False, action='store_true',
                            help='Add this flag if you want to overwrite dataset folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)