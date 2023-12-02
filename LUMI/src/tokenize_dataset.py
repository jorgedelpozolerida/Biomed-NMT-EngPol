#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Tokenize dataset for later use


This script generates tokenized dataset to be used for training and testing. 
It creates ands saves into data/tokenizers a dataset where the 'test' split belongs
to unbiased dataset and 'train_val' is the data that is yet to be filtered before 
fine-tuning. 


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
import argparse
import logging                                                                      

import pandas as pd                                                                 
import torch
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import MBart50Tokenizer


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

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

def preprocess_function(examples, tokenizer, 
                        src_col = "eng", target_col = "pol"):
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

    model_inputs = tokenizer(inputs, max_length=210, truncation=True, padding='max_length', return_tensors='pt')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=210, truncation=True, padding='max_length', return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"].squeeze()
    
    return model_inputs

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def main(args):

    assert os.path.isdir(args.base_dir)
    assert args.source_lang != args.target_lang

    data_folder = ensure_dir(os.path.join(args.base_dir, "data"))
    tokenizer_folder = ensure_dir(os.path.join(args.base_dir, "tokenizers")) # where to save tokenized dataset
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    dataset_dir = os.path.join(tokenizer_folder, args.dataset_name)
    if os.path.exists(dataset_dir):
        raise ValueError(f"Dataset dir exists already, remove: {dataset_dir}")        


    # Load tokenizer
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, 
                                                    src_lang=MAPPING_LANG[args.source_lang], 
                                                    tgt_lang=MAPPING_LANG[args.target_lang]
                                                    )
    _logger.info("Loaded tokenizer")

    # Mapping function with additional arguments
    preprocess_args = {
        "tokenizer": tokenizer,
        "src_col": args.source_lang,
        "target_col": args.target_lang
    }


    # TRAINING dataset ---------------------------------------------------------

    # Load the training dataset
    df_train = read_csv_corrupt(
        os.path.join(data_folder, args.train_corpus_name),
        cols_used=['id', 'pol', 'eng', 'src']
        )

    # Load the training dataset into Hugging Face datasets
    train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, **preprocess_args), batched=True)

    # handle src column to allow future stratified splits
    unique_classes = sorted(set(tokenized_train['src']))
    class_label_feature = ClassLabel(names=unique_classes)
    tokenized_train = tokenized_train.cast_column('src', class_label_feature)
    _logger.info("Training-Val set generated")

    # TEST dataset -------------------------------------------------------------

    df_test = read_csv_corrupt(
        os.path.join(data_folder, args.test_corpus_name),
        cols_used=['id', 'pol', 'eng']
        )

    # Load the test dataset into Hugging Face datasets
    test_dataset = Dataset.from_pandas(df_test, preserve_index=False)
    tokenized_test = test_dataset.map(lambda x: preprocess_function(x, **preprocess_args), batched=True)
    _logger.info("Test set generated")


    # COMBINED -----------------------------------------------------------------

    # Combine the splits to form the final dataset
    final_dataset = DatasetDict({
        'train_val': tokenized_train,
        'test': tokenized_test
    })

    final_dataset.save_to_disk(dataset_dir)
    _logger.info(f"Succesfully tokenized dataset into: {dataset_dir}")

def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default=None, required=True,
                        help='Path where all subfolders are present for the project')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized",
                        help='Name of dataset to create')
    parser.add_argument('--source_lang', type=str, default="eng", choices=['eng', 'pol'],
                        help='Source language')
    parser.add_argument('--target_lang', type=str, default="pol", choices=['eng', 'pol'],
                        help='Target language')
    parser.add_argument('--train_corpus_name', type=str, default="trainval_set_v3.tsv",
                        help='Name of training corpus. Train-val splits taken from here.')
    parser.add_argument('--test_corpus_name', type=str, default="test_set_v3.tsv", #TODO: change this
                        help='Name of test corpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)