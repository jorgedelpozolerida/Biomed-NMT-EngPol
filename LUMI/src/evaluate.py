#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for evaluating fine-tuned models

Loops over subfolders in DATA/models loads model and tests on test set from
tokenizers/dataset_tokenized. Also performs evaluaiton on pre-trained model with
no fine-tuning.

@author: jorgedelpozolerida
@date: 05/12/2023
"""
# Included in Python
import os
import argparse
import logging
import time
import sys
import datetime
import random

# Installed apart in container
import pandas as pd
import torch
import datasets
from tqdm import tqdm

from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import sentencepiece as spm  # Just to make sure it is there

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Mapping of column names for languages to mBART50 language codes
MAPPING_LANG = {
    "pol": "pl_PL",
    "eng": "en_XX"
}

PRETRAINED_MODEL_NAME = "facebook/mbart-large-50-one-to-many-mmt"


def load_model(args, model_dir, model_name="facebook/mbart-large-50-one-to-many-mmt"):
    # Check if custom model is available in the directory; otherwise, load default model
    model_path = os.path.join(model_dir, model_name) if os.path.exists(os.path.join(model_dir, model_name)) else model_name
    
    model = MBartForConditionalGeneration.from_pretrained(model_path)

    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(f"Using device: {device}")
    model.to(device)

    return model


def generate_predictions(args, model, tokenizer, test_data, model_name):
    device = model.device
    results = []

    for example in tqdm(test_data):
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
        eng_text = example['eng']
        target = example[args.target_lang]

        generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
                                            forced_bos_token_id=tokenizer.lang_code_to_id[MAPPING_LANG[args.target_lang]])
        prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        results.append({
            'model_name': model_name, 
            'id': example['id'],
            'eng': eng_text,
            'prediction': prediction,
            'groundtruth': target
        })

    return results

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def compute_bleu_score(predictions, references):
    bleu = datasets.load_metric('sacrebleu')
    bleu.add_batch(predictions=predictions, references=references)
    return bleu.compute()

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

def get_training_eval_set(args, train_val_dataset, sample_size, seed ):
    """
    Stratified split
    """

    # Subset sample_size sentences only for training-val splits
    train_val_dataset = subset_trainval_set(train_val_dataset, n_sentences=sample_size, seed=seed)
    _logger.info(f"Succesfully sampled {sample_size} sentences using seed={seed}")

    # Stratified split of the filtered 'train_val' dataset
    train_val_split = train_val_dataset.train_test_split(test_size=0.2, stratify_by_column='src')

    # Remove the 'train_val' split
    del train_val_split['train']

    return train_val_split['test']

def get_matadata(model_name):
    """ Gets metadata from training """
    if model_name != PRETRAINED_MODEL_NAME:
        name, seed, size = model_name.split("_")
        seed = int(seed.split("-")[-1])
        size = int(size.split("-")[-1])
    else:
        name, seed, size = None, 42, 700000
    
    return name, seed, size

def main(args):
    
    
    
    models_folder = os.path.join(args.base_dir, "models")  # where models are saved
    tokenizer_folder = os.path.join(args.base_dir, "tokenizers")  # where tokenized dataset is
    save_dir = os.path.join(args.base_dir, "evaluation")

    # Load only the 'test' subset from the tokenized dataset
    if args.split_name == "test":
        subf = "test"
    else:
        subf = "train_val"
    _logger.info(f"Proceeding to load from tokenized dataset split: {subf}")
    all_data_path = os.path.join(tokenizer_folder, args.dataset_name)
    tokenized_data = datasets.load_from_disk(all_data_path)


    _logger.info(f"Correctly loaded tokenized test set")

    results = []
    print("--------------------------------------------------")
    print("Results of evaluation")
    tokenizer = MBart50Tokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME,
        src_lang=MAPPING_LANG[args.source_lang])

    model_names = [PRETRAINED_MODEL_NAME] + os.listdir(models_folder)
    _logger.info(f"Evaluating the following models: {model_names}")
    results = []
    all_translations = []
    for model_name in model_names:
        try:
            start_time = time.time()
            model = load_model(args, model_name)
            _logger.info(f"Correctly loaded model: {model_name}")
            
            if args.split_name == 'test':
                test_data_temp = tokenized_data[subf]
            else:
                name, seed, size = get_matadata(model_name)
                _logger.info(f"Using the following params for sampling: seed={seed}, size={size}" )
                test_data_temp = get_training_eval_set(args, tokenized_data[subf], size, seed)
                

            translation_results = generate_predictions(args, model, tokenizer, test_data_temp, model_name)
            bleu_score = compute_bleu_score([t['prediction'] for t in translation_results], 
                                                [[t['groundtruth']] for t in translation_results])
            elapsed_time = time.time() - start_time

            results.append({
                    'model': model_name, 
                    'bleu_score': bleu_score['score'], 
                    'elapsed_time': elapsed_time
            })
            print({
                    'model': model_name, 
                    'bleu_score': bleu_score['score'], 
                    'elapsed_time': elapsed_time
            })
            all_translations.extend(translation_results)
        except Exception as e:
            _logger.error(f"Error evaluating model in {model_name}: {str(e)}")

    # SAVING --------
    
    # Define current time for folder naming
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("run_%Y-%m-%d_%H-%M-%S")
    save_folder = ensure_dir(os.path.join(save_dir, f"{formatted_time}_{args.split_name}"))

    # Results DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_folder, f'evaluation_results.tsv'), index=False, sep='\t')
    
    # DataFrame for all translations
    df_translations = pd.DataFrame(all_translations)

    # Iterate through unique model names and save their respective translations
    for model_name in df_translations['model_name'].unique():
        model_translations = df_translations[df_translations['model_name'] == model_name]
        if model_name == "facebook/mbart-large-50-one-to-many-mmt":
            model_name = "Pretrained"
        model_save_folder = ensure_dir(os.path.join(save_folder, "predictions"))
        model_translations.to_csv(os.path.join(model_save_folder, f'{model_name}_translations.tsv'), index=False, sep='\t')

    _logger.info("Evaluation results and translations saved to TSV files.")
    
def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default=None, required=True,
                        help='Path where all subfolders are present for the project')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized",
                        help='Name of device to use')
    parser.add_argument('--split_name', type=str, default="test",choices=['test', 'eval'],
                        help='Split to evaluate on')
    parser.add_argument('--source_lang', type=str, default="eng", choices=['eng', 'pol'],
                        help='Source language')
    parser.add_argument('--target_lang', type=str, default="pol", choices=['eng', 'pol'],
                        help='Target language')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
