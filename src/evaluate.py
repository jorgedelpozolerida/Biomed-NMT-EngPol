#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script to evaluate mBART50 model on test data

This script does the following:
- loads tokenized dataset and trained model
- calculates BLEU and sacreBLEU metrics

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
import logging
import datasets
from sklearn.metrics import accuracy_score
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_metric

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def load_tokenized_dataset(base_dir, dataset_name):
    tokenizer_folder = os.path.join(base_dir, "tokenizers")
    return datasets.load_from_disk(os.path.join(tokenizer_folder, dataset_name))

def load_local_model(base_dir, model_name):
    models_folder = os.path.join(base_dir, "models")
    return MBartForConditionalGeneration.from_pretrained(os.path.join(models_folder, model_name))

def evaluate_model(model, tokenizer, test_dataset, source_lang, target_lang):
    model.eval()
    preds, labels = [], []

    for example in test_dataset:
        input_ids = tokenizer.encode(example[source_lang], return_tensors='pt')
        outputs = model.generate(input_ids)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = example[target_lang]

        preds.append(pred)
        labels.append([label])  # Note: labels need to be a list of lists for nltk's corpus_bleu

    # Compute SacreBLEU
    sacrebleu_metric = load_metric('sacrebleu')
    sacrebleu_results = sacrebleu_metric.compute(predictions=preds, references=labels)
    sacrebleu_score = sacrebleu_results["score"]

    # Compute NLTK BLEU
    nltk_bleu_score = corpus_bleu(labels, preds)

    return sacrebleu_score, nltk_bleu_score

def main(args):
    
    assert os.path.isdir(args.base_dir)
    assert args.source_lang != args.target_lang
    
    model_name = "facebook/mbart-large-50-one-to-many-mmt"

    # Load tokenized dataset
    tokenized_datasets = load_tokenized_dataset(args.base_dir, args.dataset_name)
    test_dataset = tokenized_datasets['test']

    # Load tokenizer
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    models_folder = os.path.join(args.base_dir, "models")
    assert os.path.isdir(models_folder)

    # Evaluate pretrained model
    pretrained_model = MBartForConditionalGeneration.from_pretrained(model_name)
    sacrebleu_score, nltk_bleu_score = evaluate_model(pretrained_model, tokenizer, test_dataset, args.source_lang, args.target_lang)
    _logger.info(f'Pretrained Model\n' + f'\tSacreBLEU: {sacrebleu_score}\n' + f'\tNLTK BLEU: {nltk_bleu_score}\n')
    
    
    # Loop over models in the models directory
    for model_name in os.listdir(models_folder):
        if args.submodel_name is not None and model_name != args.submodel_name:
            continue
        msg = ''
        model_path = os.path.join(models_folder, model_name)
        if os.path.isdir(model_path):
            msg += f"Model: {model_name}\n"
            model = load_local_model(args.base_dir, model_path)
            sacrebleu_score, nltk_bleu_score = evaluate_model(model, tokenizer, test_dataset, args.source_lang, args.target_lang)
            msg +=f'\tSacreBLEU: {sacrebleu_score}\n'
            msg += f'\tNLTK BLEU: {nltk_bleu_score}'
            _logger.info(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mBART50 Model")

    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for the project')
    parser.add_argument('--submodel_name', type=str,  help='Name of the single fine-tuned model inside models/ folder')
    parser.add_argument('--dataset_name', type=str, default="dataset_tokenized", help='Name of the tokenized dataset')
    parser.add_argument('--source_lang', type=str, default="eng", choices=['eng', 'pol'], help='Source language')
    parser.add_argument('--target_lang', type=str, default="pol", choices=['eng', 'pol'], help='Target language')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
