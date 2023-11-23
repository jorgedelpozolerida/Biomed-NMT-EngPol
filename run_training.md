# Explanation on how to run training

There is supposed ot be some predefined folder structure that scripts expect (i created folder already in /home/data_shares/anlp_jkmj/DATA), which is:

└── args.base_dir\
    ├── data\
    │   └── *medical_corpus_clean_preprocessed.tsv*\
    │\
    ├── filtering\
    │   ├── *method1_level1.tsv*\
    │   ├── *method2_level1.tsv*\
    │   ├── *method1_level2.tsv*\
    │   ...\
    │   └── *methodn_leveln.tsv*\
    ├── logs\
    │   ├── *method1_level1*\
    │   ...\
    │   └── *methodn_leveln*\
    ├── models\
    │   ├── *method1_level1*\
    │   ...\
    │   └── *methodn_leveln*\
    ├── tokenizers\
    │   └── *dataset_tokenized*\
    └── training



First thing to do is to tokenize dataset using this script: [tokenize_dataset.py](src/tokenize_dataset.py) by running:

```bash
python tokenize_dataset.py --base_dir "/home/data_shares/anlp_jkmj/DATA" --device_name "cuda:0" \
  --dataset_name "dataset_tokenized" --source_lang "eng" --target_lang "pol" \
  --model_name "facebook/mbart-large-50-one-to-many-mmt" \
  --train_corpus_name "medical_corpus_clean_preprocessed.tsv" \
  --test_corpus_name "WHATEVERNAME_form_janos.tsv" \
  --overwrite

```

Not sure if medical_corpus_clean_preprocessed.tsv still holds afte rmikolaj changes.

once tokenize datasets exists, the idea is to create a bash script that loops over all combinations of filter filenames for arg --filter_file (let's start not setting any, which sets to none and performs no filtering) and calls [train.py](./src/train.py). Remember the script is yet not finished since I needed still to do the following:

* Set up properly args for TrainingArguments() like steps, validation steps, early stopping etc
* Do not freeze layers according to Rob, so remove that part of code
* Maybe other issues that arise

One run of the train.py script would go as: 

```bash
python train.py --base_dir "/home/data_shares/anlp_jkmj/DATA" \
                --device_name "cuda:0" \
                --dataset_name "dataset_tokenized" \
                --filter_file "LASER_above_quantile_20.tsv" \
                --source_lang "eng" \
                --target_lang "pol" \
                --num_train_epochs 3 \
                --train_batch_size 2 \
                --eval_batch_size 2
```
Validation is supposed to be done later with a different job in HPC (or locally if inference doe snot take long)




## Examples of my previous project using HPC

I had a project where we used GPUs in HPC, I believe one between "desktop" 1 to 10 (32GiB)


Pytorch conda config: [p2_torch.yml](https://github.com/jorgedelpozolerida/ComputerSystemsPerformance/blob/main/Project2/p2_torch.yml)

Bash script for running experiments: [run_experiments_GPU_pytorch.job](https://github.com/jorgedelpozolerida/ComputerSystemsPerformance/blob/main/Project2/run_experiments_GPU_pytorch.job), which is basically looping over args and inputting them to pytohn script call: [training_pytorch.py](https://github.com/jorgedelpozolerida/ComputerSystemsPerformance/blob/main/Project2/src/training_pytorch.py)



Example of scheduling tasks: [explanation of scheduling](https://github.com/jorgedelpozolerida/ComputerSystemsPerformance/blob/main/Project2/README.md)
