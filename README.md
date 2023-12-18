<p align="center">
    <img src="https://itu.dk/svg/itu/logo_dk.svg">
</p>

# Advanced Natural Language Processing - Final Project

Repository containing source code, data and results of final Group Project for Advanced Natural Language Processing and Deep Learning, course at [IT University of Copenhagen](https://en.itu.dk/).

Experiments were run on [LUMI supercomputer](https://www.lumi-supercomputer.eu/about-lumi/) and full report 
can be found here: [Advanced_NLP_report.pdf](Advanced_NLP_report.pdf).
<span style="color: red;">TODO: update report</span>


## Authors
The authors of the projects are the following:
- [*Jorge del Pozo Lerida*](https://github.com/jorgedelpozolerida), MSc Data Science (ITU)
- [*Kamil Kojs*](https://github.com/KamilKojs), MSc Data Science (ITU)
- [*Janos Mate*](https://github.com/matejanos), MSc Data Science (ITU)
- [*Mikolaj Baranski*](https://github.com/MikolajBaranski), MSc Data Science (CBS)



## Abstract
<span style="color: red;">TODO: add </span>

## Results

<span style="color: red;">TODO: include conclusion and tables with performance here  </span>


## Repository structure overview

These files are present in root folder:
* [Advanced_NLP_report.pdf](Advanced_NLP_report.pdf): PDF with full report of the project
* [Bibliography](bibliography.bib): bib file containing bibliography used

The following folders are present in the repo:

### [Data](data/)
Contains all light-weight files used in the project, like evaluaiton results and filtered ids
from original corpus.

### [LUMI](LUMI/)
This folder contains all scripts, logs and slurm jobs executed on LUMI supercomputer 
during experimentation. 

The following scripts are worth mentioning:
* [tokenize_dataset.py](LUMI/src/tokenize_dataset.py): script to generate tokenized splits from different sources: train, validation and test sets.
* [train.py](LUMI/src/train.py): script to execute training of mBART50
* [evaluate.py](LUMI/src/evaluate.py): script to execute evaluation on test set


### [Local scripts](src/)
This folder hosts Python scripts and notebooks run locally used for several tasks
of the project.
* [BERT_embedding_gen.py](src/BERT_embedding_gen.py): calculates the similiraty between English and Polish sentences with BERT model
* [BERT_filtering_levels.ipynb](src/BERT_filtering_levels.ipynb): generate different levels of filtering with BERT model
* [muse.ipynb](src/muse.ipynb): calculates the similiraty between English and Polish sentences with MUSE model
* [generate_LaBSE_embeddings_similarity.ipynb](src/generate_LaBSE_embeddings_similarity.ipynb): calculates the similiraty between English and Polish sentences with LaBSE model
* [LaBSE_quantile_split.ipynb](src/LaBSE_quantile_split.ipynb): calculates the similiraty between English and Polish sentences with LaBSE model
* [training_analyses.ipynb](src/training_analyses.ipynb): generate plots for the training steps

<span style="color: red;">TODO: include mains scripts and brief explanation  </span>

