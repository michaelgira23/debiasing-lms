## Overview

This is the codebase for paper _Debiasing Pre-Trained Language Models via Efficient Fine-Tuning_. It contains code to reproduce our experiments and to evaluate models with StereoSet.

## Usage

### Set up the environment

Install Anaconda environment:
` $ conda env create -f environment.yml `

### Reproduce our fine-tuning experiments with GPT-2

You can run experiments with:
` $ python experimental_matrix_timed.py `

### Evaluate a model with StereoSet metrics

You can evaluate a model with StereoSet metrics with:
` $ python eval_stereoset.py `

### Generate Texts

You can generate the texts given a prompt using our fine-tuned model with:
` $ python generate_samples_unprejudiced.py `
