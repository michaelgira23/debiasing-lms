# Debiasing Language Models

> Official code for _Debiasing Pre-Trained Language Models via Efficient Fine-Tuning_ published in the [Second Workshop on Language Technology for Equality, Diversity, Inclusion](https://sites.google.com/view/lt-edi-2022) at ACL 2022.

[View Demo](https://huggingface.co/spaces/michaelgira23/debiasing-lms)

**Currently placeholder. Code will be polished and published soon!** In the meantime, you can [take a look at the old code.](https://github.com/michaelgira23/debiasing-lms/tree/old)

# Dataset

Our fine-tuning dataset consists of the [WinoBias](https://github.com/uclanlp/corefBias) and [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) datasets. After cloning the Git submodules for the respective datasets, run:

```bash
python dataset/prepare.py
```

`prepare.py` combines the datasets from each repository and splits them into a training (80%), cross-validation (10%), and testing sets (10%).
