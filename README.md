# DeeBERT

This is the code base for the paper [DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference](https://www.aclweb.org/anthology/2020.acl-main.204/).

Code in this repository is also available in the Huggingface Transformer [repo](https://github.com/huggingface/transformers/tree/master/examples/research_projects/deebert) (with minor modification for version compatibility). Check [this page](https://huggingface.co/ji-xin) for models that we have trained in advance (the latest version of Huggingface Transformers Library is needed).

## Installation

This repo is tested on Python 3.7.5, PyTorch 1.3.1, and Cuda 10.1. Using a virtulaenv or conda environemnt is recommended, for example:

```
conda install pytorch==1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

After installing the required environment, clone this repo, and install the following requirements:

```
git clone https://github.com/castorini/deebert
cd deebert
pip install -r ./requirements.txt
pip install -r ./examples/requirements.txt
```



## Usage

There are four scripts in the `scripts` folder, which can be run from the repo root, e.g., `scripts/train.sh`.

In each script, there are several things to modify before running:

* path to the GLUE dataset. Check [this](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) for more details.
* path for saving fine-tuned models. Default: `./saved_models`.
* path for saving evaluation results. Default: `./plotting`. Results are printed to stdout and also saved to `npy` files in this directory to facilitate plotting figures and further analyses.
* model_type (bert or roberta)
* model_size (base or large)
* dataset (SST-2, MRPC, RTE, QNLI, QQP, or MNLI)

#### train.sh

This is for fine-tuning and evaluating models as in the original BERT paper.

#### train_highway.sh

This is for fine-tuning DeeBERT models.

#### eval_highway.sh

This is for evaluating each exit layer for fine-tuned DeeBERT models.

#### eval_entropy.sh

This is for evaluating fine-tuned DeeBERT models, given a number of different early exit entropy thresholds.



## Citation

Please cite our paper if you find the repository useful:
```
@inproceedings{xin-etal-2020-deebert,
    title = "{D}ee{BERT}: Dynamic Early Exiting for Accelerating {BERT} Inference",
    author = "Xin, Ji  and
      Tang, Raphael  and
      Lee, Jaejun  and
      Yu, Yaoliang  and
      Lin, Jimmy",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.204",
    pages = "2246--2251",
}
```

