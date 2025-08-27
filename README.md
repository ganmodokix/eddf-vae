# Exponential Dissimilarity-Dispersion Family for Domain-Specific Representation Learning

This is the official repository of code for the paper "Exponential Dissimilarity-Dispersion Family for Domain-Specific Representation Learning" (in press).

## BibTeX Citation

If you use this code in your work, please cite our paper as follows:

*Note: This BibTeX citation is provisional because our paper is in press. Once the paper officially published, it will be updated.*

```
@article{
    author  = {Togo, Ren and Nakagawa, Nao and Ogawa, Takahiro and Haseyama, Miki},
    title   = {{E}xponential {D}issimilarity-{D}ispersion {F}amily for Domain-Specific Representation Learning},
    journal = {{IEEE} Transactions on Image Processing},
    year    = {2025},
    volume  = {},
    number  = {},
    pages   = {in press}
}
```

## Installation
We developed and tested this code in the environment as follows:

- Ubuntu 22.04
- Python3.10
- CUDA 11.8
- 1x GeForce® RTX 2080 Ti
- 31.2GiB (32GB) RAM

We recommend to run this code under the `venv` envirionment of Python 3.10.
Having installed `torch`, the requirements can be easily installed using `pip`.
```
$ python3.10 -m venv .env
$ source .env/bin/activate
(.env) $ pip install -U pip
(.env) $ # install PyTorch via pip here
(.env) $ pip install wheel
(.env) $ pip install -r requirements.txt
```
In `requirements.txt`, a third-party representation learning package `vaetc` is specified, which is downloaded from `github.com` and installed via `pip`.

## How to Train
Run `train.py` with a setting file to train models.
```
(.env) $ python train.py settings/main/mnist.yaml  # EDDF-VAE (Ours)
(.env) $ python train.py settings/cms/geco-mnist.yaml  # compared methods (cms)
...
```
The results are saved in the `logger_path` directory specified in the setting YAML file.