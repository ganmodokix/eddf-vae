# Exponential Dissimilarity-Dispersion Family for Domain-Specific Representation Learning

This is the official repository of code for the paper ["Exponential Dissimilarity-Dispersion Family for Domain-Specific Representation Learning"](https://ieeexplore.ieee.org/document/11175279) ([IEEE TIP](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83)).

## BibTeX Citation

If you use this code in your work, please cite our paper as follows:

```
@article{
    author  = {Togo, Ren and Nakagawa, Nao and Ogawa, Takahiro and Haseyama, Miki},
    title   = {{E}xponential {D}issimilarity-{D}ispersion {F}amily for Domain-Specific Representation Learning},
    journal = {{IEEE} Transactions on Image Processing},
    year    = {2025},
    volume  = {34},
    number  = {},
    pages   = {6110-6125},
    doi     = {10.1109/TIP.2025.3608661}
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
