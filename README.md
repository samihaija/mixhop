# MixHop TensorFlow Implementation

Official Implementation of ICML 2019 Paper: *MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing*

If you find this code useful, please cite us as:

    @inproceedings{mixhop,
     author={Sami Abu-El-Haija AND Bryan Perozzi AND Amol Kapoor AND Hrayr Harutyunyan
             AND Nazanin Alipourfard AND Kristina Lerman AND Greg Ver Steeg AND Aram Galstyan}
     title={MixHop: Higher-Order Graph Convolution Architectures via Sparsified Neighborhood Mixing},
     booktitle = {International Conference on Machine Learning (ICML)},
     year = {2019},
    }

## File Overview

1. `mixhop_model.py`: Contaits our mixhop layer and model architecture. Use this
   file if you want to try our model on different datasets [e.g. not citation
   datasets of planetoid]. You can invoke our layer alone or our entire
   architecture. The architecture can be saved to disk (as JSON) and loaded
   later.
1. `mixhop_dataset.py`: This reads the [planetoid](https://github.com/kimiyoung/planetoid)
   datasets.
1. `mixhop_trainer.py`: End-to-end training and evaluation over the
   [planetoid](https://github.com/kimiyoung/planetoid) datasets. You probably
   want to start by invoking/modifying the shell scripts directly
   (e.g. `planetoid.sh`).

## How to use `mixhop_trainer.py`

This implementation relies on the datasets of planetoid living on your machine.
Unless you override the flag `--dataset_dir` in `mixhop_trainer`, code expects
that [planetoid](https://github.com/kimiyoung/planetoid) repo is cloned on
`~/data`. To clone it, you can run:
```
# Line clones plenetoid if it does not exist.
mkdir -p ~/data && cd ~/data && ls planetoid || git clone git@github.com:kimiyoung/planetoid.git
```

Then, we advice you to run the shell script which have good hyper-parameter values:

```
bash train_pubmed.sh
```

## Need help?

Please help us by reaching out to sami@haija.org -- Whenever possible, we would
like to improve the quality of the code and any resolve ambiguities.

## To be completed!

This code provides the complete MixHop Graph Conv Layer and Architecture,
however, it is still missing (1) Cora + Citeseer run scripts and (2) The
two-phase training. Reason for delay: Our original code is _researchy_ i.e.
not pleasant to read [you know how it goes: you try a bunch of things, until
something works, without removing the things that did not work, producing one
huge file].
The code will be completely ready, and up to our coding standards, by the ICML
conference. If you want the code sooner, please email sami@haija.org and we are
happy to provide you with the messy version or prioritize the clean-up
accordingly.

