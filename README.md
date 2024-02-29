# FAST

This repo is the Mxnet implementation of “When Traffic-Flow Meets Shortest Path: Efficient Flow-Aware Querying in Road Networks”

# Requirements

Python >= 3.9
numpy >= 1.19.5
mxboard >= 0.1.0
mxnet >= 1.7.0
tensorboard >= 2.11.2
tensorflow >= 2.8.0

# How to train/test the model

Train:
python train.py --config configurations/xxx.conf --force True   (xxx for custom file name)

Test:
python predict.py --config configurations/xxx.conf --force True   (xxx for custom file name)

# How to use the index

We have provided a compiled FHLI querying program of "index" file

Use these commands to start the querying:
cd /index 
./FHLI
