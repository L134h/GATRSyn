# GATRSyn
GATRSyn Project README
Overview
The GATRSyn project consists of two main components: feature extraction for drugs and cell lines using Graph Attention Networks (GAT), and prediction through Transformer feature fusion. This document provides a brief introduction to each component and how to get started with the project.

Feature Extraction - cell-drug-feat
The cell-drug-feat folder contains scripts used for extracting features from drug and cell line data using Graph Attention Networks (GAT). The primary script for this process is train.py.

Data
Within the GATRSyn\cell-drug-feat\data directory, you will need a file named node_features.npy. You can download this file using the following link:

node_features.npy( https://pan.baidu.com/s/1hSKTnLIX1JxOk0_C2iy0SA?pwd=3asq)

Note: When downloading from Baidu Netdisk, you may need to enter the extraction code: 3asq.

Prediction - prediction
The prediction folder houses scripts that are responsible for the feature fusion  and making predictions. The main script to run for cross-validation and prediction is cross_validation.py.

Getting Started
To start working with the GATRSyn project, ensure you have all dependencies installed and follow these steps:

Download the node_features.npy file from the provided link.
Navigate to the cell-drug-feat directory and run train.py to extract features if necessary.
Move to the prediction directory and use cross_validation.py to perform feature fusion and prediction tasks.
