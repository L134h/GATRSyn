# GATRSyn
GATRSyn Project README
Overview
GATRSyn is a generalized two-stage research framework that decomposes the task of drug synergy prediction into two key modules: offline feature extraction and multi-scenario adaptive prediction. By integrating drug-cell-associated protein-protein interaction (PPI) networks with pharmacogenomics data, GATRSyn employs a Graph Attention Network (GAT) enriched with edge information to construct offline feature extractors for generating cell line/drug GAT features. The transformer-based re-embeddings of GAT features, referred to as Trans features, are designed to enhance cross-modal interactions between specific drugs and cell lines, making them more adaptable to diverse downstream tasks. Finally, sample Trans features are input into deep neural networks to predict six distinct scenarios.
This document provides a brief introduction to each component and how to get started with the project.

Feature Extraction - cell-drug-feat
The cell-drug-feat folder contains scripts used for extracting features from drug and cell line data using Graph Attention Network(GAT). The primary script for this process is train.py.

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
