# 13_ChaLearn_ImageProcessingRegression

## About this Project
The project references a large dataset from ChaLearn stored at (this website)[https://gesture.chalearn.org/]. Please download if wishing to run the analysis yourself.

A common business use problem is the need for running model training on cloud servers. In the case of python machine learning, this involves sending a dataset and a python script to a more powerful server than the scientist has access to. In this project we want to train a model on images to estimate their age. We'll prepare and run a publicly-available dataset of faces through a neural network.

The project demonstrates how best to review and process images for a ResNet50 model. It also features the script generation and model output from the server. The script was run on Google Colab, which can be found [here](https://colab.google/).

## Running it Yourself

The Jupyter notebook is self-contained and reflects the outputs of the code contained within. If you would like to connect to your own environment and run the notebook, or make changes on your own fork of the repo, you may do so after cloning. Make sure the environment is either based in the upper-level folder to which you clone the repo, or be sure to replace the dataset file references with a direct local reference to the /datasets/ folder on your machine.

A significant portion of this project was run on Google Colab. Ensure your colab environment has access to your choice of face data downloaded from ChaLearn, and has tensorflow 2.10.0.

The project is stable with Python 3.11.
