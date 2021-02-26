# Unsupervised neural network learning

Unsupervised neural network learning version 1.0 26/02/2021.

This program instances a model of neural network based on unsupervised learning, operating on the MNIST Handwritten digits dataset.

## Getting started

Required libraries:

* numpy
* time
* sklearn
* scipy
* pandas
* math
* PIL
* matplotlib
* hausdorff

Run either mnist_random.py, mnist_mono or mnist_1v1 to start the learning process. Results are saved in an HDF5 file 'results/matrices'. Classification scores are printed on screen.

Run cluster_analysis.py to perform clustering on the resulting trained matrix.

utilities.py and CHUNeuralNetwork.py are respectively a collection of utilities functions to be imported in other files and the library containing the main class on wich the program is based.


## Author

Alessandro Dal Maso

alessandro.dalmaso@protonmail.com
