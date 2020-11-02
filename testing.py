"""
Created on Sun Nov  1 17:26:42 2020

@author: Ale
"""
import CHUNeuralNetwork as CHU
import numpy as np


def test_rank_finder_basic():
    hidden_neurons = np.array([[0,  0, -2,  3,  1,  0, 7],
                               [3,  4,  1, -2, -2, -2, 4]])
    (first_neurons, seventh_neurons) = CHU.rank_finder(hidden_neurons)
    assert not np.any(hidden_neurons[[0, 1], first_neurons] - [7, 4])


def test_product_basic():
    weight_matrix = np.array([[ 0, 1],
                              [-2, 0]])
    batch = np.array([[ 0,   -0.4],
                      [ 3,    3  ],
                      [-2,    4  ]])
    p = 3
    result = CHU.product(weight_matrix, batch, p)
    # TODO think of a test!
    
    
test_rank_finder_basic()
