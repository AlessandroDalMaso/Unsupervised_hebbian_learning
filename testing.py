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
    weights = np.array([-0.4, 1])
    batch = np.array([[ 0,   -0.4],
                      [ 3,    3  ],
                      [-2,    4  ]])
    p = 3
    assert CHU.product(weights, batch, p).shape == (3,)
    


def test_plasticity_rule_hebbian_basic():
    product_result = np.array([-0.3, 1, 0])
    single_weight = 0.2
    visible_neurons = np.array([0, -2, 3])
    R = 1
    p = 3
    one_over_time_scale = 1
    result = CHU.plasticity_rule_hebbian(product_result, single_weight,
                                   visible_neurons, R, p, one_over_time_scale)
    assert result.shape == (3,)


test_rank_finder_basic()
test_product_basic()
test_plasticity_rule_hebbian_basic()
