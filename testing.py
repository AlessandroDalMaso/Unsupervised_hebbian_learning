"""
Created on Sun Nov  1 17:26:42 2020

@author: Ale
"""
import CHUNeuralNetwork as CHU
import numpy as np


def test_rank_finder_nonzero():
    hidden_neurons = np.array([[0,  0, -2,  3,  1,  0, 7],
                               [3,  4,  1, -2, -2, -2, 4]])
    k = 7
    (learn_hebbian, learn_anti) = CHU.rank_finder(hidden_neurons, k)
    assert np.array_equal(np.nonzero(learn_hebbian), ([0,1], [6,6]))
    assert np.array_equal(np.nonzero(learn_anti), ([0,1], [2,3]))


def test_product_shape():
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
    one_over_scale = 1
    result = CHU.plasticity_rule_hebbian(product_result, single_weight,
                                         visible_neurons, R, p, one_over_scale)


def test_plasticity_rule_anti_basic():
    product_result = np.array([1, 0, -2])
    single_weight = -2.5
    visible_neurons = np.array([1, 1, 0])
    R = 1
    p = 3
    one_over_scale = 1
    delta = -0.4
    result = CHU.plasticity_rule_anti(product_result, single_weight,
                                         visible_neurons, R, p, one_over_scale,
                                         delta)


def test_relu_positive():
    hidden_neurons = np.array([[0,  0,  0],
                               [2, -2,  3]])
    assert np.all(CHU.relu(hidden_neurons) >= 0)


def test_hidden_neurons_func_basic():
    batch = np.array([[-2,  3, -2],
                      [ 0, -2,  2],
                      [-2,-10,  2],
                      [0,   1,  0]])
    weight_matrix = np.array([[0, 1, -1],
                              [1,-1,  9]])
    activation_function = CHU.relu
    
    result = CHU.hidden_neurons_func(batch, weight_matrix, activation_function)
    prod = weight_matrix @ batch[0]
    positive = np.where(prod < 0, 0, prod)
    comparison = result[0] == positive
    assert comparison.all()

def test_plasticity_rule_shape():
    weight_vector = np.array([0, 1, -2, 3])
    batch = np.array([[ 0, 1, 0, 1],
                      [-1, 0, 0, 2],
                      [ 1, 0,-2,-1]])
    first_index = 0
    kth_index = 3
    p = 3
    one_over_scale = 1
    R = 1
    delta = -0.4

    result = CHU.plasticity_rule(weight_vector, batch, first_index, kth_index,
                                 p, one_over_scale, R, delta)
    assert result.shape == (4,)

test_rank_finder_nonzero()
test_product_shape()
test_plasticity_rule_hebbian_basic()
test_plasticity_rule_anti_basic()
test_relu_positive()
test_hidden_neurons_func_basic()
#test_plasticity_rule_shape()

