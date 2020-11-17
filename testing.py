"""
Created on Sun Nov  1 17:26:42 2020

@author: Ale
"""
import CHUNeuralNetwork as CHU
import numpy as np


def test_rank_finder_basic():
    hidden_neurons = np.array([[0,  0, -2,  3,  1,  0, 7],
                               [3,  4,  1, -2, -2, -2, 4]])
    k = 7
    (learn_hebbian, learn_anti) = CHU.rank_finder(hidden_neurons, k)


def test_product_shape():
    weights = np.array([-0.4, 1])
    input_vector = np.array([0, 8])
    p = 3
    CHU.product(weights, input_vector, p)


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
    input_vector = np.array([ 0, 1, 0, 1])
    g = -0.4
    p = 3
    one_over_scale = 1
    R = 1

    result = CHU.plasticity_rule(weight_vector, input_vector, g, p, R,
                                 one_over_scale)

def test_plasticity_rule_vectorized():
    batch = np.array([[-2,  3, -2],
                      [ 0, -2,  2],
                      [-2,-10,  2],
                      [0,   1,  0]])
    weight_matrix = np.array([[0, 1, -1],
                              [1,-1,  9]])
    delta = 0.4
    p = 3
    R = 1
    one_over_scale = 1
    indexes_hebbian = [0, 1, 0, 1]
    indexes_anti = np.array([1,0,1,0])
    result = CHU.plasticity_rule_vectorized(weight_matrix, batch, delta, p,
                                            R, one_over_scale,
                                            indexes_hebbian, indexes_anti)
    batch2 = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    result2 = CHU.plasticity_rule_vectorized(weight_matrix, batch2, delta, p,
                                            R, one_over_scale,
                                            indexes_hebbian, indexes_anti)
    assert np.array_equal(result2, np.zeros(weight_matrix.shape))

test_rank_finder_basic()
test_product_shape()
test_relu_positive()
test_plasticity_rule_shape()
test_hidden_neurons_func_basic()
test_plasticity_rule_vectorized()

