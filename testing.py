"""Test public functions of CHUNeuralNetwork.py."""
import CHUNeuralNetwork as chu
import numpy as np


def test_rank_finder_shape():
    batch = np.array([[-2,  3, -2],
                      [ 0, -2,  2],
                      [-2,-10,  2],
                      [ 0,  1,  0]])
    weight_matrix = np.array([[0,  1, -1],
                              [1, -1,  9]])
    k = 2
    activation_function = chu.relu
    (learn_hebbian, learn_anti) = chu.rank_finder(batch, weight_matrix,
                                                  activation_function, k)
    assert learn_hebbian.shape == (4,)
    assert learn_anti.shape == (4,)


def test_product_odd():
    weights = np.array([-0.4, 1])
    input_vector = np.array([0, 8])
    p = 3
    result1 = chu.product(weights, input_vector, p)
    result2 = chu.product(weights, -1*input_vector, p)
    assert -1*result1 == result2


def test_relu_positive():
    hidden_neurons = np.array([[0,  0,  0],
                               [2, -2,  3]])
    assert np.all(chu.relu(hidden_neurons) >= 0)


def test_hidden_neurons_func_positive():
    batch = np.array([[-2,  3, -2],
                      [ 0, -2,  2],
                      [-2,-10,  2],
                      [ 0,  1,  0]])
    weight_matrix = np.array([[0,  1, -1],
                              [1, -1,  9]])
    activation_function = chu.relu

    result = chu.hidden_neurons_func(batch, weight_matrix, activation_function)
    prod = weight_matrix @ batch[0]
    positive = np.where(prod < 0, 0, prod)
    comparison = result[0] == positive
    assert comparison.all()

def test_plasticity_rule():
    weight_vector = np.array([0, 1, -2, 3])
    input_vector = np.array([0, 1, 0, 1])
    g = -0.4
    p = 3
    one_over_scale = 1
    R = 1

    result = chu.plasticity_rule(weight_vector, input_vector, g, p, R,
                                 one_over_scale)

def test_plasticity_rule_vectorized_null():
    batch = np.array([[-2,  3, -2],
                      [ 0, -2,  2],
                      [-2,-10,  2],
                      [ 0,  1,  0]])
    weight_matrix = np.array([[0,  1, -1],
                              [1, -1,  9]])
    delta = 0.4
    p = 3
    R = 1
    one_over_scale = 1
    indexes_hebbian = [0, 1, 0, 1]
    indexes_anti = np.array([1 ,0, 1, 0])
    result = chu.plasticity_rule_vectorized(weight_matrix, batch, delta, p,
                                            R, one_over_scale,
                                            indexes_hebbian, indexes_anti)
    batch2 = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    result2 = chu.plasticity_rule_vectorized(weight_matrix, batch2, delta, p,
                                             R, one_over_scale,
                                             indexes_hebbian, indexes_anti)
    assert np.array_equal(result2, np.zeros(weight_matrix.shape))


test_rank_finder_shape()
test_product_odd()
test_relu_positive()
test_hidden_neurons_func_positive()
test_plasticity_rule()
test_plasticity_rule_vectorized_null()
