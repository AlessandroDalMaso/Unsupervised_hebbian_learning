"""Test public functions of CHUNeuralNetwork.py."""
import CHUNeuralNetwork as chu
import numpy as np
import utilities as utils

def test_batchize():
    a = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])
    utils.batchize(a, 2)

def test_relu_positive():
    hidden_neurons = np.array([[0,  0,  0],
                               [2, -2,  3]])
    assert np.all(chu.relu(hidden_neurons) >= 0)



def test_plasticity_rule():
    weight_vector = np.array([0, 1, -2, 3])
    input_vector = np.array([0, 1, 0, 1])
    product_result = 2
    g = -0.4
    p = 3
    one_over_scale = 1
    R = 1

    result = chu.plasticity_rule(weight_vector, input_vector, product_result, g, p, R,
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
    k = 2
    one_over_scale = 1
    activation_function = chu.relu
    result = chu.plasticity_rule_vectorized(weight_matrix, batch, delta, p,
                                            R, k, one_over_scale,
                                            activation_function)
    batch2 = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    result2 = chu.plasticity_rule_vectorized(weight_matrix, batch2, delta, p,
                                             R, k, one_over_scale,
                                             activation_function)
    assert np.array_equal(result2, np.zeros(weight_matrix.shape))

test_batchize()
test_relu_positive()
test_plasticity_rule()
test_plasticity_rule_vectorized_null()
