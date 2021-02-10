"""Test public functions of CHUNeuralNetwork.py."""
import CHUNeuralNetwork as chu
import numpy as np
import utilities as utils



def test_plasticity_rule_vectorized_null():
    batch = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    weight_matrix = np.array([[0,  1, -1],
                              [1, -1,  9]])
    delta = 0.4
    p = 3
    R = 1
    k = 2
    hh = 1
    one_over_scale = 1
    result = chu.plasticity_rule_vectorized(weight_matrix=weight_matrix, batch=batch, delta=delta, p=p,
                                            R=R, k=k, hh=1, one_over_scale=one_over_scale)

    assert np.array_equal(result, np.zeros(weight_matrix.shape))


test_plasticity_rule_vectorized_null()
