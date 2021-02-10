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

def test_scale_update_monotony():
    update = np.ones((2,3))
    epochs = 10
    learn_rate = 0.02
    previous = 1e1000
    for epoch in range(epochs):
        result = chu.scale_update(update, epoch, epochs, learn_rate)
        assert np.greater(previous, result).all()
        previous = result

def test_product_slower():
    weight_matrix = np.array([[0,  1, -1],
                              [1, -1,  9]])
    batch = np.ones((27,3))
    p = 2
    result = chu.product(weight_matrix, batch, p)
    result2 = np.empty((27,2))

    for i in range(27):
        b_i = batch[i]
        for j in range(2):
            w_j = weight_matrix[j]
            result2[i,j] = np.sum(b_i * w_j * np.abs(w_j) ** (p-2))

    assert np.allclose(result, result2)

test_product_slower()
test_plasticity_rule_vectorized_null()
test_scale_update_monotony()
