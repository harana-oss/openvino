# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import numpy as np
from openvino.runtime import Core, Tensor

MODELS = [
    ("normalizer_max.prototxt", "MAX"),
    ("normalizer_l1.prototxt", "L1"),
    ("normalizer_l2.prototxt", "L2"),
    ("normalizer_1d.prototxt", "L2"),
]

def load_model(path):
    with open(path, "rb") as f:
        return onnx.load(f)


def _reference(norm: str, X: np.ndarray):
    original_shape = X.shape
    # Ensure we have at least 2D for consistent axis operations
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    if norm == "MAX":
        div = np.max(np.abs(X), axis=-1, keepdims=True)
    elif norm == "L1":
        div = np.sum(np.abs(X), axis=-1, keepdims=True)
    elif norm == "L2":
        div = np.sqrt(np.sum(X * X, axis=-1, keepdims=True))
    else:
        raise RuntimeError
    # if divisor zero -> identity
    mask = div == 0
    Y = np.divide(X, div, where=~mask, out=X.copy())
    
    # Return in the same shape as ONNX model would (keep batch dimension for 1D inputs)
    if len(original_shape) == 1:
        return Y  # Keep as (1, N) to match ONNX output
    return Y


def test_normalizer_models():
    core = Core()
    base = os.path.dirname(__file__) + "/../models/"
    # Custom deterministic inputs per model
    inputs = {
        "normalizer_max.prototxt": np.array([[0.0, 2.0, -1.0], [4.0, 0.0, -2.0]], dtype=np.float32),
        "normalizer_l1.prototxt": np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.0]], dtype=np.float32),
        "normalizer_l2.prototxt": np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float32),
        "normalizer_1d.prototxt": np.array([3.0, 4.0, 0.0], dtype=np.float32),
    }
    for fname, norm in MODELS:
        model = load_model(base + fname)
        compiled = core.compile_model(core.read_model(model.SerializeToString()), "CPU")
        X = inputs[fname]
        got = compiled(Tensor(X))[0]
        ref = _reference(norm, X)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, rtol=1e-3, atol=1e-4)
def test_normalizer_zero_div():
    core = Core()
    base = os.path.dirname(__file__) + "/../models/"
    # Create zero vector to trigger zero divisor path
    zero_inputs = {
        "normalizer_max.prototxt": np.zeros((2, 3), dtype=np.float32),
        "normalizer_l1.prototxt": np.zeros((2, 3), dtype=np.float32),
        "normalizer_l2.prototxt": np.zeros((2, 3), dtype=np.float32),
        "normalizer_1d.prototxt": np.zeros(3, dtype=np.float32),
    }
    for fname, norm in MODELS:
        model = load_model(base + fname)
        compiled = core.compile_model(core.read_model(model.SerializeToString()), "CPU")
        X = zero_inputs[fname]
        got = compiled(Tensor(X))[0]
        # Expect identity - reshape got to match input shape for comparison
        got_reshaped = got.reshape(X.shape) if got.shape != X.shape else got
        assert np.array_equal(got_reshaped, X)