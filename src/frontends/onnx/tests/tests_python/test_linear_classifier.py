# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import os
from openvino.runtime import Core, Tensor

MODELS = [
    ("linear_classifier_binary_single_plane.prototxt", [1], None),
    ("linear_classifier_binary_single_plane_no_intercept.prototxt", [0], None),  # no intercept
    ("linear_classifier_binary_logits.prototxt", [10], None),
    ("linear_classifier_multiclass_raw.prototxt", [2], None),  # intercepts pick class label 2
    ("linear_classifier_multiclass_no_intercepts.prototxt", [1], None),  # tie -> first class label 1
    ("linear_classifier_multiclass_ovr.prototxt", ["C"], None),  # intercepts + softmax pick label C
]

def load_model(path):
    with open(path, "rb") as f:
        return onnx.load(f)

def test_linear_classifier_models():
    core = Core()
    base = os.path.dirname(__file__) + "/../models/"
    for fname, expected_labels, _ in MODELS:
        model_path = base + fname
        model = load_model(model_path)
        # Serialize ONNX ModelProto to bytes for read_model
        model_bytes = model.SerializeToString()
        compiled = core.compile_model(core.read_model(model_bytes), "CPU")
        outputs = compiled()
        labels = outputs[0]
        scores = outputs[1]
        assert labels.shape[0] == len(expected_labels)
        # Basic shape checks for scores second dim equals number of classes
        if "_multiclass" in fname or "multiclass" in fname:
            assert scores.shape[1] == 3
        else:
            assert scores.shape[1] == 2
        for i, exp in enumerate(expected_labels):
            assert labels[i] == exp
        # Probability sum check for softmax model
        if "ovr" in fname:
            row_sum = float(np.sum(scores[0]))
            assert np.isclose(row_sum, 1.0, atol=1e-5)

def _compute_expected(model: onnx.ModelProto):
    node = model.graph.node[0]
    attrs = {a.name: a for a in node.attribute}
    coeffs = np.array(attrs["coefficients"].floats, dtype=np.float32)
    intercepts = np.array(attrs["intercepts"].floats, dtype=np.float32) if "intercepts" in attrs else np.array([], dtype=np.float32)
    int_labels = np.array(attrs["classlabels_ints"].ints, dtype=np.int64) if "classlabels_ints" in attrs else None
    str_labels = [s.decode() if isinstance(s, bytes) else s for s in attrs["classlabels_strings"].strings] if "classlabels_strings" in attrs else None
    post_transform = attrs["post_transform"].s.decode() if "post_transform" in attrs else "NONE"

    n_classes = len(int_labels) if int_labels is not None else len(str_labels)
    # Heuristic copied from implementation
    if intercepts.size > 0:
        k = intercepts.size
    elif n_classes > 0:
        if n_classes == 2:
            k = 1 if (coeffs.size % 2 != 0) else n_classes
        else:
            k = n_classes
    else:
        raise RuntimeError("Cannot deduce k")
    assert coeffs.size % k == 0
    n_features = coeffs.size // k
    W = coeffs.reshape(k, n_features)  # [k, F]

    # Prepare single zero input of feature size
    X = np.zeros((1, n_features), dtype=np.float32)

    scores = X @ W.T  # [1,k]
    if intercepts.size > 0:
        if intercepts.size == 1 and k > 1:
            # broadcast single intercept across? Implementation only allows size==1 with k>=1 but adds unsqueezed -> shape [1,1]; if k=2 (binary expansion) intercept added before expansion.
            # Here if k==1 that's the binary single-plane case handled later.
            pass
        scores = scores + intercepts.reshape(1, -1)

    binary_two_class = (k == 1 and n_classes == 2)
    if binary_two_class:
        s = scores[:, 0]
        scores = np.stack([-s, s], axis=1)
        k = 2

    transformed = scores.copy()
    if post_transform == "LOGISTIC":
        transformed = 1.0 / (1.0 + np.exp(-scores))
    elif post_transform in ("SOFTMAX", "SOFTMAX_ZERO"):
        # stable softmax
        m = transformed.max(axis=1, keepdims=True)
        e = np.exp(transformed - m)
        transformed = e / e.sum(axis=1, keepdims=True)
    # PROBIT / NONE -> identity

    # Argmax (TopK) semantics: first occurrence on ties -> np.argmax matches
    argmax = np.argmax(transformed, axis=1)
    if int_labels is not None:
        labels = int_labels[argmax]
    else:
        labels = np.array([str_labels[i] for i in argmax], dtype=object)
    labels = labels.reshape(-1, 1)  # match runtime gather output shape [N,1]
    return labels, transformed


def test_linear_classifier_end_to_end():
    core = Core()
    base = os.path.dirname(__file__) + "/../models/"
    for fname, _, _ in MODELS:
        model_path = base + fname
        model = load_model(model_path)
        expected_labels, expected_scores = _compute_expected(model)
        compiled = core.compile_model(core.read_model(model.SerializeToString()), "CPU")
        outputs = compiled()
        got_labels = outputs[0]
        got_scores = outputs[1]
        # Align shapes (runtime uses [N,1])
        assert got_labels.shape == expected_labels.shape
        # Compare labels element-wise
        for i in range(expected_labels.shape[0]):
            assert got_labels[i, 0] == expected_labels[i, 0]
        # Compare scores
        assert got_scores.shape == expected_scores.shape
        expected_scores = expected_scores.astype(got_scores.dtype, copy=False)
        # Relaxed tolerance: fused backend softmax may differ ~2e-4 from numpy reference
        assert np.allclose(got_scores, expected_scores, rtol=5e-4, atol=3e-4)
