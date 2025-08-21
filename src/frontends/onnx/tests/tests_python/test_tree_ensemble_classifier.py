# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import os
from openvino.runtime import Core, Tensor

MODELS = [
    ("tree_ensemble_classifier_single_tree.prototxt", [0,0,1,1], [[0.8807971,0.1192029],[0.8807971,0.1192029],[0.2689414,0.7310586],[0.2689414,0.7310586]]),
    ("tree_ensemble_classifier_missing_nan.prototxt", [10,10,20], None),
    ("tree_ensemble_classifier_string_labels.prototxt", ["cat","cat","dog","dog"], None),
]

def load_model(path):
    with open(path, "rb") as f:
        return onnx.load(f)

def test_tree_ensemble_classifier_models():
    core = Core()
    base = os.path.dirname(__file__) + "/../models/"
    for fname, expected_labels, expected_scores in MODELS:
        model_path = base + fname
        model = load_model(model_path)
        model_bytes = model.SerializeToString()
        compiled = core.compile_model(core.read_model(model_bytes), "CPU")
        outputs = compiled()
        labels = outputs[0]
        assert labels.shape[0] == len(expected_labels)
        for i, exp in enumerate(expected_labels):
            assert labels[i] == exp
        if expected_scores is not None:
            scores = outputs[1]
            assert scores.shape[0] == len(expected_scores)
            for i, row in enumerate(expected_scores):
                for j, val in enumerate(row):
                    assert np.isclose(scores[i,j], val, atol=1e-5)
