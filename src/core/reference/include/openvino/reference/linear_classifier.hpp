// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include "openvino/reference/softmax.hpp"
#include "openvino/reference/matmul.hpp"
#include <limits>
#include <stdexcept>

#include "openvino/core/shape.hpp"
#include "openvino/core/axis_set.hpp"

namespace ov {
namespace reference {


template <typename T>
void linear_classifier(const T* input,
                      const T* coefficients,
                      const T* intercepts,
                      const int* class_labels,
                      T* probabilities,
                      int* out_label,
                      const size_t input_dim,
                      const size_t num_classes) {
    // Logits using matmul
    std::vector<T> logits(num_classes, 0);
    ov::Shape input_shape{1, input_dim};
    ov::Shape weights_shape{input_dim, num_classes};
    ov::Shape logits_shape{1, num_classes};

    ov::reference::matmul<T>(input, coefficients, logits.data(), input_shape, weights_shape, logits_shape, false, false);
    // Add intercepts
    for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        logits[class_idx] += intercepts[class_idx];
    }

    // Softmax
    ov::Shape shape{num_classes};
    ov::AxisSet axes{0};
    ov::reference::softmax<T>(logits.data(), probabilities, shape, axes);

    // Argmax
    size_t max_idx = std::distance(probabilities, std::max_element(probabilities, probabilities + num_classes));
    *out_label = class_labels[max_idx];
}

} // namespace reference
} // namespace ov
