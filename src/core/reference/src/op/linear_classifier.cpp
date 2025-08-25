
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include "openvino/reference/softmax.hpp"
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
	// Logits
	std::vector<T> logits(num_classes, 0);
	for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
		size_t coeff_start = class_idx * input_dim;
		T dot_product = 0;
		for (size_t i = 0; i < input_dim; ++i) {
			dot_product += input[i] * coefficients[coeff_start + i];
		}
		logits[class_idx] = dot_product + intercepts[class_idx];
	}

	// Softmax
	ov::Shape shape{num_classes};
	ov::AxisSet axes{0};
	ov::reference::softmax<T>(logits.data(), probabilities, shape, axes);

	// Argmax
	size_t max_idx = std::distance(probabilities, std::max_element(probabilities, probabilities + num_classes));
	*out_label = class_labels[max_idx];
} // namespace reference
} // namespace ov
}
