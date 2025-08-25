// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/linear_classifier.hpp"

namespace ov {
namespace op {
namespace v13 {

LinearClassifier::LinearClassifier(const Output<Node>& input,
                                   const Output<Node>& weights,
                                   const Output<Node>& bias) {
    set_arguments({input, weights, bias});
    constructor_validate_and_infer_types();
}

void LinearClassifier::validate_and_infer_types() {
    // Basic type/shape inference, can be extended
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> LinearClassifier::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<LinearClassifier>(new_args.at(0), new_args.at(1), new_args.at(2));
}

#ifdef OPENVINO_ENABLE_EVALUATE
#include "openvino/reference/linear_classifier.hpp"
#include "openvino/core/type/element_type.hpp"
#include <openvino/core/tensor.hpp>

namespace {
bool evaluate_linear_classifier(const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    // Assumes: inputs = [input, weights, bias, class_labels]
    // outputs = [out_labels]
    const auto& input = inputs[0];
    const auto& weights = inputs[1];
    const auto& bias = inputs[2];
    const auto& class_labels = inputs[3];
    auto& out_labels = outputs[0];
    size_t batch_size = input.get_shape()[0];
    size_t input_dim = input.get_shape()[1];
    size_t num_classes = weights.get_shape()[0];
    ov::reference::linear_classifier_dispatch(
        input.data(), weights.data(), bias.data(), class_labels.data(), out_labels.data(),
        batch_size, input_dim, num_classes, input.get_element_type(), class_labels.get_element_type()
    );
    return true;
}
}

bool LinearClassifier::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate_linear_classifier(inputs, outputs);
}
#endif

} // namespace v13
} // namespace op
} // namespace ov
