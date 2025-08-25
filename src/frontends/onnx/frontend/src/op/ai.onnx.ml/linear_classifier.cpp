// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "onnx_framework_node.hpp"

#include <chrono>
#include "openvino/util/log.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {

// Ensure numeric input: prefer f32 for compute. Cast integers and f64 to f32, keep f32 as-is.
ov::Output<ov::Node> ensure_float(const ov::Output<ov::Node>& inp) {
    const auto et = inp.get_element_type();
    if (et.is_real()) {
        if (et == ov::element::f32) return inp;
        // Cast other real types (e.g., f64) to f32 for better kernel performance
        return std::make_shared<ov::op::v0::Convert>(inp, ov::element::f32);
    }
    // integer -> convert to f32
    return std::make_shared<ov::op::v0::Convert>(inp, ov::element::f32);
}

ov::OutputVector linear_classifier_impl(const ov::frontend::onnx::Node& node) {
    using clock = std::chrono::steady_clock;
    // auto t_total_begin = clock::now();

    auto X = node.get_ov_inputs().at(0);
    X = ensure_float(X);
    // Profiling code removed

    // If rank 1 -> reshape to [1,C]
    {
        if (X.get_partial_shape().rank().is_static() && X.get_partial_shape().rank().get_length() == 1) {
            // Reshape 1D input [C] to [1, C]; use -1 to infer feature dimension even if dynamic
            auto shape_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, -1});
            X = std::make_shared<ov::op::v1::Reshape>(X, shape_c, false);
        }
    }

    // attributes
    auto coefficients = node.get_attribute_value<std::vector<float>>("coefficients", {});
    CHECK_VALID_NODE(node, !coefficients.empty(), "LinearClassifier: coefficients attribute required");
    auto intercepts = node.get_attribute_value<std::vector<float>>("intercepts", {});

    bool has_int_labels = node.has_attribute("classlabels_ints");
    bool has_str_labels = node.has_attribute("classlabels_strings");
    CHECK_VALID_NODE(node, has_int_labels ^ has_str_labels, "LinearClassifier: exactly one of classlabels_ints or classlabels_strings must be present");
    std::vector<int64_t> int_labels;
    std::vector<std::string> str_labels;
    if (has_int_labels)
        int_labels = node.get_attribute_value<std::vector<int64_t>>("classlabels_ints", {});
    else
        str_labels = node.get_attribute_value<std::vector<std::string>>("classlabels_strings", {});
    size_t n_classes = has_int_labels ? int_labels.size() : str_labels.size();

    // Determine number of classifier weight vectors (k)
    size_t k = 0; // number of linear functions before binary expansion
    if (!intercepts.empty()) {
        k = intercepts.size();
    } else if (n_classes > 0) {
        // In binary case we may have k=1
        if (n_classes == 2) {
            // Heuristic: if coefficients.size() % 2 != 0 -> treat k=1 else k=n_classes
            if (coefficients.size() % 2 != 0) k = 1; else k = n_classes;
        } else {
            k = n_classes;
        }
    }
    CHECK_VALID_NODE(node, k > 0, "LinearClassifier: cannot deduce number of classifiers");
    CHECK_VALID_NODE(node, coefficients.size() % k == 0, "LinearClassifier: coefficients size incompatible with intercepts/classes");
    size_t n_features = coefficients.size() / k;

    // Build weight constant shaped [k, n_features], then transpose -> [n_features, k]
    // Keep weights in f32 to match X after ensure_float
    auto coeff_const_typed = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{k, n_features}, coefficients);
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1,0});
    auto coeff_T = std::make_shared<ov::op::v1::Transpose>(coeff_const_typed, order); // [n_features, k]

    // MatMul: X [N,C] * W [C,k] -> [N,k]
    std::shared_ptr<ov::Node> scores = std::make_shared<ov::op::v0::MatMul>(X, coeff_T);

    // Add intercepts if provided
    if (!intercepts.empty()) {
        // Bias in f32 to match X after ensure_float
        std::shared_ptr<ov::Node> bias = ov::op::v0::Constant::create(ov::element::f32,
                                                                      ov::Shape{static_cast<size_t>(intercepts.size())},
                                                                      intercepts);
        if (intercepts.size() != k) {
            // Special binary case: intercepts.size()==1 && k may be 1 or 2; only add if consistent
            CHECK_VALID_NODE(node, intercepts.size() == 1 && k >= 1, "LinearClassifier: unexpected intercepts size");
        }
        auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto bias_row = std::make_shared<ov::op::v0::Unsqueeze>(bias, axis0); // [1,k]
        scores = std::make_shared<ov::op::v1::Add>(scores, bias_row);
    }

    // post_transform
    std::string post_transform = node.get_attribute_value<std::string>("post_transform", "NONE");
    std::shared_ptr<ov::Node> transformed = scores;
    if (post_transform == "LOGISTIC") {
        transformed = std::make_shared<ov::op::v0::Sigmoid>(scores);
    } else if (post_transform == "SOFTMAX" || post_transform == "SOFTMAX_ZERO") { // approximate SOFTMAX_ZERO with softmax
        transformed = std::make_shared<ov::op::v8::Softmax>(scores, 1);
    } else if (post_transform == "PROBIT") {
        // Not implemented: identity fallback
        transformed = scores;
    } else { // NONE or unknown
        transformed = scores;
    }

    // ArgMax using TopK
    auto k_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto topk = std::make_shared<ov::op::v1::TopK>(transformed, k_const, 1, ov::op::v1::TopK::Mode::MAX, ov::op::v1::TopK::SortType::NONE, ov::element::i64);
    auto argmax = topk->output(1); // indices

    std::shared_ptr<ov::Node> labels_output;
    if (has_int_labels) {
        auto labels_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{n_classes}, int_labels);
        labels_output = std::make_shared<ov::op::v8::Gather>(labels_const, argmax, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    } else {
        auto labels_const = ov::op::v0::Constant::create(ov::element::string, ov::Shape{n_classes}, str_labels);
        labels_output = std::make_shared<ov::op::v8::Gather>(labels_const, argmax, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    }


    return {labels_output, transformed};
}

} // namespace

namespace opset_1 {
ov::OutputVector linear_classifier(const ov::frontend::onnx::Node& node) { return ::ov::frontend::onnx::ai_onnx::linear_classifier_impl(node); }
ONNX_OP("LinearClassifier", OPSET_SINCE(1), ai_onnx::opset_1::linear_classifier, AU_ONNX_ML_DOMAIN);
} // namespace opset_1

} // namespace ai_onnx
} // namespace onnx
} // namespace frontend
} // namespace ov
