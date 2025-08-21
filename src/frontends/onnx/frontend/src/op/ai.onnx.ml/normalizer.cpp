// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "onnx_framework_node.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace frontend {
namespace onnx { 
namespace ai_onnx_ml {
namespace {

ov::Output<ov::Node> ensure_float(const ov::Output<ov::Node>& inp) {
    if (inp.get_element_type().is_real())
        return inp;
    return std::make_shared<ov::op::v0::Convert>(inp, ov::element::f32);
}

ov::OutputVector normalizer(const ov::frontend::onnx::Node& node) {
    auto X = node.get_ov_inputs().at(0);
    X = ensure_float(X);

    std::string norm = node.get_attribute_value<std::string>("norm", "MAX");

    // Determine the reduction axis: normalize along the last axis (C dimension)
    // For 1D input [C], axis=-1 (axis 0)
    // For 2D input [N,C], axis=-1 (axis 1) 
    // For higher dim [N,H,W,C], axis=-1 (last axis)
    auto rank = X.get_partial_shape().rank();
    int64_t reduction_axis = -1; // Always normalize along last axis
    
    std::shared_ptr<ov::Node> div; // divisor with same shape as X except reduced axis has size 1
    if (norm == "MAX") {
        // max(abs(X), axis=-1, keepdims=True)
        auto abs = std::make_shared<ov::op::v0::Abs>(X);
        div = std::make_shared<ov::op::v1::ReduceMax>(abs, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis}), true);
    } else if (norm == "L1") {
        auto abs = std::make_shared<ov::op::v0::Abs>(X);
        div = std::make_shared<ov::op::v1::ReduceSum>(abs, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis}), true);
    } else if (norm == "L2") {
        // sqrt(sum(x^2, axis=-1, keepdims=True))
        auto pow2 = std::make_shared<ov::op::v1::Power>(X, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.f}));
        auto sumsq = std::make_shared<ov::op::v1::ReduceSum>(pow2, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {reduction_axis}), true);
        div = std::make_shared<ov::op::v0::Sqrt>(sumsq);
    } else {
        FRONT_END_GENERAL_CHECK(false, "Normalizer: unexpected norm attribute: " + norm);
    }

    // If divisor zero -> return X. Implement by computing mask and conditional select.
    // Avoid performing an actual division by zero by selecting a safe denominator first.
    auto zero = ov::op::v0::Constant::create(div->get_element_type(), ov::Shape{}, {0.0f});
    auto is_zero = std::make_shared<ov::op::v1::Equal>(div, zero);
    auto one = ov::op::v0::Constant::create(div->get_element_type(), ov::Shape{}, {1.0f});
    auto safe_div = std::make_shared<ov::op::v1::Select>(is_zero, one, div);
    auto quot = std::make_shared<ov::op::v1::Divide>(X, safe_div);
    auto Y = std::make_shared<ov::op::v1::Select>(is_zero, X, quot);

    // Ensure output shape matches input shape (important for 1D inputs)
    // The ONNX spec requires that output shape equals input shape
    auto input_shape = X.get_partial_shape();
    if (input_shape.rank().is_static() && input_shape.is_static()) {
        // For static shapes, explicitly reshape if needed
        auto Y_shape = Y->get_partial_shape();
        if (Y_shape != input_shape) {
            auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{input_shape.rank().get_length()}, input_shape.to_shape());
            Y = std::make_shared<ov::op::v1::Reshape>(Y, target_shape, false);
        }
    } else if (input_shape.rank().is_static()) {
        // For dynamic shapes, use the shape of X directly
        auto shape_of_X = std::make_shared<ov::op::v3::ShapeOf>(X);
        Y = std::make_shared<ov::op::v1::Reshape>(Y, shape_of_X, false);
    }

    return {Y};
}

} // namespace

namespace opset_1 {
ov::OutputVector normalizer(const ov::frontend::onnx::Node& node) { return ::ov::frontend::onnx::ai_onnx_ml::normalizer(node); }
ONNX_OP("Normalizer", OPSET_SINCE(1), ai_onnx_ml::opset_1::normalizer, "ai.onnx.ml");
} // namespace opset_1

} // namespace ai_onnx_ml
} // namespace onnx
} // namespace frontend
} // namespace ov
