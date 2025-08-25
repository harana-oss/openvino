// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {

class OPENVINO_API LinearClassifier : public Op {
public:
    OPENVINO_OP("LinearClassifier", "opset13");

    LinearClassifier() = default;
    LinearClassifier(const Output<Node>& input,
                    const Output<Node>& weights,
                    const Output<Node>& bias);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override { return true; }
};

} // namespace v13
} // namespace op
} // namespace ov
