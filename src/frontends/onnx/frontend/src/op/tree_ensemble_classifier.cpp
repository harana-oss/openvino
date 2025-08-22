// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "onnx_framework_node.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/convert_like.hpp"

#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {

struct NodeInfo {
    int64_t treeid;
    int64_t nodeid;
    std::string mode; // BRANCH_* or LEAF
    int64_t featureid;
    float threshold;
    int64_t truenodeid;
    int64_t falsenodeid;
    int64_t missing_tracks_true; // 0/1
};

using NodeKey = std::pair<int64_t,int64_t>;
struct NodeKeyHash {
    size_t operator()(const NodeKey& k) const noexcept {
        return (std::hash<int64_t>()(k.first) * 1315423911u) ^ std::hash<int64_t>()(k.second);
    }
};

ov::Output<ov::Node> ensure_float(const ov::Output<ov::Node>& inp) {
    if (inp.get_element_type().is_real())
        return inp;
    return std::make_shared<ov::op::v0::Convert>(inp, ov::element::f32);
}

ov::Output<ov::Node> build_feature_extract(const ov::Output<ov::Node>& X, int64_t feature_id, std::unordered_map<int64_t, ov::Output<ov::Node>>& cache) {
    auto it = cache.find(feature_id);
    if (it != cache.end())
        return it->second;
    // gather along axis 1 index feature_id -> shape [N,1]
    auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {feature_id});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gathered = std::make_shared<ov::op::v8::Gather>(X, indices, axis);
    // squeeze last dim -> [N]
    auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto squeezed = std::make_shared<ov::op::v0::Squeeze>(gathered, squeeze_axis);
    cache.emplace(feature_id, squeezed);
    return squeezed;
}

ov::Output<ov::Node> build_node_condition(const NodeInfo& ni,
                                          const ov::Output<ov::Node>& X,
                                          std::unordered_map<NodeKey, ov::Output<ov::Node>, NodeKeyHash>& cond_cache,
                                          std::unordered_map<int64_t, ov::Output<ov::Node>>& feat_cache) {
    NodeKey key{ni.treeid, ni.nodeid};
    auto it = cond_cache.find(key);
    if (it != cond_cache.end())
        return it->second;
    // LEAF has no condition; should not be queried
    auto feature_vec = build_feature_extract(X, ni.featureid, feat_cache); // shape [N]
    feature_vec = ensure_float(feature_vec);
    auto thresh = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {ni.threshold});
    auto thresh_b = std::make_shared<ov::op::v1::Reshape>(thresh, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);

    std::shared_ptr<ov::Node> cmp;
    if (ni.mode == "BRANCH_LEQ") cmp = std::make_shared<ov::op::v1::LessEqual>(feature_vec, thresh_b);
    else if (ni.mode == "BRANCH_LT") cmp = std::make_shared<ov::op::v1::Less>(feature_vec, thresh_b);
    else if (ni.mode == "BRANCH_GTE") cmp = std::make_shared<ov::op::v1::GreaterEqual>(feature_vec, thresh_b);
    else if (ni.mode == "BRANCH_GT") cmp = std::make_shared<ov::op::v1::Greater>(feature_vec, thresh_b);
    else if (ni.mode == "BRANCH_EQ") cmp = std::make_shared<ov::op::v1::Equal>(feature_vec, thresh_b);
    else if (ni.mode == "BRANCH_NEQ") cmp = std::make_shared<ov::op::v1::NotEqual>(feature_vec, thresh_b);
    else FRONT_END_GENERAL_CHECK(false, "Unexpected node mode for non-leaf: " + ni.mode);

    // missing value handling
    auto isnan = std::make_shared<ov::op::v10::IsNaN>(feature_vec);
    auto true_const = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});
    auto false_const = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {false});
    std::shared_ptr<ov::Node> cond = std::make_shared<ov::op::v1::Select>(isnan,
                                                                          ni.missing_tracks_true ? true_const : false_const,
                                                                          cmp);
    cond_cache.emplace(key, cond);
    return cond;
}

// Build path predicate for leaf (treeid,nodeid)
ov::Output<ov::Node> build_leaf_path(int64_t treeid,
                                     int64_t leaf_nodeid,
                                     const std::unordered_map<NodeKey, NodeInfo, NodeKeyHash>& node_map,
                                     const std::unordered_map<int64_t, int64_t>& root_nodeid_for_tree,
                                     const ov::Output<ov::Node>& X,
                                     std::unordered_map<NodeKey, ov::Output<ov::Node>, NodeKeyHash>& cond_cache,
                                     std::unordered_map<int64_t, ov::Output<ov::Node>>& feat_cache) {
    struct Edge { NodeInfo parent; bool took_true; };
    std::vector<Edge> path_edges;
    int64_t current = leaf_nodeid;
    int64_t root = root_nodeid_for_tree.at(treeid);
    std::unordered_map<int64_t, NodeInfo> parent_by_child; // within the tree
    for (auto& kv : node_map) {
        if (kv.first.first != treeid) continue;
        const auto& ni = kv.second;
        if (ni.mode != "LEAF") {
            parent_by_child[ni.truenodeid] = ni;
            parent_by_child[ni.falsenodeid] = ni;
        }
    }
    while (current != root) {
        auto pit = parent_by_child.find(current);
        FRONT_END_GENERAL_CHECK(pit != parent_by_child.end(), "Cannot find parent for node id " + std::to_string(current));
        const auto parent = pit->second;
        bool took_true = (parent.truenodeid == current);
        path_edges.push_back({parent, took_true});
        current = parent.nodeid;
    }
    auto true_c = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});
    ov::Output<ov::Node> path = true_c;
    for (auto it = path_edges.rbegin(); it != path_edges.rend(); ++it) {
        auto cond = build_node_condition(it->parent, X, cond_cache, feat_cache); // [N]
        ov::Output<ov::Node> edge_ok = it->took_true ? cond : std::make_shared<ov::op::v1::LogicalNot>(cond);
        path = std::make_shared<ov::op::v1::LogicalAnd>(path, edge_ok);
    }
    return path; // boolean [N]
}

ov::OutputVector tree_ensemble_classifier_impl(const ov::frontend::onnx::Node& node) {
    auto X = node.get_ov_inputs().at(0);
    if (X.get_element_type() != ov::element::f32 && X.get_element_type() != ov::element::f16 && !X.get_element_type().is_real()) {
        X = std::make_shared<ov::op::v0::Convert>(X, ov::element::f32);
    } else if (X.get_element_type() == ov::element::f64) {
        X = std::make_shared<ov::op::v0::Convert>(X, ov::element::f32);
    }

    auto nodes_treeids = node.get_attribute_value<std::vector<int64_t>>("nodes_treeids", {});
    auto nodes_nodeids = node.get_attribute_value<std::vector<int64_t>>("nodes_nodeids", {});
    auto nodes_featureids = node.get_attribute_value<std::vector<int64_t>>("nodes_featureids", {});
    auto nodes_modes = node.get_attribute_value<std::vector<std::string>>("nodes_modes", {});
    auto nodes_truenodeids = node.get_attribute_value<std::vector<int64_t>>("nodes_truenodeids", {});
    auto nodes_falsenodeids = node.get_attribute_value<std::vector<int64_t>>("nodes_falsenodeids", {});

    std::vector<float> nodes_values;
    if (node.has_attribute("nodes_values_as_tensor")) {
        nodes_values = node.get_attribute_value<std::vector<float>>("nodes_values_as_tensor", {});
    } else {
        auto raw = node.get_attribute_value<std::vector<float>>("nodes_values", {});
        nodes_values.assign(raw.begin(), raw.end());
    }
    auto nodes_missing_value_tracks_true = node.get_attribute_value<std::vector<int64_t>>("nodes_missing_value_tracks_true", {});
    if (nodes_missing_value_tracks_true.empty()) {
        nodes_missing_value_tracks_true.resize(nodes_treeids.size(), 0);
    }

    size_t n_nodes = nodes_treeids.size();
    CHECK_VALID_NODE(node, n_nodes == nodes_nodeids.size() && n_nodes == nodes_featureids.size() && n_nodes == nodes_modes.size() && n_nodes == nodes_truenodeids.size() && n_nodes == nodes_falsenodeids.size() && n_nodes == nodes_values.size(), "TreeEnsembleClassifier: nodes_* attribute size mismatch");

    // Class weights
    std::vector<float> class_weights;
    if (node.has_attribute("class_weights_as_tensor"))
        class_weights = node.get_attribute_value<std::vector<float>>("class_weights_as_tensor", {});
    else
        class_weights = node.get_attribute_value<std::vector<float>>("class_weights", {});

    auto class_treeids = node.get_attribute_value<std::vector<int64_t>>("class_treeids", {});
    auto class_nodeids = node.get_attribute_value<std::vector<int64_t>>("class_nodeids", {});
    auto class_ids = node.get_attribute_value<std::vector<int64_t>>("class_ids", {});
    CHECK_VALID_NODE(node, class_treeids.size() == class_nodeids.size() && class_ids.size() == class_nodeids.size() && class_weights.size() == class_nodeids.size(), "TreeEnsembleClassifier: class_* attribute size mismatch");

    // Class labels
    std::vector<int64_t> classlabels_int64s;
    std::vector<std::string> classlabels_strings;
    bool has_int_labels = node.has_attribute("classlabels_int64s");
    bool has_str_labels = node.has_attribute("classlabels_strings");
    CHECK_VALID_NODE(node, has_int_labels ^ has_str_labels, "TreeEnsembleClassifier: exactly one of classlabels_int64s or classlabels_strings must be present");
    if (has_int_labels)
        classlabels_int64s = node.get_attribute_value<std::vector<int64_t>>("classlabels_int64s", {});
    else
        classlabels_strings = node.get_attribute_value<std::vector<std::string>>("classlabels_strings", {});

    size_t n_classes = has_int_labels ? classlabels_int64s.size() : classlabels_strings.size();
    if (n_classes == 0) {
        n_classes = *std::max_element(class_ids.begin(), class_ids.end()) + 1;
    }

    // base_values
    std::vector<float> base_values;
    if (node.has_attribute("base_values_as_tensor"))
        base_values = node.get_attribute_value<std::vector<float>>("base_values_as_tensor", {});
    else
        base_values = node.get_attribute_value<std::vector<float>>("base_values", {});
    if (!base_values.empty())
        CHECK_VALID_NODE(node, base_values.size() == n_classes, "TreeEnsembleClassifier: base_values size must equal number of classes");

    // Build node map and identify roots per tree
    std::unordered_map<NodeKey, NodeInfo, NodeKeyHash> node_map;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> children_per_tree;
    for (size_t i = 0; i < n_nodes; ++i) {
        NodeInfo info{nodes_treeids[i], nodes_nodeids[i], nodes_modes[i], nodes_featureids[i], nodes_values[i], nodes_truenodeids[i], nodes_falsenodeids[i], nodes_missing_value_tracks_true[i]};
        node_map[{info.treeid, info.nodeid}] = info;
        if (info.mode != "LEAF") {
            children_per_tree[info.treeid].insert(info.truenodeid);
            children_per_tree[info.treeid].insert(info.falsenodeid);
        }
    }
    std::unordered_map<int64_t, int64_t> root_nodeid_for_tree; // treeid -> root nodeid
    std::unordered_map<int64_t, std::unordered_set<int64_t>> nodes_in_tree;
    for (size_t i = 0; i < n_nodes; ++i) {
        nodes_in_tree[nodes_treeids[i]].insert(nodes_nodeids[i]);
    }
    for (auto& kv : nodes_in_tree) {
        int64_t treeid = kv.first;
        int64_t root_candidate = -1;
        for (auto nid : kv.second) {
            if (!children_per_tree[treeid].count(nid)) { // node not referenced as child
                root_candidate = nid;
                break;
            }
        }
        CHECK_VALID_NODE(node, root_candidate != -1, "TreeEnsembleClassifier: cannot determine root for tree " + std::to_string(treeid));
        root_nodeid_for_tree[treeid] = root_candidate;
    }

    // Collect leaves
    struct Leaf { int64_t treeid; int64_t nodeid; };
    std::vector<Leaf> leaves;
    for (auto& kv : node_map) {
        if (kv.second.mode == "LEAF") leaves.push_back({kv.second.treeid, kv.second.nodeid});
    }

    // Precompute per-leaf class weight vectors
    std::unordered_map<NodeKey, std::vector<float>, NodeKeyHash> leaf_class_weights;
    for (size_t i = 0; i < class_ids.size(); ++i) {
        NodeKey key{class_treeids[i], class_nodeids[i]};
        auto& vec = leaf_class_weights[key];
        if (vec.empty()) vec.assign(n_classes, 0.f);
        int64_t cid = class_ids[i];
        if (static_cast<size_t>(cid) >= n_classes) continue;
        vec[cid] += class_weights[i];
    }

    std::shared_ptr<ov::Node> scores;
    std::unordered_map<NodeKey, ov::Output<ov::Node>, NodeKeyHash> cond_cache;
    std::unordered_map<int64_t, ov::Output<ov::Node>> feat_cache;

    for (const auto& leaf : leaves) {
        NodeKey key{leaf.treeid, leaf.nodeid};
        auto itw = leaf_class_weights.find(key);
        if (itw == leaf_class_weights.end()) continue;
        auto path_bool = build_leaf_path(leaf.treeid, leaf.nodeid, node_map, root_nodeid_for_tree, X, cond_cache, feat_cache);
        auto path_float = std::make_shared<ov::op::v0::Convert>(path_bool, ov::element::f32);
        auto axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto path_unsq = std::make_shared<ov::op::v0::Unsqueeze>(path_float, axis1);
        auto& wvec = itw->second;
        auto wconst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, wvec.size()}, wvec);
        auto contrib = std::make_shared<ov::op::v1::Multiply>(path_unsq, wconst);
        if (!scores)
            scores = contrib;
        else
            scores = std::make_shared<ov::op::v1::Add>(scores, contrib);
    }
    if (!scores) {
        scores = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, n_classes}, std::vector<float>(n_classes, 0.f));
    }

    if (!base_values.empty()) {
        auto base_c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, base_values.size()}, base_values);
        scores = std::make_shared<ov::op::v1::Add>(scores, base_c);
    }

    std::string post_transform = node.get_attribute_value<std::string>("post_transform", "NONE");
    std::shared_ptr<ov::Node> transformed = scores;
    if (post_transform == "LOGISTIC") {
        transformed = std::make_shared<ov::op::v0::Sigmoid>(scores);
    } else if (post_transform == "SOFTMAX") {
        transformed = std::make_shared<ov::op::v8::Softmax>(scores, 1);
    } else if (post_transform == "SOFTMAX_ZERO") {
        transformed = std::make_shared<ov::op::v8::Softmax>(scores, 1);
    } else if (post_transform == "PROBIT") {
        transformed = scores;
    }

    auto k_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto topk = std::make_shared<ov::op::v1::TopK>(transformed, k_const, 1, ov::op::v1::TopK::Mode::MAX, ov::op::v1::TopK::SortType::NONE, ov::element::i64);
    auto argmax = topk->output(1);
    std::shared_ptr<ov::Node> labels_output;
    if (has_int_labels) {
        auto labels_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(n_classes)}, classlabels_int64s);
        labels_output = std::make_shared<ov::op::v8::Gather>(labels_const, argmax, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    } else {
        auto labels_const = ov::op::v0::Constant::create(ov::element::string, ov::Shape{static_cast<size_t>(n_classes)}, classlabels_strings);
        labels_output = std::make_shared<ov::op::v8::Gather>(labels_const, argmax, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    }

    return {labels_output, transformed};
}

} // namespace

namespace opset_1 {
ov::OutputVector tree_ensemble_classifier(const ov::frontend::onnx::Node& node) { return ::ov::frontend::onnx::ai_onnx::tree_ensemble_classifier_impl(node); }
ONNX_OP("TreeEnsembleClassifier", OPSET_SINCE(1), ai_onnx::opset_1::tree_ensemble_classifier);
} // namespace opset_1
namespace opset_3 {
ov::OutputVector tree_ensemble_classifier(const ov::frontend::onnx::Node& node) { return ::ov::frontend::onnx::ai_onnx::tree_ensemble_classifier_impl(node); }
ONNX_OP("TreeEnsembleClassifier", OPSET_SINCE(3), ai_onnx::opset_3::tree_ensemble_classifier);
} // namespace opset_3

} // namespace ai_onnx
} // namespace onnx
} // namespace frontend
} // namespace ov
