# SPDX-License-Identifier: MIT

load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "gemma_savedmodel_runner",
    srcs = ["gemma_savedmodel_runner.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)
