# SPDX-License-Identifier: MIT
PYTHON ?= python3
ROOT ?= $(HOME)/supervised-depth-runs
DATA ?= $(HOME)/.cache/knlp/mnist
CHECKPOINTS ?= $(HOME)/supervised-depth-checkpoints
REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
ARGS = --root $(ROOT) --data $(DATA) --checkpoints $(CHECKPOINTS)
.PHONY: check probe qualify plan run-core run-channels analyze
check:
	cd $(REPO) && $(PYTHON) -m pytest -q research/supervised_depth/tests
probe qualify plan run-core run-channels analyze:
	cd $(REPO) && $(PYTHON) -m research.supervised_depth.study $@ $(ARGS)
