# Top-level Makefile for AdamWPrune experiments
# SPDX-License-Identifier: MIT

# Default target MUST be first - declare it before any includes
.DEFAULT_GOAL := all

# Include Kconfig support to get CONFIG variables
include Makefile.kconfig

# TIME= shorthand for GPT2_MAX_TIME (e.g., make TIME=60)
ifdef TIME
export GPT2_MAX_TIME := $(TIME)
endif

# BASELINE= for referencing previous baseline run (e.g., make BASELINE=entity/project/run_id)
ifdef BASELINE
export CONFIG_BASELINE_RUN_ID := $(BASELINE)
endif

# Define what the default target does based on configuration
.PHONY: all
ifeq ($(CONFIG_OPTIMIZER_MODE_MULTIPLE),y)
all: check-config test-matrix
else ifeq ($(CONFIG_MODEL_MODE_MULTIPLE),y)
all: check-config test-matrix
else
all: check-config train
endif

# Default model to train (can be overridden by Kconfig)
MODEL ?= $(if $(CONFIG_MODEL),$(CONFIG_MODEL),lenet5)

# Run memory comparison experiments using test matrix
memory-comparison: check-config
	@echo "Running memory comparison experiments..."
	@echo "Use 'make test-matrix' for full optimizer/pruning comparisons"
	@echo "Or configure specific tests with 'make menuconfig' then 'make'"

# Update graphs with latest results
update-graphs: check-config generate-config
	@echo "Updating graphs from test matrix results..."
	@if [ -n "$(CONFIG_TEST_RESULTS_DIR)" ]; then \
		RESULTS_DIR="$(CONFIG_TEST_RESULTS_DIR)"; \
	else \
		RESULTS_DIR=$$(ls -d test_matrix_results_* 2>/dev/null | sort | tail -1); \
		if [ -z "$$RESULTS_DIR" ]; then \
			echo "Error: No test_matrix_results_* directories found"; \
			echo "Run 'make test-matrix' first or set TEST_RESULTS_DIR in menuconfig"; \
			exit 1; \
		fi; \
	fi; \
	echo "Using results from: $$RESULTS_DIR"; \
	python3 scripts/generate_optimizer_graphs.py "$$RESULTS_DIR" "$$RESULTS_DIR/graphs"; \
	python3 scripts/generate_gpu_memory_comparison.py "$$RESULTS_DIR" --output "$$RESULTS_DIR/graphs"; \
	python3 scripts/visualize_train_vs_inference_memory.py "$$RESULTS_DIR" --output "$$RESULTS_DIR/graphs"; \
	python3 scripts/generate_research_visualizations.py "$$RESULTS_DIR"; \
	\
	# Detect which model was tested by checking test directory names \
	if ls "$$RESULTS_DIR" | grep -q "^lenet5_"; then \
		MODEL_DIR="lenet5"; \
	elif ls "$$RESULTS_DIR" | grep -q "^resnet18_"; then \
		MODEL_DIR="resnet18"; \
	elif ls "$$RESULTS_DIR" | grep -q "^resnet50_"; then \
		MODEL_DIR="resnet50"; \
	else \
		echo "Warning: Could not detect model type, defaulting to lenet5"; \
		MODEL_DIR="lenet5"; \
	fi; \
	\
	mkdir -p "images/$$MODEL_DIR"; \
	echo "Copying graphs to images/$$MODEL_DIR/..."; \
	for optimizer in sgd adam adamw adamwadv adamwspam adamwprune; do \
		if [ -f "$$RESULTS_DIR/graphs/$${optimizer}_model_comparison.png" ]; then \
			cp "$$RESULTS_DIR/graphs/$${optimizer}_model_comparison.png" "images/$$MODEL_DIR/$${optimizer}_model_comparison.png"; \
			cp "$$RESULTS_DIR/graphs/$${optimizer}_accuracy_evolution.png" "images/$$MODEL_DIR/$${optimizer}_accuracy_evolution.png"; \
			echo "  Copied $$optimizer graphs"; \
		fi; \
	done; \
	if [ -f "$$RESULTS_DIR/graphs/gpu_memory_comparison.png" ]; then \
		cp "$$RESULTS_DIR/graphs/gpu_memory_comparison.png" "images/$$MODEL_DIR/gpu_memory_comparison.png"; \
		echo "  Copied GPU memory comparison graph"; \
	fi; \
	if [ -f "$$RESULTS_DIR/graphs/training_memory_comparison.png" ]; then \
		cp "$$RESULTS_DIR/graphs/training_memory_comparison.png" "images/$$MODEL_DIR/training_memory_comparison.png"; \
		echo "  Copied training memory comparison graph"; \
	fi; \
	if [ -f "$$RESULTS_DIR/graphs/gpu_memory_timeline.png" ]; then \
		cp "$$RESULTS_DIR/graphs/gpu_memory_timeline.png" "images/$$MODEL_DIR/gpu_memory_timeline.png"; \
		echo "  Copied GPU memory timeline graph"; \
	fi; \
	if [ -f "$$RESULTS_DIR/graphs/memory_vs_accuracy_scatter.png" ]; then \
		cp "$$RESULTS_DIR/graphs/memory_vs_accuracy_scatter.png" "images/$$MODEL_DIR/memory_vs_accuracy_scatter.png"; \
		echo "  Copied memory vs accuracy scatter graph"; \
	fi; \
	echo "Graphs updated in images/$$MODEL_DIR/"

# Clean build artifacts but keep configuration and test results
clean:
	@echo "Cleaning build artifacts (keeping configuration and test results)..."
	@rm -f *.log train.log */train.log
	@rm -rf __pycache__
	@rm -rf */__pycache__
	@rm -rf lib/__pycache__
	@rm -rf scripts/__pycache__
	@rm -f *.pyc *.pyo
	@rm -f training_metrics.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@if [ -d $(MODEL) ]; then $(MAKE) -C $(MODEL) clean; fi

# Analyze GPU memory usage from battle results
analyze-gpu: check-config
	@echo "Analyzing GPU memory usage from test matrix results..."
	@LATEST_DIR=$$(ls -d test_matrix_results_* 2>/dev/null | sort | tail -1); \
	if [ -n "$$LATEST_DIR" ]; then \
		echo "Using results from: $$LATEST_DIR"; \
		python scripts/visualize_test_matrix_gpu.py "$$LATEST_DIR"; \
		echo "GPU analysis complete. View $$LATEST_DIR/gpu_comparison.png"; \
	else \
		echo "No test_matrix_results_* directories found."; \
		echo "Run 'make' with CONFIG_GPU_MONITOR=y to generate GPU data."; \
	fi

# Clean everything including configuration and model files
# Returns workspace to pristine distribution state (keeps downloaded datasets)
mrproper: clean
	@echo "Removing all generated files, configuration, and model files..."
	@rm -f .config .config.old config.py
	@rm -f include/generated/autoconf.h
	@rm -rf include/config include/generated
	@rm -f *.pth
	@rm -f *.json
	@rm -f test-matrix.yaml
	@rm -f *.patch
	@rm -rf results/ */results/
	@rm -f *.tmp *.swp *~ */*~
	@rm -f *_backup.py */*_backup.py
	@if [ -d $(MODEL) ]; then $(MAKE) -C $(MODEL) mrproper 2>/dev/null || true; fi
	@echo "Workspace cleaned to pristine state (datasets preserved)."

# Clean downloaded datasets (use with caution - requires re-downloading)
data-clean:
	@echo "Removing downloaded datasets..."
	@rm -rf data/
	@rm -rf */data/
	@echo "Datasets removed. They will be re-downloaded on next training run."


# Dataset preparation targets
.PHONY: prepare-datasets prepare-gpt2-datasets

# Prepare datasets based on configuration
prepare-datasets: generate-config
	@if [ "$(CONFIG_MODEL_GPT2)" = "y" ]; then \
		$(MAKE) prepare-gpt2-datasets; \
	fi

# Prepare GPT2 datasets
prepare-gpt2-datasets:
	@echo "Preparing GPT2 datasets..."
	@if [ "$(CONFIG_GPT2_DATASET_SHAKESPEARE)" = "y" ]; then \
		echo "Downloading Shakespeare dataset..."; \
		python3 gpt2/prepare_data.py --dataset shakespeare; \
	elif [ "$(CONFIG_GPT2_DATASET_FINEWEBEDU)" = "y" ]; then \
		echo "Downloading FineWebEdu dataset (this may take a while)..."; \
		python3 gpt2/prepare_data.py --dataset finewebedu; \
	elif [ "$(CONFIG_GPT2_DATASET_OPENWEBTEXT)" = "y" ]; then \
		echo "Downloading OpenWebText dataset (this may take a while)..."; \
		python3 gpt2/prepare_data.py --dataset openwebtext; \
	fi

# Validate architecture with dry-run mode (RATIO ablation)
# Quick check to catch configuration/architecture errors before GPU training
.PHONY: check
check: FORCE
	@echo "============================================================"
	@echo "Running dry-run architecture validation"
	@echo "============================================================"
	@START_TIME=$$(date +%s); \
	$(MAKE) defconfig-gpt2-ratio-ablation DRY_RUN=1 > /dev/null 2>&1 || exit 1; \
	./scripts/validate_ablation_steps.sh; \
	RESULT=$$?; \
	END_TIME=$$(date +%s); \
	DURATION=$$((END_TIME - START_TIME)); \
	echo ""; \
	echo "============================================================"; \
	if [ $$RESULT -eq 0 ]; then \
		echo "✓ Architecture validation completed in $${DURATION}s"; \
		echo "  All 19 RATIO ablation steps validated successfully"; \
		echo "  Ready for GPU training"; \
	else \
		echo "✗ Architecture validation failed after $${DURATION}s"; \
		echo "  Fix errors before committing GPU resources"; \
		exit 1; \
	fi; \
	echo "============================================================"

# Train with current configuration (using test matrix framework for consistency)
# Automatically detects and uses multiple GPUs with DDP when available
train: check-config generate-config prepare-datasets
	@echo "Training with configuration from .config..."
	@echo "Using test matrix framework with automatic multi-GPU support"
	@if [ -n "$(MAX_ITERS)" ]; then \
		echo "Setting MAX_ITERS to $(MAX_ITERS) for GPT-2 testing..."; \
	fi
	@if [ -n "$(EPOCHS)" ]; then \
		echo "Overriding epochs to $(EPOCHS) for testing..."; \
		GPT2_MAX_ITERS=$(MAX_ITERS) YES=$(YES) python3 scripts/run_test_matrix.py --config .config --override-epochs $(EPOCHS); \
	else \
		GPT2_MAX_ITERS=$(MAX_ITERS) YES=$(YES) python3 scripts/run_test_matrix.py --config .config; \
	fi

# RA+MLA training targets
.PHONY: train-ra-mla train-ra-mla-baseline train-ra-mla-full train-ra-mla-ablation

# Direct RA+MLA training (bypasses test matrix framework)
train-ra-mla: check-config generate-config prepare-gpt2-datasets
	@echo "Training GPT-2 with RA+MLA..."
	@if [ "$(CONFIG_ENABLE_RA_MLA)" != "y" ]; then \
		echo "Error: RA+MLA not enabled in current configuration"; \
		echo "Load an RA+MLA defconfig first:"; \
		echo "  make defconfig-gpt2-ra-mla-baseline"; \
		echo "  make defconfig-gpt2-ra-mla-full"; \
		echo "  make defconfig-gpt2-ra-mla-ablation"; \
		exit 1; \
	fi
	@# Build tracker argument from config
	@TRACKER_ARGS=""; \
	if [ "$(CONFIG_ENABLE_TRACKIO)" = "y" ]; then \
		TRACKER_ARGS="trackio"; \
	fi; \
	if [ "$(CONFIG_ENABLE_WANDB)" = "y" ]; then \
		if [ -n "$$TRACKER_ARGS" ]; then \
			TRACKER_ARGS="$$TRACKER_ARGS,wandb"; \
		else \
			TRACKER_ARGS="wandb"; \
		fi; \
	fi; \
	if [ -z "$$TRACKER_ARGS" ]; then \
		TRACKER_ARGS="none"; \
	fi; \
	# Run training with config-derived parameters \
	python3 gpt2/train_ra_mla.py \
		--model-name "$(CONFIG_GPT2_MODEL_NAME)" \
		--dataset "$(CONFIG_GPT2_DATASET_NAME)" \
		--batch-size $(CONFIG_BATCH_SIZE) \
		--learning-rate $(CONFIG_LEARNING_RATE) \
		--latent-dim $(CONFIG_RA_MLA_LATENT_DIM) \
		--ra-window $(CONFIG_RA_MLA_RA_WINDOW) \
		--ra-alpha $(CONFIG_RA_MLA_RA_ALPHA) \
		--tracker "$$TRACKER_ARGS" \
		--tracker-project "$(CONFIG_TRACKER_PROJECT)" \
		--max-iters $(if $(MAX_ITERS),$(MAX_ITERS),10000) \
		$(if $(EXTRA_ARGS),$(EXTRA_ARGS),)

# Quick shortcuts for RA+MLA experiments
train-ra-mla-baseline:
	@$(MAKE) defconfig-gpt2-ra-mla-baseline
	@$(MAKE) train-ra-mla MAX_ITERS=$(if $(MAX_ITERS),$(MAX_ITERS),5000)

train-ra-mla-full:
	@$(MAKE) defconfig-gpt2-ra-mla-full
	@$(MAKE) train-ra-mla MAX_ITERS=$(if $(MAX_ITERS),$(MAX_ITERS),5000)

train-ra-mla-ablation:
	@$(MAKE) defconfig-gpt2-ra-mla-ablation
	@echo "Run ablation with: make train-ra-mla EXTRA_ARGS='--latent-dim 64 --ra-alpha 0.0'"
	@echo "Or manually: python gpt2/train_ra_mla.py --latent-dim 64 --ra-alpha 0.5 --tracker trackio,wandb"

# Test matrix targets
test-matrix: check-config prepare-datasets
	@echo "Running test matrix with configuration from .config..."
	@if [ -n "$(MAX_ITERS)" ]; then \
		echo "Setting MAX_ITERS to $(MAX_ITERS) for GPT-2 testing..."; \
	fi
	@if [ -n "$(EPOCHS)" ]; then \
		echo "Overriding epochs to $(EPOCHS) for testing..."; \
		GPT2_MAX_ITERS=$(MAX_ITERS) YES=$(YES) python3 scripts/run_test_matrix.py --config .config --override-epochs $(EPOCHS); \
	else \
		GPT2_MAX_ITERS=$(MAX_ITERS) YES=$(YES) python3 scripts/run_test_matrix.py --config .config; \
	fi
	@$(MAKE) summary

# Estimate completion time for running test matrix
estimate:
	@echo "Estimating completion time for running test matrix..."
	@python3 scripts/estimate_completion.py

# Continue an interrupted test matrix run
continue:
	@# Find the latest test_matrix* directory
	@LATEST_DIR=$$(ls -d test_matrix_results_* 2>/dev/null | sort | tail -1); \
	if [ -z "$$LATEST_DIR" ]; then \
		echo "Error: No test_matrix_results_* directories found"; \
		echo "Run 'make test-matrix' first to start a test matrix"; \
		exit 1; \
	fi; \
	echo "Found latest test matrix directory: $$LATEST_DIR"; \
	echo "Checking for incomplete runs..."; \
	python3 scripts/run_test_matrix.py --continue-dir "$$LATEST_DIR"; \
	$(MAKE) summary

test-matrix-yaml:
	@echo "Running test matrix with YAML configuration..."
	@YES=$(YES) python3 scripts/run_test_matrix.py --config-yaml test-matrix.yaml

test-matrix-dry-run: check-config
	@echo "Test matrix dry run (shows what would be executed)..."
	@python3 scripts/run_test_matrix.py --config .config --dry-run

# Re-run specific tests in an existing results directory
# Usage: make test-rerun TARGET=test_matrix_results_20250826_181029 [OPTIMIZER=adamwprune]
test-rerun:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET directory must be specified"; \
		echo "Usage: make test-rerun TARGET=test_matrix_results_YYYYMMDD_HHMMSS [OPTIMIZER=adamwprune]"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@if [ ! -d "$(TARGET)" ]; then \
		echo "Error: Directory '$(TARGET)' does not exist"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@echo "Re-running tests in: $(TARGET)"
	@if [ -n "$(OPTIMIZER)" ]; then \
		echo "Filtering to optimizer: $(OPTIMIZER)"; \
		python3 scripts/run_test_matrix.py \
			--config .config \
			--rerun-dir $(TARGET) \
			--filter-optimizer $(OPTIMIZER); \
	else \
		python3 scripts/run_test_matrix.py \
			--config .config \
			--rerun-dir $(TARGET); \
	fi

# Parallel execution targets
parallel: check-config
	@echo "Running test matrix with parallel execution..."
	@scripts/run_parallel_test_matrix.sh

parallel-4: check-config
	@echo "Running test matrix with 4 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 4

parallel-8: check-config
	@echo "Running test matrix with 8 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 8

parallel-16: check-config
	@echo "Running test matrix with 16 parallel jobs..."
	@scripts/run_parallel_test_matrix.sh -j 16

# Hyperparameter sweep targets
.PHONY: sweep-generate sweep-run sweep sweep-clean

# Generate configurations for hyperparameter sweep
sweep-generate:
	@if [ -z "$(RANGE_CONFIG)" ]; then \
		echo "Usage: make sweep-generate RANGE_CONFIG=path/to/range-config"; \
		echo "Example: make sweep-generate RANGE_CONFIG=resnet18/defconfigs/resnet18-state-pruning-compare-range"; \
		exit 1; \
	fi
	@echo "Generating config combinations from $(RANGE_CONFIG)..."
	@python scripts/generate_config_combinations.py $(RANGE_CONFIG) sweep_configs
	@echo "Configurations generated in sweep_configs/"

# Run hyperparameter sweep with generated configs
sweep-run:
	@if [ ! -d "sweep_configs" ]; then \
		echo "Error: sweep_configs directory not found. Run 'make sweep-generate' first."; \
		exit 1; \
	fi
	@echo "Running hyperparameter sweep with configs from sweep_configs/..."
	@python scripts/run_test_matrix.py --config-dir sweep_configs

# Combined sweep: generate and run
sweep:
	@if [ -z "$(RANGE_CONFIG)" ]; then \
		echo "Usage: make sweep RANGE_CONFIG=path/to/range-config"; \
		echo "Example: make sweep RANGE_CONFIG=resnet18/defconfigs/resnet18-state-pruning-compare-range"; \
		exit 1; \
	fi
	@$(MAKE) sweep-generate RANGE_CONFIG=$(RANGE_CONFIG)
	@$(MAKE) sweep-run

# Clean sweep configs
sweep-clean:
	@echo "Cleaning sweep configurations..."
	@rm -rf sweep_configs

# Monitor sweep progress
sweep-monitor:
	@python scripts/monitor_sweep.py

# Watch sweep progress (continuous monitoring)
sweep-watch:
	@python scripts/monitor_sweep.py --watch

# Show sweep leaderboard
sweep-leaderboard:
	@python scripts/sweep_leaderboard.py

# Usage: make parallel-rerun TARGET=test_matrix_results_20250826_181029 [JOBS=8] [OPTIMIZER=adamwprune]
parallel-rerun:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET directory must be specified"; \
		echo "Usage: make parallel-rerun TARGET=test_matrix_results_YYYYMMDD_HHMMSS [JOBS=8] [OPTIMIZER=adamwprune]"; \
		echo ""; \
		echo "Available result directories:"; \
		ls -d test_matrix_results_* 2>/dev/null || echo "  (none found)"; \
		exit 1; \
	fi
	@JOBS=$${JOBS:-8}; \
	CMD_ARGS="-j $$JOBS -r $(TARGET)"; \
	if [ -n "$(OPTIMIZER)" ]; then \
		CMD_ARGS="$$CMD_ARGS -f $(OPTIMIZER)"; \
		echo "Re-running $(OPTIMIZER) tests in $(TARGET) with $$JOBS parallel jobs..."; \
	else \
		echo "Re-running all tests in $(TARGET) with $$JOBS parallel jobs..."; \
	fi; \
	scripts/run_parallel_test_matrix.sh $$CMD_ARGS

# Regenerate summary report from existing test results
# Usage: make summary [TEST_DIR=test_matrix_results_YYYYMMDD_HHMMSS]
summary:
	@if [ -f scripts/regenerate_summary_with_gpu.py ]; then \
		echo "Regenerating summary with real GPU memory data..."; \
		if [ -n "$(TEST_DIR)" ]; then \
			DIR="$(TEST_DIR)"; \
		else \
			DIR=$$(ls -d test_matrix_results_* 2>/dev/null | sort | tail -1); \
			if [ -z "$$DIR" ]; then \
				echo "Error: No test_matrix_results_* directories found."; \
				echo "Usage: make summary TEST_DIR=<test_results_dir>"; \
				exit 1; \
			fi; \
		fi; \
		echo "Using results from: $$DIR"; \
		python3 scripts/regenerate_summary_with_gpu.py "$$DIR"; \
	else \
		python3 scripts/regenerate_summary.py; \
	fi

# Defconfig targets - simple pattern rule for all defconfigs
# Override kconfig's defconfig handling
# Supports command-line overrides: make defconfig-name VAR=value
.PHONY: defconfig-%
defconfig-%: FORCE
	@# Check in main defconfigs directory first
	@if [ -f defconfigs/$* ]; then \
		echo "Loading defconfig: $*"; \
		cp defconfigs/$* .config; \
	elif [ -f gpt2/defconfigs/$* ]; then \
		echo "Loading GPT-2 defconfig: $*"; \
		cp gpt2/defconfigs/$* .config; \
	elif [ -f lenet5/defconfigs/$* ]; then \
		echo "Loading LeNet-5 defconfig: $*"; \
		cp lenet5/defconfigs/$* .config; \
	elif [ -f resnet18/defconfigs/$* ]; then \
		echo "Loading ResNet-18 defconfig: $*"; \
		cp resnet18/defconfigs/$* .config; \
	else \
		echo "Error: defconfig '$*' not found"; \
		echo ""; \
		$(MAKE) list-all-defconfigs; \
		exit 1; \
	fi; \
	\
	if [ -n "$(DRY_RUN)" ]; then \
		echo "  Enabling DRY_RUN mode..."; \
		echo "CONFIG_DRY_RUN=y" >> .config; \
	fi; \
	\
	python scripts/kconfig2py.py .config > config.py; \
	echo "Configuration loaded: $*"; \
	echo "Ready to run: make"

FORCE:

# Note: list-defconfigs is already defined in Makefile.kconfig
# Use 'make list-all-defconfigs' for detailed view or 'make list-defconfigs' for basic view

# Detailed listing of all defconfigs by category
.PHONY: list-all-defconfigs
list-all-defconfigs:
	@echo "Available defconfigs by category:"
	@echo "  Main configs:"
	@ls defconfigs/ 2>/dev/null | sed 's/^/    /' || echo "    (none)"
	@echo "  GPT-2 configs:"
	@ls gpt2/defconfigs/ 2>/dev/null | sed 's/^/    /' || echo "    (none)"
	@echo "  LeNet-5 configs:"
	@ls lenet5/defconfigs/ 2>/dev/null | sed 's/^/    /' || echo "    (none)"
	@echo "  ResNet-18 configs:"
	@ls resnet18/defconfigs/ 2>/dev/null | sed 's/^/    /' || echo "    (none)"

# Quick test matrix configurations
test-all-optimizers:
	@echo "Testing all optimizers with LeNet-5..."
	@cp defconfigs/test-matrix-optimizers .config
	@$(MAKE) test-matrix

test-all-pruning:
	@echo "Testing all pruning methods with LeNet-5..."
	@cp defconfigs/test-matrix-pruning .config
	@$(MAKE) test-matrix

test-everything:
	@echo "Testing all combinations (optimizers × pruning)..."
	@cp defconfigs/test-matrix-full .config
	@$(MAKE) test-matrix

# WandB integration test
wandb-test: check-config generate-config
	@echo "Running WandB integration test with fake data..."
	@python3 scripts/wandb_test.py

# Trackio integration test
trackio-test: check-config generate-config
	@echo "Running Trackio integration test with fake data..."
	@python3 scripts/trackio_test.py

# Launch TrackIO console dashboard (our new terminal UI)
trackio-view:
	@echo "Launching TrackIO Console Dashboard..."
	@if [ -z "$(PROJECT)" ]; then \
		PROJECT=$$(grep CONFIG_TRACKER_PROJECT .config 2>/dev/null | cut -d'"' -f2); \
		if [ -z "$$PROJECT" ]; then \
			echo "Auto-detecting project from training logs..."; \
			python3 scripts/trackio_console.py; \
		else \
			python3 scripts/trackio_console.py --project "$$PROJECT"; \
		fi; \
	else \
		python3 scripts/trackio_console.py --project "$(PROJECT)"; \
	fi

# Alternative name for console dashboard
trackio-console: trackio-view

# Launch TrackIO web dashboard server (original web UI)
trackio-web-server:
	@echo "Launching TrackIO web dashboard server..."
	@if [ -z "$(PROJECT)" ]; then \
		PROJECT=$$(grep CONFIG_TRACKER_PROJECT .config 2>/dev/null | cut -d'"' -f2); \
		if [ -z "$$PROJECT" ]; then \
			echo "No project found. Specify with: make trackio-web-server PROJECT=<project-name>"; \
			exit 1; \
		fi; \
	else \
		PROJECT=$(PROJECT); \
	fi; \
	@echo "Starting TrackIO server..."; \
	@echo "Dashboard will be available at the URL shown below."; \
	@echo "Press Ctrl+C to stop the server."; \
	@echo ""; \
	python3 -c "import trackio; trackio.show(project='$$PROJECT')"

# Show Trackio web URL (without launching server)
trackio-web:
	@echo "TrackIO Web Dashboard Info:"
	@if [ -z "$(PROJECT)" ]; then \
		PROJECT=$$(grep CONFIG_TRACKER_PROJECT .config 2>/dev/null | cut -d'"' -f2); \
		if [ -z "$$PROJECT" ]; then \
			echo "No project found. Specify with: make trackio-web PROJECT=<project-name>"; \
			exit 1; \
		fi; \
	else \
		PROJECT=$(PROJECT); \
	fi; \
	echo "URL: http://localhost:7860/?project=$$PROJECT"; \
	echo ""; \
	echo "To start the web server, run:"; \
	echo "  make trackio-view PROJECT=$$PROJECT"

# Help menu
help:
	@echo "AdamWPrune Experiments Makefile"
	@echo "================================"
	@echo ""
	@echo "Kconfig targets:"
	@echo "  menuconfig        - Configure using ncurses menu"
	@echo "  defconfig         - Load a default configuration (DEFCONFIG=name)"
	@echo "  defconfig-<tab>   - Tab-completable defconfig targets"
	@echo "  allyesconfig      - Enable all features (test matrix mode)"
	@echo "  allnoconfig       - Minimal configuration (SGD only)"
	@echo "  list-defconfigs   - List available default configurations"
	@echo "  savedefconfig     - Save current config as default"
	@echo "  kconfig-help      - Show all Kconfig targets"
	@echo ""
	@echo "Training targets:"
	@echo "  make              - Run experiments with current config (use this!)"
	@echo "  all               - Run memory comparison and update graphs (default)"
	@echo "  memory-comparison - Run all optimizer experiments with memory tracking"
	@echo "  update-graphs     - Update visualization graphs with latest results"
	@echo "  analyze-gpu       - Analyze GPU memory usage from battle results"
	@echo ""
	@echo "Note: Use plain 'make' to run experiments. The build system"
	@echo "automatically adapts based on configuration (single training vs"
	@echo "test matrix mode) and detects available GPUs for multi-GPU training."
	@echo "Supports NVIDIA, AMD, and other GPU vendors transparently."
	@echo ""
	@echo "RA+MLA (Reciprocal Attention) targets:"
	@echo "  train-ra-mla      - Train GPT-2 with RA+MLA (requires RA+MLA defconfig)"
	@echo "  train-ra-mla-baseline - Quick: Pure MLA (alpha=0.0, no reciprocal)"
	@echo "  train-ra-mla-full     - Quick: Full RA+MLA (alpha=0.5, recommended)"
	@echo "  train-ra-mla-ablation - Quick: Ablation template (manual parameter sweep)"
	@echo "  Example: make train-ra-mla MAX_ITERS=1000"
	@echo "  Example: make train-ra-mla EXTRA_ARGS='--latent-dim 32 --ra-alpha 0.5'"
	@echo ""
	@echo "Setup targets:"
	@echo "  deps              - Install Python dependencies from requirements.txt"
	@echo ""
	@echo "Cleaning targets:"
	@echo "  clean             - Clean build artifacts only (keeps config & datasets)"
	@echo "  mrproper          - Clean everything except datasets (removes config)"
	@echo "  data-clean        - Remove downloaded datasets (requires re-download)"
	@echo ""
	@echo "Test matrix targets:"
	@echo "  test-matrix       - Run test matrix from .config (serial)"
	@echo "  test-matrix-yaml  - Run test matrix from test-matrix.yaml"
	@echo "  test-matrix-dry-run - Show what would be tested without running"
	@echo "  test-rerun        - Re-run tests in existing directory"
	@echo "                      Usage: make test-rerun TARGET=<dir> [OPTIMIZER=<name>]"
	@echo "  summary           - Regenerate summary report from latest test results"
	@echo "  test-all-optimizers - Test all optimizers with LeNet-5"
	@echo "  test-all-pruning  - Test all pruning methods"
	@echo "  test-everything   - Test all combinations (optimizers × pruning)"
	@echo "  wandb-test        - Test WandB integration with fake training data"
	@echo "  trackio-test      - Test Trackio integration with fake training data"
	@echo "  trackio-view      - Launch TrackIO web dashboard server"
	@echo "  trackio-web       - Show TrackIO web URL info (doesn't launch server)"
	@echo ""
	@echo "Parallel execution targets (for high-memory GPUs):"
	@echo "  parallel          - Run test matrix with parallel jobs (default: 8 jobs)"
	@echo "  parallel-4        - Run with 4 parallel jobs"
	@echo "  parallel-8        - Run with 8 parallel jobs (recommended for 48GB GPU)"
	@echo "  parallel-16       - Run with 16 parallel jobs"
	@echo "  parallel-rerun    - Re-run tests with parallel execution"
	@echo "                      Usage: make parallel-rerun TARGET=<dir> [JOBS=8] [OPTIMIZER=<name>]"
	@echo ""
	@echo "Quick start:"
	@echo "  make defconfig-lenet5           # Load LeNet-5 full config (tab-completable)"
	@echo "  make defconfig-lenet5-sgd       # Load LeNet-5 with SGD"
	@echo "  make defconfig-lenet5-adamwprune # Load LeNet-5 with AdamWPrune"
	@echo "  make menuconfig                 # Customize configuration"
	@echo "  make                            # Run experiments with current config"
	@echo ""
	@echo "Test matrix mode:"
	@echo "  make allyesconfig               # Configure for all combinations"
	@echo "  make test-matrix                # Run all test combinations (serial)"
	@echo "  make parallel                   # Run all test combinations (8 parallel jobs)"
	@echo "  make allyesconfig parallel-16   # Configure and run with 16 parallel jobs"
	@echo ""
	@echo "Parallel execution (recommended for GPUs with >24GB memory):"
	@echo "  make parallel                   # 8 jobs (good for 48GB GPU like W7900)"
	@echo "  make parallel-16                # 16 jobs (for very high memory GPUs)"
	@echo ""
	@echo "Re-running specific tests:"
	@echo "  make test-rerun TARGET=test_matrix_results_20250826_181029"
	@echo "                                  # Re-run all tests in existing directory"
	@echo "  make parallel-rerun TARGET=test_matrix_results_20250826_181029 OPTIMIZER=adamwprune"
	@echo "                                  # Re-run only adamwprune tests with parallel execution"
	@echo ""
	@echo "Monitoring and continuation:"
	@echo "  make estimate                   # Estimate completion time for running tests"
	@echo "  make continue                   # Continue latest interrupted test matrix"
	@echo "                                  # Automatically finds latest test_matrix_results_*,"
	@echo "                                  # removes incomplete runs, and continues remaining tests"
	@echo ""

.PHONY: all memory-comparison update-graphs analyze-gpu clean mrproper data-clean help \
        train test-matrix test-matrix-yaml test-matrix-dry-run test-rerun summary \
        test-all-optimizers test-all-pruning test-everything deps \
        parallel parallel-4 parallel-8 parallel-16 parallel-rerun continue estimate

# Dependencies installation target
deps:
	@echo "Installing Python dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"
	@echo ""
	@echo "Note: You may also need to log in to tracking services:"
	@echo "  wandb login"
	@echo "  trackio login  # if required"
