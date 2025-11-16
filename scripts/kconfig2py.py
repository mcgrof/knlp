#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Convert Kconfig .config file to Python configuration module.
"""

import sys
import re
from pathlib import Path


def parse_config_file(config_file):
    """Parse .config file and extract configuration values."""
    config = {}

    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                # Handle "# CONFIG_XXX is not set" lines
                if line.startswith("# CONFIG_") and "is not set" in line:
                    match = re.match(r"# CONFIG_(\w+) is not set", line)
                    if match:
                        key = match.group(1)
                        config[key] = False
                continue

            # Parse CONFIG_KEY=value lines
            if "=" in line:
                key, value = line.split("=", 1)
                if key.startswith("CONFIG_"):
                    key = key[7:]  # Remove CONFIG_ prefix

                    # Strip inline comments (anything after # that's not in quotes)
                    # For quoted strings, find the closing quote first
                    if value.startswith('"'):
                        # Find the closing quote
                        end_quote = value.find('"', 1)
                        if end_quote != -1:
                            # Check for comment after the closing quote
                            if "#" in value[end_quote + 1 :]:
                                value = value[: end_quote + 1].strip()
                    elif "#" in value:
                        # For non-quoted values, strip everything after #
                        value = value.split("#")[0].strip()

                    # Process value based on type
                    if value == "y":
                        config[key] = True
                    elif value == "n":
                        config[key] = False
                    elif value.startswith('"') and value.endswith('"'):
                        config[key] = value[1:-1]  # Remove quotes
                    elif value.isdigit():
                        config[key] = int(value)
                    elif (
                        "." in value
                        and value.replace(".", "").replace("-", "").isdigit()
                    ):
                        try:
                            config[key] = float(value)
                        except ValueError:
                            config[key] = value
                    else:
                        config[key] = value

    return config


def generate_python_config(config):
    """Generate Python configuration module content."""
    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append("# SPDX-License-Identifier: MIT")
    lines.append("# Auto-generated configuration from Kconfig")
    lines.append("# DO NOT EDIT - Generated from .config")
    lines.append("")
    lines.append('"""')
    lines.append("AdamWPrune configuration generated from Kconfig.")
    lines.append('"""')
    lines.append("")

    # Add derived ADAMWPRUNE_BASE_OPTIMIZER_NAME based on base selection
    if config.get("ADAMWPRUNE_BASE_ADAM", False):
        config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"] = "adam"
    elif config.get("ADAMWPRUNE_BASE_ADAMW", False):
        config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"] = "adamw"
    else:
        config["ADAMWPRUNE_BASE_OPTIMIZER_NAME"] = "adamw"  # Default to AdamW

    # Group configurations by category
    categories = {
        "model": [],
        "optimizer": [],
        "pruning": [],
        "training": [],
        "data": [],
        "advanced": [],
        "debug": [],
        "other": [],
    }

    for key, value in sorted(config.items()):
        key_lower = key.lower()

        if "model" in key_lower or "lenet" in key_lower or "dataset" in key_lower:
            categories["model"].append((key, value))
        elif (
            "optimizer" in key_lower
            or "sgd" in key_lower
            or "adam" in key_lower
            or "spam" in key_lower
        ):
            categories["optimizer"].append((key, value))
        elif (
            "pruning" in key_lower
            or "sparsity" in key_lower
            or "movement" in key_lower
            or "magnitude" in key_lower
        ):
            categories["pruning"].append((key, value))
        elif any(
            x in key_lower for x in ["batch", "epoch", "learning", "worker", "device"]
        ):
            categories["training"].append((key, value))
        elif any(x in key_lower for x in ["data", "augment"]):
            categories["data"].append((key, value))
        elif any(
            x in key_lower
            for x in [
                "compile",
                "mixed",
                "gpu",
                "pin",
                "persistent",
                "gradient",
                "checkpoint",
            ]
        ):
            categories["advanced"].append((key, value))
        elif any(x in key_lower for x in ["debug", "verbose", "log", "profile"]):
            categories["debug"].append((key, value))
        else:
            categories["other"].append((key, value))

    # Generate class-based configuration
    lines.append("class Config:")
    lines.append('    """Configuration class with all settings."""')
    lines.append("    ")

    for category, items in categories.items():
        if items:
            lines.append(f"    # {category.capitalize()} configuration")
            for key, value in items:
                if isinstance(value, str):
                    lines.append(f'    {key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f"    {key} = {value}")
                elif isinstance(value, (int, float)):
                    lines.append(f"    {key} = {value}")
                else:
                    lines.append(f"    {key} = {repr(value)}")
            lines.append("    ")

    # Add get() method for dictionary-like interface
    lines.append("    def get(self, key, default=None):")
    lines.append('        """Get configuration value by key (dictionary-like interface)."""')
    lines.append("        return getattr(self, key, default)")
    lines.append("")

    # Add convenience properties
    lines.append("    @property")
    lines.append("    def is_pruning_enabled(self):")
    lines.append('        """Check if pruning is enabled."""')
    lines.append("        # Check both ENABLE_PRUNING and PRUNING_MODE_NONE")
    lines.append('        if getattr(self, "PRUNING_MODE_NONE", False):')
    lines.append("            return False")
    lines.append('        return getattr(self, "ENABLE_PRUNING", False)')
    lines.append("    ")
    lines.append("    @property")
    lines.append("    def model_name(self):")
    lines.append('        """Get the model name."""')
    lines.append('        return getattr(self, "MODEL", "lenet5")')
    lines.append("    ")
    lines.append("    @property")
    lines.append("    def optimizer_name(self):")
    lines.append('        """Get the optimizer name."""')
    lines.append('        return getattr(self, "OPTIMIZER", "sgd")')
    lines.append("    ")
    lines.append("    @property")
    lines.append("    def pruning_method(self):")
    lines.append('        """Get the pruning method."""')
    lines.append("        # Explicitly handle PRUNING_MODE_NONE")
    lines.append('        if getattr(self, "PRUNING_MODE_NONE", False):')
    lines.append('            return "none"')
    lines.append("        if not self.is_pruning_enabled:")
    lines.append('            return "none"')
    lines.append('        return getattr(self, "PRUNING_METHOD", "none")')
    lines.append("")

    # Create global instance
    lines.append("# Global configuration instance")
    lines.append("config = Config()")
    lines.append("")

    # Add helper functions
    lines.append("def get_training_args():")
    lines.append('    """Get training arguments as a dictionary."""')
    lines.append("    return {")
    lines.append('        "batch_size": config.BATCH_SIZE,')
    lines.append('        "num_epochs": config.NUM_EPOCHS,')
    lines.append('        "learning_rate": float(config.LEARNING_RATE),')
    lines.append('        "num_workers": config.NUM_WORKERS,')
    lines.append('        "device": config.DEVICE,')
    lines.append("    }")
    lines.append("")

    lines.append("def get_optimizer_args():")
    lines.append('    """Get optimizer-specific arguments."""')
    lines.append("    args = {")
    lines.append('        "optimizer": config.OPTIMIZER,')
    lines.append("    }")
    lines.append("    ")
    lines.append('    if config.OPTIMIZER == "sgd":')
    lines.append(
        '        args["momentum"] = float(getattr(config, "SGD_MOMENTUM", 0.9))'
    )
    lines.append('    elif config.OPTIMIZER in ["adamwspam", "adamwprune"]:')
    lines.append(
        '        args["spam_theta"] = float(getattr(config, "SPAM_THETA", 50.0))'
    )
    lines.append(
        '        args["spam_enable_clip"] = getattr(config, "SPAM_ENABLE_CLIP", False)'
    )
    lines.append('        args["spam_interval"] = getattr(config, "SPAM_INTERVAL", 0)')
    lines.append(
        '        args["spam_warmup_steps"] = getattr(config, "SPAM_WARMUP_STEPS", 0)'
    )
    lines.append("")
    lines.append("    # AdamWPrune tuning parameters")
    lines.append('    if config.OPTIMIZER == "adamwprune":')
    lines.append("        # Determine base optimizer name from config")
    lines.append('        if getattr(config, "ADAMWPRUNE_BASE_ADAM", False):')
    lines.append('            args["adamwprune_base_optimizer_name"] = "adam"')
    lines.append('        elif getattr(config, "ADAMWPRUNE_BASE_ADAMWADV", False):')
    lines.append('            args["adamwprune_base_optimizer_name"] = "adamwadv"')
    lines.append('        elif getattr(config, "ADAMWPRUNE_BASE_ADAMWSPAM", False):')
    lines.append('            args["adamwprune_base_optimizer_name"] = "adamwspam"')
    lines.append("        else:")
    lines.append(
        '            args["adamwprune_base_optimizer_name"] = "adamw"  # default'
    )
    lines.append("        ")
    lines.append(
        '        args["adamwprune_enable_pruning"] = getattr(config, "ADAMWPRUNE_ENABLE_PRUNING", True)'
    )
    lines.append(
        '        args["adamwprune_beta1"] = float(getattr(config, "ADAMWPRUNE_BETA1", "0.9"))'
    )
    lines.append(
        '        args["adamwprune_beta2"] = float(getattr(config, "ADAMWPRUNE_BETA2", "0.999"))'
    )
    lines.append(
        '        args["adamwprune_weight_decay"] = float(getattr(config, "ADAMWPRUNE_WEIGHT_DECAY", "0.01"))'
    )
    lines.append(
        '        args["adamwprune_amsgrad"] = getattr(config, "ADAMWPRUNE_AMSGRAD", True)'
    )
    lines.append("    ")
    lines.append("    return args")
    lines.append("")

    lines.append("def get_pruning_args():")
    lines.append('    """Get pruning-specific arguments."""')
    lines.append("    if not config.is_pruning_enabled:")
    lines.append('        return {"pruning_method": "none"}')
    lines.append("    ")
    lines.append("    return {")
    lines.append('        "pruning_method": config.pruning_method,')
    lines.append(
        '        "target_sparsity": float(getattr(config, "TARGET_SPARSITY", 0.9)),'
    )
    lines.append('        "pruning_warmup": getattr(config, "PRUNING_WARMUP", 100),')
    lines.append("    }")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python kconfig2py.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    if not Path(config_file).exists():
        print(f"Error: {config_file} not found")
        sys.exit(1)

    config = parse_config_file(config_file)
    python_config = generate_python_config(config)

    print(python_config)


if __name__ == "__main__":
    main()
