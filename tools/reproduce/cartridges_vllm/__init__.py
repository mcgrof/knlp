"""CartridgeConnector vLLM serving validation orchestrator.

Runs Tiers -1 through 6 of the CartridgeConnector validation plan.
Invoked from Makefile.cartridges as:
    python3 -m tools.reproduce.cartridges_vllm.run <subcommand> --config .config
"""
