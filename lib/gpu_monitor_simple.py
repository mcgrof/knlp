#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simplified GPU monitoring for training integration.
Focuses on JSON data collection without interactive display.
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class SimpleGPUMonitor:
    """Simple GPU monitor focused on data collection."""

    def __init__(self, stats_file: str):
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

        self.running = False
        self.monitor_thread = None
        self.data = []
        self.start_time = None

        # Try to detect GPU type
        self.gpu_type = self._detect_gpu_type()

    def _detect_gpu_type(self) -> str:
        """Detect GPU type (AMD or NVIDIA)."""
        try:
            # Try NVIDIA first
            subprocess.run(
                ["nvidia-smi", "--version"], capture_output=True, check=True, timeout=5
            )
            return "nvidia"
        except:
            try:
                # Try AMD
                subprocess.run(
                    ["rocm-smi", "--version"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                return "amd"
            except:
                return "unknown"

    def _get_nvidia_stats(self) -> Optional[Dict[str, Any]]:
        """Get NVIDIA GPU stats using nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None

            # Handle multi-GPU output - take first line only (GPU 0)
            lines = result.stdout.strip().split("\n")
            if not lines:
                return None
            line = lines[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 6:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "gpu_name": parts[0],
                    "gpu_type": "nvidia",
                    "utilization": float(parts[1] or 0),
                    "memory_used": int(parts[2] or 0),
                    "memory_total": int(parts[3] or 0),
                    "memory_percent": (int(parts[2] or 0) / int(parts[3] or 1)) * 100,
                    "power": float(parts[4] or 0),
                    "temperature": float(parts[5] or 0),
                    "fan_speed": 0,  # Not available via nvidia-smi
                }
        except Exception:
            pass
        return None

    def _get_amd_stats(self) -> Optional[Dict[str, Any]]:
        """Get AMD GPU stats using rocm-smi."""
        try:
            # Get utilization and memory
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            gpu_data = data.get("card0", {})

            # Get temperature and power
            temp_result = subprocess.run(
                ["rocm-smi", "--showtemp", "--showpower", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if temp_result.returncode == 0:
                temp_data = json.loads(temp_result.stdout)
                gpu_temp_data = temp_data.get("card0", {})
                gpu_data.update(gpu_temp_data)

            # Extract values - note the data is NOT nested, it's direct
            try:
                utilization = int(gpu_data.get("GPU use (%)", 0))
            except (ValueError, TypeError):
                utilization = 0

            try:
                memory_used = int(gpu_data.get("VRAM Total Used Memory (B)", 0)) // (
                    1024**2
                )
                memory_total = int(gpu_data.get("VRAM Total Memory (B)", 0)) // (
                    1024**2
                )
            except (ValueError, TypeError):
                memory_used = 0
                memory_total = 49152  # W7900 has 48GB

            # Power and temp might not be available
            try:
                power = float(gpu_data.get("Average Graphics Package Power (W)", 0))
            except (ValueError, TypeError, KeyError):
                power = 0

            try:
                temp = float(gpu_data.get("Temperature (Sensor edge) (C)", 0))
            except (ValueError, TypeError, KeyError):
                temp = 0

            return {
                "timestamp": datetime.now().isoformat(),
                "gpu_name": "AMD GPU",
                "gpu_type": "amd",
                "utilization": float(utilization or 0),
                "memory_used": int(memory_used or 0),
                "memory_total": int(memory_total or 0),
                "memory_percent": (
                    int(memory_used or 0) / max(int(memory_total or 1), 1)
                )
                * 100,
                "power": float(power or 0),
                "temperature": float(temp or 0),
                "fan_speed": 0,  # Would need separate command
            }
        except Exception:
            pass
        return None

    def _get_stats(self) -> Optional[Dict[str, Any]]:
        """Get GPU stats based on detected type."""
        if self.gpu_type == "nvidia":
            return self._get_nvidia_stats()
        elif self.gpu_type == "amd":
            return self._get_amd_stats()
        return None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                stats = self._get_stats()
                if stats:
                    # Add elapsed time
                    if self.start_time:
                        stats["elapsed_seconds"] = time.time() - self.start_time

                    self.data.append(stats)

                    # Save periodically (every 5 samples for crash resilience)
                    if len(self.data) % 5 == 0:
                        self._save_data()

                time.sleep(1)  # Sample every second
            except Exception as e:
                # Silently continue on errors
                pass

    def _save_data(self):
        """Save collected data to JSON file."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass  # Silently fail

    def start(self):
        """Start monitoring."""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring and save final data."""
        if not self.running:
            return

        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Final save
        self._save_data()

        # Generate summary stats
        self._generate_summary()

    def _generate_summary(self):
        """Generate summary statistics."""
        if not self.data:
            return

        try:
            summary_file = str(self.stats_file).replace(".json", "_summary.txt")

            # Calculate stats
            utilizations = [d["utilization"] for d in self.data if "utilization" in d]
            memory_percents = [
                d["memory_percent"] for d in self.data if "memory_percent" in d
            ]
            powers = [d["power"] for d in self.data if "power" in d and d["power"] > 0]
            temps = [
                d["temperature"]
                for d in self.data
                if "temperature" in d and d["temperature"] > 0
            ]

            with open(summary_file, "w") as f:
                f.write("GPU Monitoring Summary\n")
                f.write("=" * 40 + "\n\n")

                if self.data:
                    f.write(f"GPU: {self.data[0].get('gpu_name', 'Unknown')}\n")
                    f.write(f"Type: {self.data[0].get('gpu_type', 'unknown')}\n")
                    f.write(f"Duration: {len(self.data)} seconds\n")
                    f.write(f"Start: {self.data[0]['timestamp']}\n")
                    f.write(f"End: {self.data[-1]['timestamp']}\n\n")

                if utilizations:
                    f.write(f"GPU Utilization:\n")
                    f.write(f"  Average: {sum(utilizations)/len(utilizations):.1f}%\n")
                    f.write(f"  Maximum: {max(utilizations):.1f}%\n")
                    f.write(f"  Minimum: {min(utilizations):.1f}%\n\n")

                if memory_percents:
                    f.write(f"Memory Usage:\n")
                    f.write(
                        f"  Average: {sum(memory_percents)/len(memory_percents):.1f}%\n"
                    )
                    f.write(f"  Maximum: {max(memory_percents):.1f}%\n")
                    f.write(f"  Minimum: {min(memory_percents):.1f}%\n\n")

                if powers:
                    f.write(f"Power Consumption:\n")
                    f.write(f"  Average: {sum(powers)/len(powers):.1f}W\n")
                    f.write(f"  Maximum: {max(powers):.1f}W\n")
                    f.write(f"  Minimum: {min(powers):.1f}W\n\n")

                if temps:
                    f.write(f"Temperature:\n")
                    f.write(f"  Average: {sum(temps)/len(temps):.1f}°C\n")
                    f.write(f"  Maximum: {max(temps):.1f}°C\n")
                    f.write(f"  Minimum: {min(temps):.1f}°C\n")

        except Exception:
            pass  # Silently fail

    def get_stats_file(self):
        """Get the path to the stats file."""
        return self.stats_file


def create_simple_monitor(stats_file: str) -> SimpleGPUMonitor:
    """Create a simple GPU monitor."""
    return SimpleGPUMonitor(stats_file)


# Context manager for easy use
class SimpleGPUMonitorContext:
    """Context manager for simple GPU monitoring."""

    def __init__(self, stats_file: str, metadata: Dict[str, Any] = None):
        self.monitor = create_simple_monitor(stats_file)
        self.metadata = metadata or {}
        self.stats_file = Path(stats_file)

    def __enter__(self) -> SimpleGPUMonitor:
        self.monitor.start()
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()

        # Add metadata to the JSON file
        if self.metadata and self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)

                # Add metadata to first entry
                if data and len(data) > 0:
                    data[0]["training_metadata"] = self.metadata

                    with open(self.stats_file, "w") as f:
                        json.dump(data, f, indent=2)
            except Exception:
                pass  # Silently fail


if __name__ == "__main__":
    # Test the simple monitor
    print("Testing simple GPU monitoring...")

    with SimpleGPUMonitorContext("test_stats.json", {"test": True}) as monitor:
        print("Monitoring for 10 seconds...")
        time.sleep(10)

    print("Monitoring completed - check test_stats.json")
