"""Runtime / disk estimates per profile and detected hardware."""
from __future__ import annotations
from .decode_config import DecodeConfig
from .hardware import HostInfo


def hardware_class(host: HostInfo) -> str:
    if host.gpu_count == 0:
        return "no_gpu"
    name = (host.gpu_names[0] if host.gpu_names else "").upper()
    if "H100" in name or "H200" in name:
        return "h100" if host.gpu_count == 1 else f"h100_x{host.gpu_count}"
    if "B200" in name:
        return "b200"
    if "A100" in name:
        return "a100"
    if "L40S" in name:
        return "l40s"
    if "MI300X" in name:
        return "mi300x"
    if "W7900" in name:
        return "w7900"
    return "other"


def estimate(cfg: DecodeConfig, host: HostInfo) -> dict:
    profile = cfg.profile
    cls = hardware_class(host)

    table = {
        "decode": {
            "h100":      {"cold_h": (8, 14),  "warm_h": (4, 8),    "disk_gb": (600, 1000)},
            "h100_x8":   {"cold_h": (3, 6),   "warm_h": (1.5, 3),  "disk_gb": (700, 1200)},
            "a100":      {"cold_h": (12, 20), "warm_h": (8, 12),   "disk_gb": (600, 1000)},
            "no_gpu":    {"cold_h": None,     "warm_h": None,      "disk_gb": (50, 100)},
        },
        "decode-sat": {
            "h100":      {"cold_h": (24, 40), "warm_h": (18, 36),  "disk_gb": (300, 600)},
            "h100_x8":   {"cold_h": (5, 9),   "warm_h": (4, 8),    "disk_gb": (300, 600)},
            "no_gpu":    {"cold_h": None,     "warm_h": None,      "disk_gb": (50, 100)},
        },
        "decode-full": {
            "h100":      {"cold_h": (72, 168), "warm_h": (48, 144), "disk_gb": (1200, 2500)},
            "h100_x8":   {"cold_h": (20, 40),  "warm_h": (18, 36),  "disk_gb": (1500, 3000)},
            "no_gpu":    {"cold_h": None,      "warm_h": None,      "disk_gb": (100, 200)},
        },
    }

    profile_table = table.get(profile, table["decode"])
    entry = profile_table.get(cls) or profile_table.get("h100")
    return {
        "profile": profile,
        "hardware_class": cls,
        "headline_gpu": host.gpu_names[0] if host.gpu_names else "no GPU detected",
        "gpu_count": host.gpu_count,
        "cold_setup_hours": entry["cold_h"],
        "warm_rerun_hours": entry["warm_h"],
        "disk_estimate_gb": entry["disk_gb"],
        "free_disk_gb": round(host.free_disk_gb, 1),
        "ram_gb": round(host.total_ram_gb, 1),
    }


def render(est: dict) -> str:
    lines = [
        f"Profile: {est['profile']}",
        f"Detected: {est['gpu_count']}× {est['headline_gpu']} (class={est['hardware_class']})",
    ]
    cold = est["cold_setup_hours"]
    warm = est["warm_rerun_hours"]
    if cold:
        lines.append(f"Cold setup estimate: {cold[0]}–{cold[1]} h")
    if warm:
        lines.append(f"Warm rerun estimate: {warm[0]}–{warm[1]} h")
    if cold is None:
        lines.append("No GPU detected: only build/lint/codec stages can run; full reproduction requires a GPU.")
    disk = est["disk_estimate_gb"]
    lines.append(f"Disk estimate: {disk[0]}–{disk[1]} GB  (free: {est['free_disk_gb']:.1f} GB)")
    lines.append(f"RAM: {est['ram_gb']:.1f} GB")
    return "\n".join(lines)
