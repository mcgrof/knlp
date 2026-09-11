import importlib.util
from pathlib import Path


REPORT = (
    Path(__file__).parents[1]
    / "tools"
    / "reproduce"
    / "kvtide"
    / "pcie-report.py"
)
SPEC = importlib.util.spec_from_file_location("kvtide_pcie_report", REPORT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_report_uses_recorded_medians_and_ignores_warmups(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "profile,arm,tool,driver,pattern,iosize,qd,threads,rep,iops,"
        "MiBps,failed,wall_seconds,system_cpu_cores\n"
        "kv,premap,uring_nvm_perf,nvme,randread,4096,16,1,warmup-1,"
        "9999,39,0,1,1\n"
        "kv,premap,uring_nvm_perf,nvme,randread,4096,16,1,1,200,"
        "0.78,0,1,0.8\n"
        "kv,premap,uring_nvm_perf,nvme,randread,4096,16,1,2,300,"
        "1.17,0,1,1.2\n"
        "kv,spdk,spdk_nvme_perf,vfio-pci,randread,4096,16,1,1,100,"
        "0.39,0,1,1.0\n"
        "kv,spdk,spdk_nvme_perf,vfio-pci,randread,4096,16,1,2,ERR,"
        "ERR,ERR,1,1.0\n"
        "kv,linux,xnvmeperf,nvme,randread,4096,16,1,1,0,0,99,1,"
        "1.0\n",
        encoding="utf-8",
    )

    summary = MODULE.summarize(MODULE.load_rows(csv_path))
    rendered = MODULE.render(summary)

    assert summary[("kv", 4096, 16, 1, "premap")]["iops"] == 250
    assert "2.50x  premap higher" in rendered
    assert "9999" not in rendered
    assert ("kv", 4096, 16, 1, "linux") not in summary
