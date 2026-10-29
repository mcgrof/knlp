# CS312 tutorial lane

Sweeps from the CS312 "Deep Learning Alchemy" assignments on the course's
small decoder, driven through knlp's Kconfig and make targets. The course
repository is an external checkout pinned by commit; this directory holds
only the planner (`run.py`), the entry script that runs inside the course
environment (`sweep_entry.py`), and a Modal data-volume seeder
(`modal_seed_data.py`). The full tutorial, including both backends and the
report format, is in [docs/cs312-tutorial.md](../../../docs/cs312-tutorial.md).

```sh
make defconfig-cs312-problem-1        # Modal H100s through the course launcher
make defconfig-cs312-problem-1-local  # one local GPU, runs in sequence
make cs312-fetch && make cs312-data && make
make cs312-test                       # planner unit tests, no torch needed
```
