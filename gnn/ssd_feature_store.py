#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""On-disk feature store that turns modeled read amplification into real I/O.

The in-RAM GNN benchmark holds the feature matrix in DRAM and *counts*
which 4 KiB pages a batch would touch. That is enough to compare layouts
analytically, but it issues no block-device reads, so external tooling
(eBPF biosnoop/biolatency, blktrace, iostat) cannot observe the read
intent. This module serves features from a raw file laid out by the page
layout and reads each batch's pages back with real I/O, so the read
amplification becomes physically measurable.

Layout convention (matches build_graph_layout.py):
    node_order[storage_pos] = original_id   # storage order permutation
We derive storage_pos[original_id] (the inverse) and place node records
contiguously in storage order, feature_bytes each. A node's device pages
are the 4 KiB pages its record spans:
    start = storage_pos * feature_bytes
    pages = floor(start / PS) .. floor((start + feature_bytes - 1) / PS)
Records can straddle a page boundary (68 B records do not divide 4096),
so we read every page a record spans. We report both the physical page
count (what the device actually serves) and the modeled start-page-only
count (what the in-RAM benchmark counts) so the two can be cross-checked.

Making a RAM-sized dataset exercise the SSD:
  - O_DIRECT (default): every logical page read becomes a device read
    regardless of dataset size. This is the clean way to verify per-I/O
    read intent without evicting caches.
  - evict (buffered fallback): posix_fadvise(DONTNEED) before each read.
  - replicas (inflate): write N copies and scatter reads across them so
    the working set genuinely exceeds page cache even with buffered I/O.

This module deliberately depends only on numpy + os so it can be tested
without torch. Run it directly for a self-test:

    python gnn/ssd_feature_store.py
"""
import mmap
import os
import numpy as np

PAGE_SIZE = 4096

# O_DIRECT is Linux-specific; fall back to 0 (no flag) elsewhere.
_O_DIRECT = getattr(os, "O_DIRECT", 0)


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _coalesce(pages: np.ndarray):
    """Group a sorted unique page-index array into (start_page, length) runs."""
    if len(pages) == 0:
        return []
    runs = []
    run_start = prev = int(pages[0])
    for p in pages[1:]:
        p = int(p)
        if p == prev + 1:
            prev = p
            continue
        runs.append((run_start, prev - run_start + 1))
        run_start = prev = p
    runs.append((run_start, prev - run_start + 1))
    return runs


def write_feature_file(x, node_order, path, page_size=PAGE_SIZE, replicas=1):
    """Write the feature matrix to ``path`` in storage order.

    Args:
        x: (N, F) float32-able feature matrix (numpy array or torch tensor).
        node_order: (N,) storage order permutation, storage_pos -> orig_id.
        path: output file path (place it on the SSD under test).
        page_size: device page size in bytes.
        replicas: number of contiguous copies to write (inflate the store
            past RAM). Each copy is padded to a page-size multiple so copy
            boundaries stay aligned for O_DIRECT.

    Returns:
        meta dict describing the file (N, F, feature_bytes, page_size,
        replicas, copy_bytes, data_bytes, file_bytes).
    """
    x = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N, F); got shape {x.shape}")
    n, f = x.shape
    feature_bytes = f * 4

    order = np.asarray(node_order).reshape(-1).astype(np.int64)
    if order.shape[0] != n:
        raise ValueError(f"node_order length {order.shape[0]} != N {n}")

    ordered = x[order]  # storage_pos k holds features of node order[k]
    data = ordered.tobytes()
    copy_bytes = _round_up(len(data), page_size)
    pad = copy_bytes - len(data)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    zeros = b"\x00" * pad if pad else b""
    with open(path, "wb") as fh:
        for _ in range(max(1, replicas)):
            fh.write(data)
            if zeros:
                fh.write(zeros)

    return {
        "N": int(n),
        "F": int(f),
        "feature_bytes": int(feature_bytes),
        "page_size": int(page_size),
        "replicas": int(max(1, replicas)),
        "copy_bytes": int(copy_bytes),
        "data_bytes": int(len(data)),
        "file_bytes": int(copy_bytes * max(1, replicas)),
    }


def storage_pos_from_order(node_order, n=None):
    """Invert a storage-order permutation: storage_pos[orig_id] = position."""
    order = np.asarray(node_order).reshape(-1).astype(np.int64)
    n = n if n is not None else order.shape[0]
    storage_pos = np.empty(n, dtype=np.int64)
    storage_pos[order] = np.arange(n, dtype=np.int64)
    return storage_pos


def drop_caches() -> bool:
    """Best-effort 'echo 3 > /proc/sys/vm/drop_caches'. Returns success."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3\n")
        return True
    except Exception:
        import subprocess

        try:
            rc = subprocess.call(
                ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
            )
            return rc == 0
        except Exception:
            return False


class SSDFeatureReader:
    """Read node features from an on-disk store with real, counted I/O."""

    def __init__(
        self,
        path,
        node_order,
        meta,
        direct=True,
        evict=False,
    ):
        self.path = path
        self.meta = dict(meta)
        self.FB = int(meta["feature_bytes"])
        self.F = int(meta["F"])
        self.N = int(meta["N"])
        self.page_size = int(meta.get("page_size", PAGE_SIZE))
        self.replicas = int(meta.get("replicas", 1))
        self.copy_bytes = int(meta.get("copy_bytes", 0))
        self.storage_pos = storage_pos_from_order(node_order, self.N)
        self.evict = bool(evict)
        self._replica = 0

        # Reset counters.
        self.reset_stats()

        # Open with O_DIRECT if requested, then probe it; some filesystems
        # (tmpfs, overlayfs) accept the flag but reject the read.
        self.direct = bool(direct) and _O_DIRECT != 0
        flags = os.O_RDONLY | (_O_DIRECT if self.direct else 0)
        self.fd = os.open(path, flags)
        if self.direct and not self._probe_direct():
            os.close(self.fd)
            self.direct = False
            self.evict = True  # without O_DIRECT, evict to still hit device
            self.fd = os.open(path, os.O_RDONLY)

    def reset_stats(self):
        self.pages_read = 0  # physical pages served by the device
        self.modeled_pages = 0  # start-page-only (matches in-RAM metric)
        self.useful_bytes = 0  # feature bytes actually requested
        self.nodes_read = 0
        self.read_ops = 0  # coalesced preadv/pread calls (≈ block I/Os)
        self.batches = 0

    def _probe_direct(self) -> bool:
        try:
            buf = mmap.mmap(-1, self.page_size)
            try:
                os.preadv(self.fd, [buf], 0)
            finally:
                buf.close()
            return True
        except OSError:
            return False

    def _read_run(self, run_start_page, run_len):
        offset = run_start_page * self.page_size
        length = run_len * self.page_size
        if self.direct:
            buf = mmap.mmap(-1, length)
            os.preadv(self.fd, [buf], offset)
            return memoryview(buf)
        if self.evict and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.fd, offset, length, os.POSIX_FADV_DONTNEED)
        return memoryview(os.pread(self.fd, length, offset))

    def read_nodes(self, node_ids):
        """Read features for ``node_ids`` via real I/O; update counters.

        Returns an (len(node_ids), F) float32 numpy array.
        """
        node_ids = np.asarray(node_ids).reshape(-1).astype(np.int64)
        if len(node_ids) == 0:
            return np.empty((0, self.F), dtype=np.float32)

        sp = self.storage_pos[node_ids]
        base = 0
        if self.replicas > 1:
            base = self._replica * self.copy_bytes
            self._replica = (self._replica + 1) % self.replicas

        start = base + sp * self.FB
        end = start + self.FB
        start_page = start // self.page_size
        end_page = (end - 1) // self.page_size

        # Physical pages: every page any record spans (handles straddle).
        pages = np.unique(np.concatenate([start_page, end_page]))
        runs = _coalesce(pages)

        run_starts = np.array([r[0] for r in runs], dtype=np.int64)
        run_bufs = [self._read_run(rs, rl) for (rs, rl) in runs]

        out = np.empty((len(node_ids), self.F), dtype=np.float32)
        ridx = np.searchsorted(run_starts, start_page, side="right") - 1
        for i in range(len(node_ids)):
            rs = int(run_starts[ridx[i]])
            local = int(start[i] - rs * self.page_size)
            rec = run_bufs[ridx[i]][local : local + self.FB]
            out[i] = np.frombuffer(rec, dtype=np.float32, count=self.F)

        # Accounting.
        phys_pages = int(sum(rl for (_, rl) in runs))
        modeled = int(np.unique((sp * self.FB) // self.page_size).size)
        self.pages_read += phys_pages
        self.modeled_pages += modeled
        self.useful_bytes += int(len(node_ids)) * self.FB
        self.nodes_read += int(len(node_ids))
        self.read_ops += len(runs)
        self.batches += 1
        return out

    @property
    def bytes_read(self) -> int:
        return self.pages_read * self.page_size

    @property
    def ra_physical(self) -> float:
        return self.bytes_read / self.useful_bytes if self.useful_bytes else 0.0

    @property
    def ra_modeled(self) -> float:
        if not self.useful_bytes:
            return 0.0
        return (self.modeled_pages * self.page_size) / self.useful_bytes

    def stats(self) -> dict:
        return {
            "direct": self.direct,
            "evict": self.evict,
            "batches": self.batches,
            "nodes_read": self.nodes_read,
            "pages_read": self.pages_read,
            "modeled_pages": self.modeled_pages,
            "read_ops": self.read_ops,
            "bytes_read": self.bytes_read,
            "useful_bytes": self.useful_bytes,
            "ra_physical": self.ra_physical,
            "ra_modeled": self.ra_modeled,
        }

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


def _self_test():
    import tempfile

    rng = np.random.default_rng(0)
    n, f = 500, 17
    x = rng.standard_normal((n, f)).astype(np.float32)

    for layout_name, order in (
        ("natural", np.arange(n)),
        ("random", rng.permutation(n)),
    ):
        for replicas in (1, 2):
            d = tempfile.mkdtemp(prefix="ssd_ft_")
            path = os.path.join(d, "feat.bin")
            meta = write_feature_file(x, order, path, replicas=replicas)
            r = SSDFeatureReader(path, order, meta, direct=True)

            # Read every node in shuffled batches and verify correctness.
            perm = rng.permutation(n)
            for s in range(0, n, 64):
                ids = perm[s : s + 64]
                got = r.read_nodes(ids)
                exp = x[ids]
                assert np.array_equal(
                    got, exp
                ), f"{layout_name}: feature mismatch at batch {s}"

            st = r.stats()
            # Physical RA must be >= modeled RA (straddle only adds pages).
            assert st["ra_physical"] >= st["ra_modeled"] - 1e-9
            # Every node's feature bytes were accounted for exactly once.
            assert st["nodes_read"] == n
            assert st["useful_bytes"] == n * meta["feature_bytes"]
            r.close()
            print(
                f"  {layout_name:7s} replicas={replicas} "
                f"direct={st['direct']} RA_phys={st['ra_physical']:.3f} "
                f"RA_model={st['ra_modeled']:.3f} pages={st['pages_read']} "
                f"ops={st['read_ops']}"
            )
            os.remove(path)
            os.rmdir(d)
    print("ssd_feature_store self-test: OK")


if __name__ == "__main__":
    _self_test()
