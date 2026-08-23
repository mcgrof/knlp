"""Same-shape parameter batching for the vendored reference SOAP.

The reference implementation in `fim/fisher_pruning/soap.py` walks the
parameter list in Python and runs every linear-algebra op on one
parameter at a time. A GPT-2 124M refresh therefore issues roughly
ninety separate eigendecompositions/QR factorizations plus several
hundred small matmuls, and the step is dominated by kernel launch and
dispatch overhead rather than arithmetic.

Every operation SOAP performs is either elementwise or a per-parameter
matmul/decomposition, and nothing couples one parameter to another. So
the twelve identical `[768, 3072]` MLP weights of a GPT-2 can be held
as one `[12, 768, 3072]` slab and driven with `torch.bmm`,
`torch.linalg.eigh` and `torch.linalg.qr`, all of which are batched.
The arithmetic each parameter sees is unchanged; only the kernel
granularity changes.

This class is a drop-in replacement, not an improvement. Where the
reference has a quirk it is replicated verbatim, including:

  * the first gradient a parameter presents is folded into the
    preconditioner and then the step is skipped, so a gradient never
    enters its own projection;
  * the per-parameter step counter and its Adam bias correction;
  * `exp_avg` is projected back and re-projected around every
    preconditioner update, while `exp_avg_sq` is only re-indexed at a
    refresh, to keep it aligned with the re-sorted eigenbasis;
  * the eigenbasis refresh casts `GG` and `Q` down to float32 before
    the power iteration and QR, then casts the result back;
  * dimensions larger than `max_precond_dim` are left unpreconditioned
    and 1-D parameters are only preconditioned when asked;
  * `normalize_grads` normalizes by a *per-parameter* RMS.

It subclasses the reference so that parameters which cannot be batched
(a shape that occurs once, or a channels-last 4-D tensor) run through
the reference's own `project` / `update_preconditioner` /
`get_orthogonal_matrix*` code rather than through a re-derivation of
it.

Deviations from the reference are documented at `_step_bucket`,
`_plan_group` and `state_dict`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.soap import SOAP  # noqa: E402


class _Bucket:
    """A set of parameters sharing (shape, dtype, device).

    `state` mirrors the reference's per-parameter state dict, except
    every tensor carries a leading K axis and an unpreconditioned
    dimension is stored as ``None`` where the reference stores ``[]``.
    """

    __slots__ = ("params", "shape", "ndim", "state", "dead")

    def __init__(self, params):
        self.params = list(params)
        self.shape = tuple(self.params[0].shape)
        self.ndim = len(self.shape)
        self.state = {}
        self.dead = False

    def __len__(self):
        return len(self.params)


def _is_mat(x):
    return isinstance(x, torch.Tensor)


class BatchedSOAP(SOAP):
    """SOAP with same-shape parameters batched into stacked tensors.

    Constructor signature and semantics are the reference's. See the
    module docstring for what is preserved and what deviates.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            data_format=data_format,
            correct_bias=correct_bias,
        )
        # group index -> list[_Bucket]
        self._buckets = {}
        self._planned = set()
        self._batched_ids = set()

    # ------------------------------------------------------------------
    # bucketing
    # ------------------------------------------------------------------

    def _unbatchable(self, p) -> bool:
        # merge_dims permutes 4-D channels-last tensors before merging;
        # rather than replicate that permutation under a batch axis,
        # such parameters take the reference path.
        return self._data_format == "channels_last" and p.dim() == 4

    def _plan_group(self, gi, group):
        """Partition a group into buckets, once.

        DEVIATION: the partition is decided on the first step at which
        the group has any gradient, using only the parameters that have
        one. A parameter whose gradient first appears later takes the
        unbatched reference path for the rest of the run. Shapes that
        occur once are never bucketed -- stacking a single tensor only
        adds a copy.
        """
        if gi in self._planned:
            return self._buckets.get(gi, [])
        have = [p for p in group["params"] if p.grad is not None]
        if not have:
            return []
        keyed = {}
        for p in have:
            if self._unbatchable(p):
                continue
            keyed.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
        buckets = [_Bucket(ps) for ps in keyed.values() if len(ps) > 1]
        self._buckets[gi] = buckets
        self._planned.add(gi)
        for b in buckets:
            for p in b.params:
                self._batched_ids.add(id(p))
        return buckets

    def bucket_report(self):
        """[(shape, dtype, device, count)] per batched bucket.

        Exposed for tests and for reporting what a model actually
        batches into.
        """
        out = []
        for gi in sorted(self._buckets):
            for b in self._buckets[gi]:
                if b.dead:
                    continue
                p0 = b.params[0]
                out.append((b.shape, p0.dtype, p0.device, len(b.params)))
        return out

    def _dissolve(self, b):
        """Turn a bucket back into independent per-parameter state.

        Used when a bucket member loses its gradient: the members'
        step counters would then diverge and the shared refresh
        schedule would no longer describe all of them.
        """
        self._publish(b, clone=True)
        for p in b.params:
            self._batched_ids.discard(id(p))
        b.dead = True

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            loss = closure()

        for gi, group in enumerate(self.param_groups):
            for b in self._plan_group(gi, group):
                if not b.dead:
                    self._step_bucket(group, b)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if id(p) in self._batched_ids:
                    continue
                self._step_param(group, p)
        return loss

    def _step_param(self, group, p):
        """The reference per-parameter body, verbatim.

        Copied from SOAP.step's loop body so that unbatched parameters
        run exactly the reference code path (the helpers it calls are
        the inherited reference methods).
        """
        grad = p.grad

        state = self.state[p]

        if "step" not in state:
            state["step"] = 0

        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)

        if "Q" not in state:
            self.init_preconditioner(
                grad,
                state,
                precondition_frequency=group["precondition_frequency"],
                precondition_1d=group["precondition_1d"],
                shampoo_beta=(
                    group["shampoo_beta"]
                    if group["shampoo_beta"] >= 0
                    else group["betas"][1]
                ),
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
            )
            self.update_preconditioner(
                grad,
                state,
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
                precondition_1d=group["precondition_1d"],
            )
            return  # first step is skipped, see the reference

        grad_projected = self.project(
            grad,
            state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        state["step"] += 1

        exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
        exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

        denom = exp_avg_sq.sqrt().add_(group["eps"])

        exp_avg_projected = exp_avg

        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** (state["step"])
            bias_correction2 = 1.0 - beta2 ** (state["step"])
            step_size = step_size * (bias_correction2**0.5) / bias_correction1

        norm_grad = self.project_back(
            exp_avg_projected / denom,
            state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        if group["normalize_grads"]:
            norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad**2) ** 0.5)

        p.add_(norm_grad, alpha=-step_size)

        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        self.update_preconditioner(
            grad,
            state,
            max_precond_dim=group["max_precond_dim"],
            merge_dims=group["merge_dims"],
            precondition_1d=group["precondition_1d"],
        )

    def _step_bucket(self, group, b):
        """The same body as `_step_param`, with a leading K axis.

        DEVIATION (ordering only): buckets are processed before the
        group's unbatched parameters instead of in parameter order.
        Parameters do not interact, so this changes nothing an
        individual parameter computes.
        """
        grads = [p.grad for p in b.params]
        if any(g is None for g in grads):
            # A member stopped producing gradients: its step counter
            # would fall behind the rest of the bucket. Split up and
            # leave the survivors to step()'s own parameter loop,
            # which steps every id no longer marked as batched —
            # stepping them here too would apply the update, the
            # weight decay and the counter increment twice.
            self._dissolve(b)
            return

        grad = torch.stack(grads)
        st = b.state

        if "step" not in st:
            st["step"] = 0

        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(grad)
            st["exp_avg_sq"] = torch.zeros_like(grad)

        if "Q" not in st:
            self._b_init_preconditioner(
                grad,
                b,
                precondition_frequency=group["precondition_frequency"],
                precondition_1d=group["precondition_1d"],
                shampoo_beta=(
                    group["shampoo_beta"]
                    if group["shampoo_beta"] >= 0
                    else group["betas"][1]
                ),
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
            )
            self._b_update_preconditioner(
                grad,
                b,
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
                precondition_1d=group["precondition_1d"],
            )
            self._publish(b)
            return  # first step is skipped, see the reference

        grad_projected = self._b_project(
            grad,
            b,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        st["step"] += 1

        exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
        exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

        denom = exp_avg_sq.sqrt().add_(group["eps"])

        exp_avg_projected = exp_avg

        step_size = group["lr"]
        if group["correct_bias"]:
            bias_correction1 = 1.0 - beta1 ** (st["step"])
            bias_correction2 = 1.0 - beta2 ** (st["step"])
            step_size = step_size * (bias_correction2**0.5) / bias_correction1

        norm_grad = self._b_project_back(
            exp_avg_projected / denom,
            b,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        if group["normalize_grads"]:
            # Per parameter, as in the reference -- NOT over the slab.
            dims = tuple(range(1, norm_grad.dim()))
            norm_grad = norm_grad / (
                1e-30 + torch.mean(norm_grad**2, dim=dims, keepdim=True) ** 0.5
            )

        torch._foreach_add_(b.params, list(norm_grad.unbind(0)), alpha=-step_size)

        if group["weight_decay"] > 0.0:
            torch._foreach_add_(
                b.params,
                list(b.params),
                alpha=(-group["lr"] * group["weight_decay"]),
            )

        self._b_update_preconditioner(
            grad,
            b,
            max_precond_dim=group["max_precond_dim"],
            merge_dims=group["merge_dims"],
            precondition_1d=group["precondition_1d"],
        )
        self._publish(b)

    # ------------------------------------------------------------------
    # batched linear algebra
    # ------------------------------------------------------------------

    def _merged_shape(self, shape, max_precond_dim):
        """`SOAP.merge_dims`'s shape arithmetic, on a logical shape.

        Only reached for tensors the batched path accepts, so the
        channels-last 4-D permutation the reference applies first
        cannot arise here.
        """
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return new_shape

    @staticmethod
    def _contract(t, mat, back):
        """Batched `tensordot(t, mat, dims=[[0], [0 or 1]])`.

        `t` is [K, d, *rest] and `mat` is [K, d, d]. The reference
        contracts the leading (non-batch) dimension away and appends
        the result dimension at the end, cycling the axes; this does
        the same under the batch axis.
        """
        K = t.shape[0]
        d = t.shape[1]
        rest = t.shape[2:]
        g = t.reshape(K, d, -1)
        # project:      out[k, b, r] = sum_a mat[k, a, b] * g[k, a, r]
        # project_back: out[k, i, r] = sum_a mat[k, i, a] * g[k, a, r]
        out = torch.bmm(mat if back else mat.transpose(1, 2), g)
        out = out.reshape(K, out.shape[1], *rest)
        return out.movedim(1, -1)

    def _b_project(self, t, b, merge_dims=False, max_precond_dim=10000):
        original_shape = t.shape
        if merge_dims:
            t = t.reshape(t.shape[0], *self._merged_shape(b.shape, max_precond_dim))
        for mat in b.state["Q"]:
            if _is_mat(mat):
                t = self._contract(t, mat, back=False)
            else:
                t = t.movedim(1, -1)
        if merge_dims:
            t = t.reshape(original_shape)
        return t

    def _b_project_back(self, t, b, merge_dims=False, max_precond_dim=10000):
        original_shape = t.shape
        if merge_dims:
            t = t.reshape(t.shape[0], *self._merged_shape(b.shape, max_precond_dim))
        for mat in b.state["Q"]:
            if _is_mat(mat):
                t = self._contract(t, mat, back=True)
            else:
                t = t.movedim(1, -1)
        if merge_dims:
            t = t.reshape(original_shape)
        return t

    @staticmethod
    def _b_outer(g, idx):
        """Batched contraction of everything but dimension `idx`."""
        K = g.shape[0]
        d = g.shape[1 + idx]
        gm = g.movedim(1 + idx, 1).reshape(K, d, -1)
        return torch.bmm(gm, gm.transpose(1, 2))

    def _b_init_preconditioner(
        self,
        grad,
        b,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        st = b.state
        K = grad.shape[0]
        st["GG"] = []
        if b.ndim == 1:
            if not precondition_1d or b.shape[0] > max_precond_dim:
                st["GG"].append(None)
            else:
                st["GG"].append(
                    torch.zeros(K, b.shape[0], b.shape[0], device=grad.device)
                )
        else:
            shape = b.shape
            if merge_dims:
                shape = self._merged_shape(b.shape, max_precond_dim)
            for sh in shape:
                if sh > max_precond_dim:
                    st["GG"].append(None)
                else:
                    # dtype deliberately left to the default, as the
                    # reference's torch.zeros(sh, sh, device=...) does.
                    st["GG"].append(torch.zeros(K, sh, sh, device=grad.device))

        st["Q"] = None
        st["precondition_frequency"] = precondition_frequency
        st["shampoo_beta"] = shampoo_beta

    def _b_update_preconditioner(
        self,
        grad,
        b,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
    ):
        st = b.state
        if st["Q"] is not None:
            # `.contiguous()` here and below is not cosmetic: the
            # contraction leaves its result as a permuted view, and a
            # strided gemm operand rounds differently from a packed
            # one. Storing `exp_avg` in a canonical layout is what
            # makes a checkpoint round trip resume bitwise identically
            # instead of merely to float64 rounding, and it matches
            # what the reference gets for free (`torch.tensordot`
            # reshapes a permuted tensor, which copies).
            st["exp_avg"] = self._b_project_back(
                st["exp_avg"],
                b,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            ).contiguous()
        if b.ndim == 1:
            if precondition_1d and b.shape[0] <= max_precond_dim:
                st["GG"][0].lerp_(
                    grad.unsqueeze(2) @ grad.unsqueeze(1),
                    1 - st["shampoo_beta"],
                )
        else:
            g = grad
            if merge_dims:
                g = grad.reshape(
                    grad.shape[0], *self._merged_shape(b.shape, max_precond_dim)
                )
            for idx, sh in enumerate(g.shape[1:]):
                if sh <= max_precond_dim:
                    outer_product = self._b_outer(g, idx)
                    st["GG"][idx].lerp_(outer_product, 1 - st["shampoo_beta"])

        if st["Q"] is None:
            st["Q"] = self._b_get_orthogonal_matrix(st["GG"])
        if st["step"] > 0 and st["step"] % st["precondition_frequency"] == 0:
            st["Q"] = self._b_get_orthogonal_matrix_QR(b, max_precond_dim, merge_dims)

        if st["step"] > 0:
            st["exp_avg"] = self._b_project(
                st["exp_avg"],
                b,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            ).contiguous()

    def _b_get_orthogonal_matrix(self, mat):
        """Batched `SOAP.get_orthogonal_matrix`.

        The reference's `float_data` / `original_type` are assigned
        inside its loop and read after it, so the last non-empty
        factor decides the cast for every factor. Within a bucket all
        factors share a dtype, so replicating the leak is trivial --
        but it is replicated, not fixed.
        """
        matrix = []
        float_data = None
        original_type = original_device = None
        for m in mat:
            if not _is_mat(m):
                matrix.append(None)
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if m is None:
                final.append(None)
                continue
            n = m.shape[-1]
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(n, device=m.device))
            except Exception:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-30 * torch.eye(n, device=m.device)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [-1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q.contiguous())
        return final

    @staticmethod
    def _b_index_select(t, dim, idx):
        """Per-slab `index_select`: row k of `t` uses row k of `idx`."""
        view = [1] * t.dim()
        view[0] = idx.shape[0]
        view[dim] = idx.shape[1]
        expand = list(t.shape)
        expand[dim] = idx.shape[1]
        return torch.gather(t, dim, idx.view(view).expand(expand))

    def _b_get_orthogonal_matrix_QR(self, b, max_precond_dim=10000, merge_dims=False):
        """Batched `SOAP.get_orthogonal_matrix_QR`.

        Note that the reference does the power iteration and the QR in
        float32 unconditionally (`m.data.float()` in both branches),
        even when `GG` is float64. That is replicated.
        """
        st = b.state
        precond_list = st["GG"]
        orth_list = st["Q"]

        matrix = []
        orth_matrix = []
        float_data = None
        original_type = original_device = None
        for m, o in zip(precond_list, orth_list):
            if not _is_mat(m):
                matrix.append(None)
                orth_matrix.append(None)
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
            else:
                float_data = True
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

        orig_shape = st["exp_avg_sq"].shape
        if merge_dims:
            exp_avg_sq = st["exp_avg_sq"].reshape(
                orig_shape[0], *self._merged_shape(b.shape, max_precond_dim)
            )
        else:
            exp_avg_sq = st["exp_avg_sq"]

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if m is None:
                final.append(None)
                continue
            est_eig = torch.diagonal(
                torch.bmm(torch.bmm(o.transpose(1, 2), m), o), dim1=1, dim2=2
            )
            sort_idx = torch.argsort(est_eig, descending=True, dim=-1)
            exp_avg_sq = self._b_index_select(exp_avg_sq, 1 + ind, sort_idx)
            o = torch.gather(o, 2, sort_idx.unsqueeze(1).expand(-1, o.shape[1], -1))
            power_iter = torch.bmm(m, o)
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            # LAPACK hands back a column-major Q. Canonicalize, for the
            # same reason `exp_avg` is canonicalized above: a restacked
            # checkpoint would otherwise feed the gemms a differently
            # strided operand and round differently.
            final.append(Q.contiguous())

        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        st["exp_avg_sq"] = exp_avg_sq
        return final

    # ------------------------------------------------------------------
    # per-parameter state views, checkpointing
    # ------------------------------------------------------------------

    def _publish(self, b, clone=False):
        """Mirror the bucket's stacked state into `self.state[p]`.

        The mirrored tensors are views into the slab, so anything that
        reads `optimizer.state[p]["exp_avg_sq"]` or `["Q"]` (the
        pruning signal harvest in `train_optimizer.live_state_scores`,
        for instance) sees the live values. `clone=True` materializes
        them instead, which is what checkpointing and bucket
        dissolution need.
        """
        st = b.state
        GG = st.get("GG")
        Q = st.get("Q")

        def take(t):
            return t.clone() if clone else t

        for k, p in enumerate(b.params):
            s = self.state[p]
            s["step"] = st["step"]
            s["exp_avg"] = take(st["exp_avg"][k])
            s["exp_avg_sq"] = take(st["exp_avg_sq"][k])
            s["precondition_frequency"] = st["precondition_frequency"]
            s["shampoo_beta"] = st["shampoo_beta"]
            s["GG"] = [take(g[k]) if _is_mat(g) else [] for g in GG]
            if Q is None:
                s["Q"] = None
            else:
                s["Q"] = [take(q[k]) if _is_mat(q) else [] for q in Q]

    def _restack(self, b):
        """Inverse of `_publish`: gather per-parameter state into a slab."""
        st = b.state
        first = self.state[b.params[0]]
        steps = {self.state[p]["step"] for p in b.params}
        if len(steps) != 1:
            raise RuntimeError(
                f"bucket {b.shape} has mixed step counters {sorted(steps)}; "
                "state_dict was not produced by a batched run"
            )
        st["step"] = first["step"]
        st["precondition_frequency"] = first["precondition_frequency"]
        st["shampoo_beta"] = first["shampoo_beta"]
        st["exp_avg"] = torch.stack([self.state[p]["exp_avg"] for p in b.params])
        st["exp_avg_sq"] = torch.stack([self.state[p]["exp_avg_sq"] for p in b.params])
        st["GG"] = [
            (
                torch.stack([self.state[p]["GG"][i] for p in b.params])
                if _is_mat(first["GG"][i])
                else None
            )
            for i in range(len(first["GG"]))
        ]
        if first.get("Q") is None:
            st["Q"] = None
        else:
            st["Q"] = [
                (
                    torch.stack([self.state[p]["Q"][i] for p in b.params])
                    if _is_mat(first["Q"][i])
                    else None
                )
                for i in range(len(first["Q"]))
            ]
        self._publish(b)

    def state_dict(self):
        """A standard per-Parameter optimizer state_dict.

        The stacked slabs are unstacked into independent per-parameter
        tensors first, so the file is byte-for-byte the shape a
        reference-SOAP checkpoint has (same keys, same per-parameter
        shapes) and neither implementation can read the other's
        checkpoint wrongly. Cloning matters: the mirrored state
        normally holds views, and `torch.save` of a view serializes
        the whole underlying slab.
        """
        for buckets in self._buckets.values():
            for b in buckets:
                if not b.dead:
                    self._publish(b, clone=True)
        sd = super().state_dict()
        # torch.optim.Optimizer.state_dict() re-keys `self.state` but
        # keeps the inner dicts by reference, so restoring the views
        # below would reach into the returned dict. Detach them.
        sd["state"] = {k: dict(v) for k, v in sd["state"].items()}
        for buckets in self._buckets.values():
            for b in buckets:
                if not b.dead:
                    self._publish(b)
        return sd

    def load_state_dict(self, state_dict):
        """Load a per-parameter state_dict and re-form the buckets.

        DEVIATION: bucketing is re-derived from which parameters carry
        state, rather than from which carried gradients at the first
        step. For any parameter that had ever taken a step these agree.
        """
        super().load_state_dict(state_dict)
        self._buckets = {}
        self._planned = set()
        self._batched_ids = set()
        for gi, group in enumerate(self.param_groups):
            keyed = {}
            for p in group["params"]:
                s = self.state.get(p)
                if not s or "Q" not in s:
                    continue
                if self._unbatchable(p):
                    continue
                keyed.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
            buckets = []
            for ps in keyed.values():
                if len(ps) < 2:
                    continue
                b = _Bucket(ps)
                self._restack(b)
                buckets.append(b)
                for p in ps:
                    self._batched_ids.add(id(p))
            self._buckets[gi] = buckets
            self._planned.add(gi)
