#!/usr/bin/env python3
"""
Trellis in one readable file: the memory operator, and a router that uses it.

This is a teaching companion to trellis_lm/. The production operator
(trellis_lm/trellis_memory.py) carries ~20 keyword arguments for stabilizers,
chunk kernels, gradient-mode dispatch and ablation switches, which makes the
actual algorithm hard to see. Everything here is the default path, spelled out.

THE MECHANISM

Every model in this family keeps a fixed-size recurrent state instead of a
growing KV cache. The state is a fast-weight memory that each token WRITES to
with one online learning step, and each token READS from with a plain matmul.

  DeltaNet        S <- S + beta_t (v_t - S k_t) k_t^T      linear write
  GatedDeltaNet   S <- a_t S + beta_t (v_t - a_t S k_t) k_t^T   + forget gate
  Trellis         M <- beta_t M - gamma u_t w_t^T          NONLINEAR write

The symbols, all of them, before any of them get used:

  M       the memory itself: `n_slots` rows of head dimension D. This is the
          whole recurrent state. It is NOT a parameter -- it starts at zero for
          every sequence and is rewritten token by token.
  w_t     the WRITE vector for token t: a learned linear projection of the
          token's hidden state, W_w x_t, L2-normalized. It plays the role of
          DeltaNet's key k_t -- the address this token writes at.
  z_t     = M w_t, a vector of `n_slots` numbers. What the memory currently
          says when addressed by w_t. Recomputed every step, since M changes.
  alpha_t the TARGET code for token t: another learned linear projection,
          W_a x_t, also `n_slots` wide. What the token wants z_t to look like.
  u_t     the ERROR code: how wrong the memory currently is, in slot space,
          measured through phi. Defined below.
  beta_t  a forget gate in (0,1), predicted from the token.
  gamma   a per-head step size, one positive scalar.
  phi     a fixed elementwise nonlinearity applied across the slot axis.

"Learned per-token" everywhere above means "produced by a learned projection
that is applied to each token", not "one parameter per position". The weights
W_w and W_a are shared across all positions and all sequences; what varies per
token is their output. There is nothing in the model indexed by t.

Now the rule. Token t takes ONE gradient step on its own private objective

    L_t(M) = 1/2 || phi(M w_t) - alpha_t ||^2

that is: "adjust the memory so that reading it at address w_t returns alpha_t."
The gradient with respect to z_t = M w_t is

    u_t = J_phi(z_t)^T (phi(z_t) - alpha_t)

-- the raw residual phi(z_t) - alpha_t, pulled back through phi's Jacobian.
Since dz/dM = w_t, the gradient with respect to M is the outer product
u_t w_t^T, and one gated descent step is exactly the update at the top. Read is
write-before-read: update M with the current token, then query it.

That is the whole model. The single difference from a gated delta rule is phi.
Set phi = identity: J_phi = I, so u_t = z_t - alpha_t, which is linear in M, and
the update IS the gated delta rule. Run this file with --phi identity to see
that control in the same shell.

WHY THE NONLINEARITY COSTS SPEED (and what is actually done about it)

With phi linear, a whole chunk of C tokens collapses into a single affine map on
the state, M_out = A M_in + b, computed with a couple of matmuls. That identity
is what gives DeltaNet and Gated DeltaNet their exact chunk kernels. With phi
nonlinear it does not hold, because u_t depends nonlinearly on M.

Trellis is still parallelized, by approximating rather than by an exact
identity. In trellis_lm's chunked path every token in a chunk computes its z
against the CHUNK-START state M0 instead of the live state:

    z_t = M0 w_t      for all t in the chunk    (one batched matmul)
    u_t = J_phi(z_t)^T (phi(z_t) - alpha_t)     (still every token)
    M   = decay * M0 - gamma * sum_t r_t u_t w_t^T   (one batched matmul)

So the nonlinearity is evaluated once per token, exactly as in the sequential
rule -- it is not skipped or applied every few steps. What is stale is the
STATE it sees: within a chunk the memory is frozen, and it advances once per
chunk. chunk_size=1 recovers the exact sequential recurrence, which is what this
file implements and what the router uses. The 2026-08 audit fixed the GRADIENT
semantics of that path (whether the outer loss differentiates through the inner
step); it did not remove the forward staleness, which is a deliberate
approximation.

TWO PASSES

A Trellis layer runs the operator twice. A key pass writes keys and reads with
queries, producing an intermediate code; a fixed map f (an L2/LayerNorm-SiLU)
turns that code into a read vector; a value pass writes values and reads with
that vector, producing the layer output. Both passes share the write target
alpha. The router below is that structure, standing alone as an episodic memory:
bind alias->tool pairs into the two memories, then query by alias.

THE TASK: an in-context key-value store

The router is a stand-in for what a real Trellis layer has to do inside a single
context window: read a fact stated earlier in the sequence, and honor it when it
is later contradicted. Stripped of language, that is a key-value store built at
inference time.

Concretely, one episode is a short conversation of the form

    "the tool called <alias A> is the calculator"      bind
    "the tool called <alias B> is the SQL runner"      bind
    "what is <alias A>?"                               query  -> calculator
    "actually, <alias A> is the shell"                 overwrite
    "what is <alias A>?"                               query  -> shell, not calculator

except the aliases are not words, they are freshly drawn random 32-d vectors,
different in every episode. That is the point. The slow weights literally cannot
memorize which alias maps to which tool, because no alias is ever seen twice.
The only thing left for them to learn is a PROTOCOL: how to turn a token into a
write address, what target code to aim for, how hard to step, how to read back.
All the episode-specific content lives in M, which is thrown away afterward.

The overwrite is the sharp part. A memory that merely accumulates will answer
with the stale tool forever; getting it right requires the write to subtract the
old association, which is exactly what the error code u_t does.

  python scripts/trellis_minimal.py                  # nonlinear write (SiLU)
  python scripts/trellis_minimal.py --phi identity   # gated delta-rule control
  python scripts/trellis_minimal.py --show-episode   # print one episode in full

Both arms are checked against the production operator before training.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# The memory operator
# ---------------------------------------------------------------------------


def error_code(z: torch.Tensor, alpha: torch.Tensor, phi: str) -> torch.Tensor:
    """u = J_phi(z)^T (phi(z) - alpha), written out instead of autograd'd.

    Both branches are ordinary differentiable tensor ops, so if `z` is still
    attached to the graph the outer backward differentiates through the inner
    step too -- that is the exact bilevel gradient, for free, with no nested
    autograd.grad. Detaching `z` before the call gives the cheaper first-order
    approximation instead.
    """
    if phi == "identity":
        # J = I, so u is just the residual: the (gated) delta rule.
        return z - alpha
    # phi = SiLU is elementwise, so J is diagonal and the VJP is a product.
    sig = torch.sigmoid(z)
    s = z * sig  # silu(z)
    dsilu = sig + s * (1.0 - sig)  # silu'(z)
    return dsilu * (s - alpha)


def trellis_scan(
    write: torch.Tensor,  # [B,H,T,D]  what each token writes into memory
    alpha: torch.Tensor,  # [B,H,T,M]  each token's target slot code
    beta: torch.Tensor,  # [B,H,T,1]  per-token forget gate in (0,1)
    gamma: torch.Tensor,  # [H]        per-head inner step size, positive
    phi: str,
    M0: torch.Tensor,  # [B,H,M,D]  incoming state
    read: torch.Tensor | None = None,  # [B,H,T,D] optional per-step query
):
    """Run the Trellis recurrence over T tokens. Returns (final_state, reads).

    `read` is what a language model would use: the write-before-read output
    y_t = M_t r_t at every step. The router below binds with read=None (it only
    wants the resulting state) and queries the finished state separately.
    """
    state = M0
    g = gamma.view(1, -1, 1, 1)  # broadcast over [B,H,M,D]
    outputs = []

    for t in range(write.shape[2]):
        w = write[:, :, t, :]  # [B,H,D]
        z = torch.einsum("bhmd,bhd->bhm", state, w)  # slot code
        u = error_code(z, alpha[:, :, t, :], phi)  # error code
        update = torch.einsum("bhm,bhd->bhmd", u, w)  # outer(u, w)
        state = beta[:, :, t, :].unsqueeze(-1) * state - g * update

        if read is not None:
            outputs.append(torch.einsum("bhmd,bhd->bhm", state, read[:, :, t, :]))

    reads = torch.stack(outputs, dim=2) if read is not None else None
    return state, reads


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------


class MiniRouter(nn.Module):
    """Two Trellis memories used as an episodic alias -> tool table.

    key memory:    written with key(alias),  targets the slot code alpha(alias)
    value memory:  written with vec(tool),   targets the same alpha(alias)

    So both memories are addressed by the SAME internal code. A query then goes
    alias -> key memory -> code -> f -> value memory -> reconstructed tool
    vector, and the tool vectors double as the classifier (a tied readout).
    """

    def __init__(self, phi: str, n_tools: int = 6, n_slots: int = 32):
        super().__init__()
        self.phi = phi
        self.n_slots = n_slots
        self.d_head = 32
        d_model, alias_dim = 64, 32

        self.encode = nn.Sequential(
            nn.Linear(alias_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.RMSNorm(d_model),
        )
        self.to_key = nn.Linear(d_model, self.d_head, bias=False)
        self.to_alpha = nn.Linear(d_model, n_slots, bias=False)
        self.tools = nn.Embedding(n_tools, self.d_head)

        # gamma is kept positive by softplus; beta is one learned scalar pinned
        # into a high-retention band (the full model predicts it per token).
        self.gamma_raw = nn.Parameter(torch.tensor([math.log(math.expm1(0.05))]))
        self.beta_raw = nn.Parameter(torch.zeros(1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        nn.init.normal_(self.to_alpha.weight, std=0.02)
        nn.init.normal_(self.tools.weight, std=0.02)

    def _address(self, aliases: torch.Tensor):
        """aliases [B,T,32] -> key [B,1,T,D], alpha [B,1,T,M], beta [B,1,T,1]."""
        h = self.encode(aliases)
        key = F.normalize(self.to_key(h), dim=-1).unsqueeze(1)
        alpha = self.to_alpha(h).unsqueeze(1)
        beta = 0.98 + 0.019 * torch.sigmoid(self.beta_raw)
        beta = beta.expand(aliases.shape[0], 1, aliases.shape[1], 1)
        return key, alpha, beta

    def _tool_bank(self) -> torch.Tensor:
        return F.normalize(self.tools.weight, dim=-1)  # [n_tools, D]

    def blank_state(self, batch: int, device, dtype):
        shape = (batch, 1, self.n_slots, self.d_head)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def bind(self, state, aliases: torch.Tensor, tool_ids: torch.Tensor):
        """Write alias->tool pairs into the fast memories. No parameters move."""
        key_state, value_state = state
        key, alpha, beta = self._address(aliases)
        value = self._tool_bank()[tool_ids].unsqueeze(1)  # [B,1,T,D]
        gamma = F.softplus(self.gamma_raw)

        key_state, _ = trellis_scan(key, alpha, beta, gamma, self.phi, key_state)
        value_state, _ = trellis_scan(value, alpha, beta, gamma, self.phi, value_state)
        return key_state, value_state

    def query(self, state, aliases: torch.Tensor) -> torch.Tensor:
        """Read the finished memories. Returns tool logits [B,Q,n_tools]."""
        key_state, value_state = state
        query, _, _ = self._address(aliases)

        code = torch.einsum("bhmd,bhqd->bhqm", key_state, query)
        code = F.silu(code)
        code = code - code.mean(-1, keepdim=True)
        code = code / (code.std(-1, keepdim=True, unbiased=False) + 1e-6)  # f

        got = torch.einsum("bhmd,bhqm->bhqd", value_state, code)
        got = F.normalize(got, dim=-1).squeeze(1)

        logits = got @ self._tool_bank().T
        return logits * self.logit_scale.exp().clamp(1.0, 100.0)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

# Names only exist so the printout reads like something. The model sees ids.
TOOL_NAMES = ("calculator", "web-search", "sql-runner", "shell", "emailer", "plotter")


def make_episode(batch: int, n_bindings: int, n_tools: int, gen, device):
    """Fresh random aliases every time, plus one binding to overwrite later."""
    aliases = F.normalize(torch.randn(batch, n_bindings, 32, generator=gen), dim=-1)
    tools = torch.randint(0, n_tools, (batch, n_bindings), generator=gen)

    index = torch.randint(0, n_bindings, (batch,), generator=gen)
    old = tools[torch.arange(batch), index]
    # Draw a replacement uniformly from the tools that are not the current one.
    new = torch.randint(0, n_tools - 1, (batch,), generator=gen)
    new = new + (new >= old).long()

    return (
        aliases.to(device),
        tools.to(device),
        index.to(device),
        new.to(device),
    )


def run_episode(model: MiniRouter, aliases, tools, index, new):
    """Bind everything, query, overwrite one binding, query again."""
    batch = aliases.shape[0]
    rows = torch.arange(batch, device=aliases.device)

    state = model.blank_state(batch, aliases.device, aliases.dtype)
    state = model.bind(state, aliases, tools)
    logits_before = model.query(state, aliases)

    state = model.bind(state, aliases[rows, index].unsqueeze(1), new.unsqueeze(1))
    targets = tools.clone()
    targets[rows, index] = new
    logits_after = model.query(state, aliases)

    return logits_before, logits_after, targets


@torch.no_grad()
def show_episode(model: MiniRouter, episode, n_tools: int) -> None:
    """Print one episode end to end, in words, for the first batch row."""
    aliases, tools, index, new = episode
    before, after, targets = run_episode(model, *episode)
    name = lambda i: TOOL_NAMES[i]  # noqa: E731

    print("\none episode, batch row 0 (aliases are fresh random 32-d vectors):\n")
    for k in range(aliases.shape[1]):
        print(f"    bind       A{k} -> {name(tools[0, k])}")

    print()
    for k in range(aliases.shape[1]):
        got = before[0, k].argmax().item()
        mark = "ok " if got == tools[0, k] else "MISS"
        print(f"    query      A{k} ?  {name(got):<11} {mark}")

    j = index[0].item()
    print(f"\n    overwrite  A{j} -> {name(new[0])}   (was {name(tools[0, j])})\n")

    for k in range(aliases.shape[1]):
        got = after[0, k].argmax().item()
        want = targets[0, k].item()
        mark = "ok " if got == want else "MISS"
        tag = "  <- the overwritten one" if k == j else ""
        print(f"    query      A{k} ?  {name(got):<11} {mark}{tag}")

    print(
        "\n    the memory M is thrown away here; the next episode draws"
        "\n    entirely new aliases, so nothing above can be memorized.\n"
    )


# ---------------------------------------------------------------------------
# Parity with the production operator
# ---------------------------------------------------------------------------


def check_against_trellis_lm(device: str) -> None:
    """Assert this file's scan matches trellis_lm/trellis_memory.py exactly."""
    from trellis_lm.activations import get_activation
    from trellis_lm.trellis_memory import run_trellis_memory

    torch.manual_seed(0)
    B, H, T, D, M = 2, 2, 6, 8, 5
    write = torch.randn(B, H, T, D, device=device, dtype=torch.float64)
    read = torch.randn(B, H, T, D, device=device, dtype=torch.float64)
    alpha = torch.randn(B, H, T, M, device=device, dtype=torch.float64)
    beta = torch.rand(B, H, T, 1, device=device, dtype=torch.float64) * 0.02 + 0.98
    gamma = torch.rand(H, device=device, dtype=torch.float64) * 0.05
    M0 = torch.zeros(B, H, M, D, device=device, dtype=torch.float64)

    for phi in ("identity", "silu"):
        mine_state, mine_read = trellis_scan(
            write, alpha, beta, gamma, phi, M0, read=read
        )
        theirs_read, theirs_state = run_trellis_memory(
            write=write,
            read=read,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=get_activation(phi),
            read_mode="M_q",
            training=False,
            exact_inner=True,
            M_init=M0,
            return_state=True,
        )
        assert torch.allclose(mine_state, theirs_state, atol=1e-12), phi
        assert torch.allclose(mine_read, theirs_read, atol=1e-12), phi
    print("parity vs trellis_lm.run_trellis_memory: OK (identity, silu, fp64)")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--phi", choices=("silu", "identity"), default="silu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--show-episode",
        action="store_true",
        help="print one episode in words before and after training",
    )
    args = parser.parse_args()

    n_tools, n_bindings, batch = 6, 4, 128
    check_against_trellis_lm(args.device)

    torch.manual_seed(0)
    model = MiniRouter(args.phi, n_tools=n_tools).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)

    print(f"write phi: {args.phi}   tools: {n_tools} (chance {1/n_tools:.3f})")

    if args.show_episode:
        demo = make_episode(batch, n_bindings, n_tools, torch.Generator(), args.device)
        print("\n--- before training: the protocol has not been learned yet ---")
        show_episode(model, demo, n_tools)

    print("step    loss    all-correct  overwritten-correct")

    for step in range(1, args.steps + 1):
        episode = make_episode(batch, n_bindings, n_tools, gen, args.device)
        aliases, tools, index, new = episode
        before, after, targets = run_episode(model, *episode)

        # Weight the overwritten slot up: it is one of K bindings but it is the
        # whole point of the probe.
        loss = F.cross_entropy(
            before.reshape(-1, n_tools), tools.reshape(-1)
        ) + F.cross_entropy(after.reshape(-1, n_tools), targets.reshape(-1))
        rows = torch.arange(batch, device=args.device)
        loss = loss + 1.5 * F.cross_entropy(after[rows, index], new)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                correct = (after.argmax(-1) == targets).float()
                print(
                    f"{step:5d}  {loss.item():6.3f}   {correct.mean():.3f}"
                    f"        {correct[rows, index].mean():.3f}"
                )

    if args.show_episode:
        print("\n--- after training: same episode, same frozen weights ---")
        model.eval()
        show_episode(model, demo, n_tools)


if __name__ == "__main__":
    main()
