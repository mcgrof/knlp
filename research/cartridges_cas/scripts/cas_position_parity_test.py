#!/usr/bin/env python3
"""Position-parity unit test for the P_iso cartridge geometry.

CAS reports that composed cartridges are subset- and permutation-invariant: a
query attending to one visible cartridge must produce the same logits whether
that cartridge is physically alone or one of N resident with the rest attention-
masked, and independent of physical order. Our joint trainer keeps all N
cartridges physically resident and hides distractors by mask, offsetting query
positions by the VISIBLE cartridge-token count. If "physically absent" and
"present-but-masked" disagree, the rescue result carries a geometry confound.

This test loads a trained cartridge set and, for a real LongHealth query, compares
next-token logits under:
  A  cart_j alone in the cache (1 resident, reveal=[j])
  B  all N carts resident, only cart_j visible (reveal=one-hot(j))
  C  same as B but with the physical cartridge order permuted
All use the same visible-position offset, so they must match to numerical noise.
A pass means masking is equivalent to absence and order does not matter.

Env: CART_DIR, PATIENTS (space-sep, the cart set), KV_TOKENS, DEVICE, SINK_MAX,
N_QUERIES (per cart, default 2), MODEL.
"""

import os
import random

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["CARTRIDGES_COMPILE_FLEX"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset
import cartridges.models.qwen.modeling_qwen3 as mq
from torch.nn.attention.flex_attention import create_block_mask

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CART_DIR = os.environ["CART_DIR"]
PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
KV_TOKENS = int(os.environ.get("KV_TOKENS", "512"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
N_QUERIES = int(os.environ.get("N_QUERIES", "2"))

SYSTEM_PROMPT = (
    "Please reference the patient medical records to answer the user's "
    "questions. Choose the single best option and provide your answer exactly "
    "as it appears in the options.\n\n"
    "Wrap your answer in: <answer> The correct option text here </answer>"
)

# --- parameterized reveal-vector mask (mirrors cas_train_piso) ----------------
_STATE = {"reveal": None, "kvt": KV_TOKENS}


def parity_block_mask(cache, seq_ids, device):
    cache_len = cache.num_cartridge_tokens()
    R = seq_ids.shape[0]
    reveal = _STATE["reveal"].to(device)
    kvt = _STATE["kvt"]
    kv_vis = torch.zeros(cache_len + R, dtype=torch.bool, device=device)
    kv_vis[:cache_len] = reveal.repeat_interleave(kvt).bool()
    cl = cache_len

    def mask_func(_b, _h, q_idx, kv_idx):
        is_cart = kv_idx < cl
        cart_ok = kv_vis[kv_idx]
        req_ok = (q_idx + cl) >= kv_idx
        return torch.where(is_cart, cart_ok, req_ok)

    return create_block_mask(mask_func, B=1, H=1, Q_LEN=R, KV_LEN=R + cl, device=device)


mq.create_block_mask_w_cache = parity_block_mask


def main():
    di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
    torch.cuda.set_device(di)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )

    def load_cart_kv(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)

        def t(p):
            return torch.as_tensor(p.data if hasattr(p, "data") else p).to(
                torch.bfloat16
            )

        fk = ck.get("frozen_keys") or []
        nfrozen = t(fk[0]).shape[2] if fk else 0
        use_frozen = 0 < nfrozen <= SINK_MAX

        def cat(fro, tra):
            tt = [t(p) for p in tra]
            if fro and use_frozen:
                ff = [t(p) for p in fro]
                return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
            return tt

        return (
            cat(ck.get("frozen_keys"), ck["trainable_keys"]),
            cat(ck.get("frozen_values"), ck["trainable_values"]),
        )

    def make_cache(kv_list):
        nl = ac.n_layers
        ik = [
            torch.cat([kv_list[c][0][li] for c in range(len(kv_list))], dim=2)
            for li in range(nl)
        ]
        iv = [
            torch.cat([kv_list[c][1][li] for c in range(len(kv_list))], dim=2)
            for li in range(nl)
        ]
        return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)

    carts = [load_cart_kv(os.path.join(CART_DIR, f"{p}.pt")) for p in PATIENTS]
    N = len(carts)
    kvt = carts[0][0][0].shape[2]  # actual cart token count
    assert all(
        c[0][0].shape[2] == kvt for c in carts
    ), "heterogeneous cart sizes: repeat_interleave masking assumes uniform kvt"
    _STATE["kvt"] = kvt
    print(
        f"[parity] N={N} carts, kv_tokens={kvt} (env KV_TOKENS={KV_TOKENS})", flush=True
    )

    @torch.no_grad()
    def logits_last(cache, ids, reveal, n_resident):
        """Next-token logits at the final input position. Query positions are
        offset by the VISIBLE cart count so all configs put the request at the
        same RoPE offset (the forward adds n_resident*kvt = num_cartridge_tokens)."""
        _STATE["reveal"] = reveal
        R = ids.shape[0]
        visible = int(reveal.sum().item())
        sids = torch.zeros(R, dtype=torch.long, device=DEVICE)
        pos = (
            torch.arange(R, dtype=torch.long, device=DEVICE)
            + visible * kvt
            - n_resident * kvt
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=ids,
                seq_ids=sids,
                position_ids=pos,
                use_cache=True,
                past_key_values=cache,
            )
        return out.logits[0, -1].float()

    def kl(p_logits, q_logits):
        lp = torch.log_softmax(p_logits, -1)
        lq = torch.log_softmax(q_logits, -1)
        return (lp.exp() * (lp - lq)).sum().item()

    patients = {p.patient_id: p for p in load_longhealth_dataset(PATIENTS)}
    rng = random.Random(0)
    worst_ab = worst_ac = worst_noise = 0.0
    kl_ab = kl_ac = kl_noise = 0.0
    argmax_mismatch = 0
    total = 0

    for j, pid in enumerate(PATIENTS):
        patient = patients[pid]
        cache_alone = make_cache([carts[j]])  # config A: cart_j alone
        cache_all = make_cache(carts)  # config B: all N resident
        perm = list(range(N))
        rng.shuffle(perm)
        jp = perm.index(j)  # cart_j's slot after perm
        cache_perm = make_cache([carts[c] for c in perm])  # config C
        # NOISE-FLOOR control matched to B's exact geometry: cart_j visible at
        # slot 0 plus N-1 masked DUPLICATES of cart_j. Identical physical KV
        # length and mask pattern to config B (N resident, 1 visible, N-1
        # masked) -- the ONLY difference from B is that the masked content is
        # cart_j instead of the real distractors. So KL(A||dup) is the pure
        # numerical floor for this geometry, and KL(A||B) ~ KL(A||dup) means the
        # masked distractor content does not leak into attention (true parity);
        # KL(A||B) >> KL(A||dup) would be a real leak/confound.
        cache_dup = make_cache([carts[j]] * N)

        onehot_alone = torch.zeros(1, device=DEVICE)
        onehot_alone[0] = 1.0
        onehot_all = torch.zeros(N, device=DEVICE)
        onehot_all[j] = 1.0
        onehot_perm = torch.zeros(N, device=DEVICE)
        onehot_perm[jp] = 1.0
        onehot_dup = torch.zeros(N, device=DEVICE)
        onehot_dup[0] = 1.0

        for q in patient.questions[:N_QUERIES]:
            body = (
                f"{q.question}\n\nA. {q.answer_a}  B. {q.answer_b}  "
                f"C. {q.answer_c}  D. {q.answer_d}  E. {q.answer_e}"
            )
            ids = (
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": body},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=True,
                )
                .to(DEVICE)
                .flatten()
            )

            la = logits_last(cache_alone, ids, onehot_alone, 1)
            lb = logits_last(cache_all, ids, onehot_all, N)
            lc = logits_last(cache_perm, ids, onehot_perm, N)
            ld = logits_last(cache_dup, ids, onehot_dup, N)  # noise floor

            d_ab = (la - lb).abs().max().item()
            d_ac = (la - lc).abs().max().item()
            d_noise = (la - ld).abs().max().item()
            worst_ab = max(worst_ab, d_ab)
            worst_ac = max(worst_ac, d_ac)
            worst_noise = max(worst_noise, d_noise)
            kl_ab = max(kl_ab, kl(la, lb))
            kl_ac = max(kl_ac, kl(la, lc))
            kl_noise = max(kl_noise, kl(la, ld))
            if not (la.argmax() == lb.argmax() == lc.argmax()):
                argmax_mismatch += 1
            total += 1
            print(
                f"  [{pid} q{total}] max|A-B|={d_ab:.3f} max|A-C|={d_ac:.3f} "
                f"noise|A-dup|={d_noise:.3f}  argmax "
                f"A={int(la.argmax())} B={int(lb.argmax())} C={int(lc.argmax())}",
                flush=True,
            )
        del cache_alone, cache_all, cache_perm, cache_dup
        torch.cuda.empty_cache()

    print(
        f"\n[parity] worst max|A-B|={worst_ab:.3f} max|A-C|={worst_ac:.3f} "
        f"NOISE-FLOOR|A-dup|={worst_noise:.3f}"
    )
    print(
        f"[parity] worst KL(A||B)={kl_ab:.5f} KL(A||C)={kl_ac:.5f} "
        f"KL-noise={kl_noise:.5f}  argmax_mismatch={argmax_mismatch}/{total}"
    )
    # PASS = predictions identical (argmax) AND A-B / A-C differences are within
    # ~2x the pure-numerical noise floor (masked-duplicate control) in both max-
    # logit and KL. A real geometry confound exceeds the floor and flips argmaxes.
    tol = max(2.0 * worst_noise, 0.05)
    kl_tol = max(2.0 * kl_noise, 1e-4)
    ok = (
        argmax_mismatch == 0
        and worst_ab <= tol
        and worst_ac <= tol
        and kl_ab <= kl_tol
        and kl_ac <= kl_tol
    )
    print(
        f"POSITION_PARITY_{'PASS' if ok else 'FAIL'} "
        f"(masked-present {'==' if ok else '!='} physically-absent; "
        f"logit tol={tol:.3f} kl tol={kl_tol:.5f})"
    )


if __name__ == "__main__":
    main()
