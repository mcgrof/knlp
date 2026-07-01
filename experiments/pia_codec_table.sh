#!/bin/bash
# Expand the PIA codec table with asymmetric KV quantization -- byte-aware, with
# fair (per-channel-K / per-token-V) scaling, a naive per-tensor stress control,
# and byte-matched prefix-safe positional selectors as the Pareto comparator.
#
# Every codec is query-independent and keeps all blocks, so on the CONTRACT axis
# it is SAFE_ONLY_WITH_CUSTOM_CONNECTOR (needs a codec-aware dtype connector);
# the positional selectors are SAFE_FOR_PREFIX_OFFLOAD but drop whole blocks. The
# point of the table is the DRIFT-vs-BYTES trade at fixed contract safety.
set -e
cd /root/pia
export PYTHONPATH=/root/pia
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null || true)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=Qwen/Qwen2.5-7B-Instruct
CART=/root/pia/cart
Q=/root/pia/queries_ext.jsonl
OUT=/root/pia/out
mkdir -p "$OUT"

if [ ! -f "$CART/cartridge.pt" ]; then
  echo "### building cartridge"
  python experiments/pia_build_cartridge.py --model "$MODEL" \
    --target-tokens 4096 --block-size 16 --out-dir "$CART"
fi

# 48 citation queries matched to the cartridge topics (drift needs variety
# across requests, not a labeled task).
python - "$Q" <<'PY'
import json, sys
topics = ["attention mechanisms","the transformer architecture","layer normalization",
    "residual learning","batch normalization","adaptive optimization",
    "dropout regularization","word embeddings","sequence to sequence learning",
    "neural machine translation","language model pretraining","mixture of experts",
    "sparse attention","rotary position embeddings","state space models",
    "retrieval augmentation","knowledge distillation","quantized inference",
    "key-value cache compression","long-context modeling"]
forms = ["Which reference should I cite for {t}?",
         "What is the BibTeX key for the paper on {t}?",
         "In what year was the {t} paper published?"]
qs=[]
i=0
for t in topics:
    for f in forms:
        qs.append({"id":f"q{i}","query":f.format(t=t)}); i+=1
        if len(qs)>=48: break
    if len(qs)>=48: break
open(sys.argv[1],"w").write("\n".join(json.dumps(q) for q in qs))
print(f"wrote {len(qs)} queries")
PY

# codec rows (query-independent, all blocks present)
CODECS="none int8 fp8 k8v16_pt k8v16 k16v8 k8v8 k16v4"
for c in $CODECS; do
  echo "### drift codec=$c"
  python -m routing.prefix_integrity.semantic_drift \
    --model "$MODEL" --cartridge "$CART" --queries "$Q" \
    --algorithm full --mode codec --codec "$c" --budget-k 256 --pins A1R2 \
    --out "$OUT/sem_codec_$c.json"
done

# byte-matched prefix-safe positional selectors (anchor+recency, query-independent).
# 256 blocks total; keep 192=75% (matches k16v8), 161=63% (k16v4), 128=50% (k8v8).
for kb in 192 161 128; do
  echo "### drift selector anchor_recency K=$kb"
  python -m routing.prefix_integrity.semantic_drift \
    --model "$MODEL" --cartridge "$CART" --queries "$Q" \
    --algorithm anchor_recency --mode selector --budget-k "$kb" --pins A1R2 \
    --out "$OUT/sem_sel_$kb.json"
done

python - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
HD = 128  # head_dim for scale overhead accounting
# codec -> (K bits, V bits, scale-overhead bits/elem summed, granularity, contract)
def frac(kb, vb, ov):  # fraction of bf16 (32 b/pair-elem) including scale meta
    return (kb + vb + ov) / 32.0
CODEC = {
  "none":     (16,16, 0.0,  "bf16 reload control", "SAFE_w_CONNECTOR"),
  "int8":     (8, 8,  0.0,  "per-tensor (naive stress)", "SAFE_w_CONNECTOR"),
  "fp8":      (8, 8,  0.0,  "per-tensor fp8 (naive stress)", "SAFE_w_CONNECTOR"),
  "k8v16_pt": (8, 16, 0.0,  "K per-tensor (naive-K stress)", "SAFE_w_CONNECTOR"),
  "k8v16":    (8, 16, 16.0/64, "K per-channel (fair)", "SAFE_w_CONNECTOR"),
  "k16v8":    (16, 8, 16.0/HD, "V per-token (fair)", "SAFE_w_CONNECTOR"),
  "k8v8":     (8, 8,  16.0/64+16.0/HD, "K per-chan + V per-tok (fair)", "SAFE_w_CONNECTOR"),
  "k16v4":    (16, 4, 16.0/HD, "V per-token int4 (fair)", "SAFE_w_CONNECTOR"),
}
rows=[]
for f in glob.glob(os.path.join(out,"sem_codec_*.json")):
    d=json.load(open(f)); c=d["detail"]["codec"]
    kb,vb,ov,gran,con = CODEC.get(c,(0,0,0,"?","?"))
    rows.append(("codec:"+c, round(frac(kb,vb,ov),3), gran, con,
                 d["kl"], d["detail"]["kl_max"], d["top1"]))
for f in glob.glob(os.path.join(out,"sem_sel_*.json")):
    d=json.load(open(f)); kb=int(os.path.basename(f).split("_")[2].split(".")[0])
    rows.append((f"selector:anchor_recency_K{kb}", round(kb/256.0,3),
                 "drops whole blocks", "SAFE_FOR_PREFIX_OFFLOAD",
                 d["kl"], d["detail"]["kl_max"], d["top1"]))
rows.sort(key=lambda r:(r[0].startswith("selector"), r[1]))
with open(os.path.join(out,"codec_table.csv"),"w") as fh:
    fh.write("method,bytes_frac_vs_bf16,granularity,contract,kl_mean,kl_max,top1\n")
    for m,bf,g,con,kl,klm,t1 in rows:
        fh.write(f"{m},{bf},{g},{con},{kl:.5f},{klm:.5f},{t1:.3f}\n")
        print(f"{m:34s} bytes={bf:.3f} {g:32s} kl={kl:.4f} kl_max={klm:.4f} top1={t1:.3f}")
print("wrote", os.path.join(out,"codec_table.csv"))
PY
echo "DONE"
