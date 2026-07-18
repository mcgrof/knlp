#!/usr/bin/env python3
"""Dump each LongHealth patient's medical record to a text file so KVFromText can do
CAS-style cartridge-specific truncation init (each cartridge starts from its own document,
not a shared gradient.txt). Writes /root/cart_records/<patient>.txt."""
import os
os.environ.setdefault("CARTRIDGES_DIR", os.environ.get("CARTRIDGES_DIR", "/root/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cart_out")
from pathlib import Path
from cartridges.data.longhealth.utils import load_longhealth_dataset

OUT = Path(os.environ.get("RECORDS_DIR", "/root/cart_records")); OUT.mkdir(parents=True, exist_ok=True)
pts = load_longhealth_dataset(None)
TMPL = ("Below is patient {name}'s medical record (ID: {pid}). Born {bd}. "
        "Diagnosis: {dx}. The record consists of {n} notes.\n{notes}")
for p in pts:
    notes = "\n".join(f"<{nid}>\n{txt}\n</{nid}>" for nid, txt in p.texts.items())
    txt = TMPL.format(name=p.name, pid=p.patient_id, bd=p.birthday, dx=p.diagnosis,
                      n=len(p.texts), notes=notes)
    (OUT / f"{p.patient_id}.txt").write_text(txt)
    print(f"{p.patient_id}: {len(txt)} chars, {len(p.texts)} notes")
print("DONE")
