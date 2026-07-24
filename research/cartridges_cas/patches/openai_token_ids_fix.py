#!/usr/bin/env python3
"""Patch cartridges/clients/openai.py so vLLM logprobs carry real token ids.

The client filled top_logprobs.token_ids with -1 (OpenAI logprobs return token
strings, not ids, and the client discarded the strings). Fix: request vLLM's
return_tokens_as_token_ids (tokens come back as "token_id:<int>") and parse the
ids into both the sampled token_ids and the top-logprobs id matrix.
"""
import sys

P = "/home/mcgrof/cartridges/cartridges/clients/openai.py"
src = open(P).read()

# 1. request token-ids-as-tokens from vLLM (non-openai endpoints only)
old1 = (
    "        if modal_upstream_id is not None and self.type != \"openai\":\n"
    "            extra_body[\"modal_upstream_id\"] = modal_upstream_id\n"
)
new1 = old1 + (
    "\n"
    "        # vLLM returns logprob tokens as \"token_id:<int>\" so we recover exact\n"
    "        # vocab ids for distillation (the OpenAI logprobs API has no id field).\n"
    "        if self.type != \"openai\":\n"
    "            extra_body[\"return_tokens_as_token_ids\"] = True\n"
)
assert src.count(old1) == 1, f"anchor1 not found uniquely ({src.count(old1)})"
src = src.replace(old1, new1)

# 2. parse ids in the logprobs loop
old2 = (
    "            if choice.logprobs and choice.logprobs.content:\n"
    "                # For now, we don't have token IDs from OpenAI API, but we can extract logprobs\n"
    "                \n"
    "                # Create logprobs matrix (simplified version)\n"
    "                logprobs_list = []\n"
    "                for token in choice.logprobs.content:\n"
    "                    token_logprobs = [token.logprob]\n"
    "                    if token.top_logprobs:\n"
    "                        token_logprobs.extend([t.logprob for t in token.top_logprobs])\n"
    "                    logprobs_list.append(token_logprobs)\n"
)
new2 = (
    "            if choice.logprobs and choice.logprobs.content:\n"
    "                def _tid(tok):\n"
    "                    s = getattr(tok, \"token\", \"\") or \"\"\n"
    "                    if s.startswith(\"token_id:\"):\n"
    "                        try:\n"
    "                            return int(s.split(\":\", 1)[1])\n"
    "                        except ValueError:\n"
    "                            return -1\n"
    "                    tid = self.tokenizer.convert_tokens_to_ids(s) if hasattr(\n"
    "                        self.tokenizer, \"convert_tokens_to_ids\") else -1\n"
    "                    return tid if isinstance(tid, int) and tid >= 0 else -1\n"
    "\n"
    "                logprobs_list = []\n"
    "                ids_list = []\n"
    "                sampled_ids = []\n"
    "                for token in choice.logprobs.content:\n"
    "                    token_logprobs = [token.logprob]\n"
    "                    row_ids = [_tid(token)]\n"
    "                    sampled_ids.append(_tid(token))\n"
    "                    if token.top_logprobs:\n"
    "                        token_logprobs.extend([t.logprob for t in token.top_logprobs])\n"
    "                        row_ids.extend([_tid(t) for t in token.top_logprobs])\n"
    "                    logprobs_list.append(token_logprobs)\n"
    "                    ids_list.append(row_ids)\n"
)
assert src.count(old2) == 1, f"anchor2 not found uniquely ({src.count(old2)})"
src = src.replace(old2, new2)

# 3. build the id-padded matrix and use it (drop the -1 fill)
old3 = (
    "                if logprobs_list:\n"
    "                    # Pad all rows to same length\n"
    "                    max_len = max(len(row) for row in logprobs_list)\n"
    "                    padded_logprobs = []\n"
    "                    for row in logprobs_list:\n"
    "                        padded_row = row + [-1000.0] * (max_len - len(row))\n"
    "                        padded_logprobs.append(padded_row)\n"
    "                    \n"
    "                    # Create TopLogprobs object - simplified since we don't have token IDs\n"
    "                    top_logprobs = TopLogprobs(\n"
    "                        logprobs=np.array(padded_logprobs, dtype=np.float32),\n"
    "                        token_ids=np.full((len(padded_logprobs), max_len), -1, dtype=np.int32)\n"
    "                    )\n"
)
new3 = (
    "                if logprobs_list:\n"
    "                    max_len = max(len(row) for row in logprobs_list)\n"
    "                    padded_logprobs = [row + [-1000.0] * (max_len - len(row))\n"
    "                                       for row in logprobs_list]\n"
    "                    padded_ids = [row + [-1] * (max_len - len(row))\n"
    "                                  for row in ids_list]\n"
    "                    token_ids = np.array(sampled_ids, dtype=np.int32)\n"
    "                    top_logprobs = TopLogprobs(\n"
    "                        logprobs=np.array(padded_logprobs, dtype=np.float32),\n"
    "                        token_ids=np.array(padded_ids, dtype=np.int32),\n"
    "                    )\n"
)
assert src.count(old3) == 1, f"anchor3 not found uniquely ({src.count(old3)})"
src = src.replace(old3, new3)

open(P, "w").write(src)
print("PATCH_OK: openai.py patched (return_tokens_as_token_ids + id parsing)")
