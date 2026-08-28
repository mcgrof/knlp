#!/usr/bin/env python3
"""Patch cartridges/clients/openai.py so vLLM logprobs carry real token ids,
and so each target row is a distribution rather than a double-counted one.

Two things are wrong with the stock client for our purposes.

First, it filled top_logprobs.token_ids with -1, because the OpenAI logprobs
API returns token strings rather than ids and the client discarded the
strings.  That makes the path unusable for distillation.  We fix it by
requesting vLLM's return_tokens_as_token_ids (tokens come back as
"token_id:<int>") and parsing the ids.

Second -- and this is the part an earlier version of this patch got wrong --
the stock row layout is [sampled token] followed by [top-k tokens].  Upstream
could get away with that because its ids were all -1 and nothing trained on
them.  Once the ids are real the layout is a live defect: synthesis runs
greedily, so the sampled token IS the top-1 token, and every row then carries
it twice with equal probability.  The trainer sums over entries, so that token
is counted twice -- an accidental confidence-weighted hard-label term on top
of the intended soft-target loss -- and the cumulative-mass truncation sees
2*p and stops early, discarding alternatives it was supposed to keep.

So this patch writes a canonical row: the sampled token id is kept in its own
field, the distribution is the top-k list deduplicated by id, and the sampled
token is unioned in exactly once if the server did not return it among the
top-k (which happens only under non-greedy sampling).  Existing parquets are
NOT rewritten -- they are the inputs behind published results and stay
readable through the explicit legacy transform in control_aware/targets.py --
but nothing generated from here on carries the duplicate.
"""

import sys

P = "/home/mcgrof/cartridges/cartridges/clients/openai.py"
src = open(P).read()

# 1. request token-ids-as-tokens from vLLM (non-openai endpoints only)
old1 = (
    '        if modal_upstream_id is not None and self.type != "openai":\n'
    '            extra_body["modal_upstream_id"] = modal_upstream_id\n'
)
new1 = old1 + (
    "\n"
    '        # vLLM returns logprob tokens as "token_id:<int>" so we recover exact\n'
    "        # vocab ids for distillation (the OpenAI logprobs API has no id field).\n"
    '        if self.type != "openai":\n'
    '            extra_body["return_tokens_as_token_ids"] = True\n'
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
    '                    s = getattr(tok, "token", "") or ""\n'
    '                    if s.startswith("token_id:"):\n'
    "                        try:\n"
    '                            return int(s.split(":", 1)[1])\n'
    "                        except ValueError:\n"
    "                            return -1\n"
    "                    tid = self.tokenizer.convert_tokens_to_ids(s) if hasattr(\n"
    '                        self.tokenizer, "convert_tokens_to_ids") else -1\n'
    "                    return tid if isinstance(tid, int) and tid >= 0 else -1\n"
    "\n"
    "                logprobs_list = []\n"
    "                ids_list = []\n"
    "                sampled_ids = []\n"
    "                for token in choice.logprobs.content:\n"
    "                    sid = _tid(token)\n"
    "                    sampled_ids.append(sid)\n"
    "                    # Canonical row: the top-k list deduplicated by id.\n"
    "                    # Do NOT prepend the sampled token -- under greedy\n"
    "                    # synthesis it is already top-1, and prepending it\n"
    "                    # makes the trainer count it twice.\n"
    "                    token_logprobs = []\n"
    "                    row_ids = []\n"
    "                    seen = set()\n"
    "                    for t in (token.top_logprobs or []):\n"
    "                        tid = _tid(t)\n"
    "                        if tid in seen:\n"
    "                            continue\n"
    "                        seen.add(tid)\n"
    "                        row_ids.append(tid)\n"
    "                        token_logprobs.append(t.logprob)\n"
    "                    # union the sampled token once if the server did not\n"
    "                    # return it among the top-k (non-greedy sampling)\n"
    "                    if sid not in seen:\n"
    "                        row_ids.insert(0, sid)\n"
    "                        token_logprobs.insert(0, token.logprob)\n"
    "                    assert len(set(row_ids)) == len(row_ids), (\n"
    "                        'duplicate token id in a target row: the row is not '\n"
    "                        'a distribution and the trainer would count it twice'\n"
    "                    )\n"
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
