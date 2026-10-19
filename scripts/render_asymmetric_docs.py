#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the asymmetric installation and methods references for the static site.

Run with Python-Markdown 3.9: uv run --no-project --with markdown==3.9
scripts/render_asymmetric_docs.py
"""

from pathlib import Path
import re

import markdown

ROOT = Path(__file__).resolve().parents[1] / "docs"
for name, title in [
    ("asymmetric-install", "Install asymmetric KV"),
    ("asymmetric-methods", "Asymmetric KV methods and results"),
]:
    source = (ROOT / (name + ".md")).read_text()
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"])
    body = re.sub(r'href="(asymmetric-(?:install|methods))\.md', r'href="\1.html', body)
    # Name each scrollable table from its nearest section heading.
    section = title
    pieces = re.split(r"(<h[12]\b[^>]*>.*?</h[12]>|<table>)", body)
    for i, piece in enumerate(pieces):
        if re.match(r"<h[12]\b", piece):
            section = re.sub(r"<[^>]+>", "", piece)
        elif piece == "<table>":
            pieces[i] = (
                '<div class="table-wrap" tabindex="0" role="region" '
                f'aria-label="{section}"><table><caption>{section}</caption>'
            )
    body = "".join(pieces)
    body = body.replace("</table>", "</table></div>")
    body = re.sub(r"<th(?=[ >])", '<th scope="col"', body)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — knlp</title>
<link rel="canonical" href="https://knlp.io/{name}.html">
<link rel="stylesheet" href="assets/asymmetric-guide.css">
</head><body><a class="skip" href="#main">Skip to content</a>
<header><a class="brand" href="./">knlp<span> / inference</span></a>
<nav aria-label="Reference navigation"><a href="paper_memory_decode.html">Overview</a><a href="asymmetric-install.html">Install</a><a href="asymmetric-methods.html">Methods</a></nav></header>
<main id="main" class="reference">{body}</main>
<footer><a href="paper_memory_decode.html">Asymmetric KV Cache Quantization</a> · <a href="https://github.com/mcgrof/knlp/blob/main/docs/{name}.md">Markdown source</a></footer>
</body></html>
"""
    (ROOT / (name + ".html")).write_text(page)
