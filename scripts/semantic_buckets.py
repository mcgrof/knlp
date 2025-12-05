#!/usr/bin/env python3
"""
Semantic Bucketing for KV Cache Compression.

Assigns text spans to semantic buckets for specialized compression.

Buckets:
- narrative: Story-like prose, descriptions
- dialogue: Conversational text, Q&A
- code: Programming code
- math: Mathematical expressions, equations
- reasoning: Step-by-step logic, explanations
- instructions: Commands, how-to content

Usage:
    python scripts/semantic_buckets.py --text "Your text here"
    python scripts/semantic_buckets.py --file input.txt --output buckets.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


BUCKET_NAMES = ["narrative", "dialogue", "code", "math", "reasoning", "instructions"]


@dataclass
class TextSpan:
    """A span of text with its semantic bucket."""

    text: str
    start: int
    end: int
    bucket: str
    confidence: float


class RuleBasedBucketer:
    """
    Rule-based semantic bucketing using pattern matching.

    Fast and deterministic, good baseline before ML-based approaches.
    """

    def __init__(self):
        # Code patterns
        self.code_patterns = [
            r"```[\s\S]*?```",  # Markdown code blocks
            r"def\s+\w+\s*\(",  # Python functions
            r"class\s+\w+\s*[:\(]",  # Python classes
            r"import\s+\w+",  # Import statements
            r"from\s+\w+\s+import",
            r"function\s+\w+\s*\(",  # JS functions
            r"const\s+\w+\s*=",  # JS const
            r"let\s+\w+\s*=",
            r"var\s+\w+\s*=",
            r"\w+\s*=\s*\w+\(",  # Assignment with function call
            r"if\s*\(.+\)\s*\{",  # C-style if
            r"for\s*\(.+\)\s*\{",  # C-style for
            r"#include\s*<",  # C includes
            r"public\s+class\s+\w+",  # Java class
        ]

        # Math patterns
        self.math_patterns = [
            r"\$\$.+?\$\$",  # LaTeX display math
            r"\$.+?\$",  # LaTeX inline math
            r"\\frac\{",  # LaTeX fraction
            r"\\sum",  # LaTeX sum
            r"\\int",  # LaTeX integral
            r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Arithmetic
            r"=\s*\d+",  # Equals number
            r"\d+\s*%",  # Percentage
            r"[xyz]\s*=\s*\d",  # Variable assignment
            r"\d+\s*\+\s*\d+\s*=",  # Addition equation
        ]

        # Dialogue patterns
        self.dialogue_patterns = [
            r'"[^"]+"\s*,?\s*(he|she|they|I)\s+(said|asked|replied)',
            r"\"[^\"]+\"",  # Quoted speech
            r"'[^']+'",  # Single quoted speech
            r"Q:\s*.+\nA:\s*",  # Q&A format
            r"User:\s*",  # Chat format
            r"Assistant:\s*",
            r"Human:\s*",
            r"\?\s*$",  # Questions
        ]

        # Instructions patterns
        self.instruction_patterns = [
            r"^\d+\.\s+",  # Numbered list
            r"^-\s+",  # Bullet list
            r"^•\s+",
            r"^Step\s+\d+",  # Step instructions
            r"First,\s+",  # Sequential instructions
            r"Then,\s+",
            r"Finally,\s+",
            r"^Note:\s+",
            r"^Warning:\s+",
            r"^Important:\s+",
            r"(click|press|type|enter|select|choose)\s+",  # UI instructions
        ]

        # Reasoning patterns
        self.reasoning_patterns = [
            r"because\s+",
            r"therefore\s+",
            r"thus\s+",
            r"hence\s+",
            r"so\s+that\s+",
            r"in order to\s+",
            r"if\s+.+\s+then\s+",
            r"let's\s+(think|consider|analyze)",
            r"this\s+(means|implies|shows)\s+",
            r"we\s+can\s+(conclude|infer|deduce)",
        ]

    def classify_span(self, text: str) -> Tuple[str, float]:
        """
        Classify a text span into a semantic bucket.

        Returns (bucket_name, confidence).
        """
        text_lower = text.lower()

        # Score each bucket
        scores = {bucket: 0.0 for bucket in BUCKET_NAMES}

        # Code detection (highest priority for code blocks)
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                scores["code"] += 2.0

        # Math detection
        for pattern in self.math_patterns:
            if re.search(pattern, text):
                scores["math"] += 1.5

        # Dialogue detection
        for pattern in self.dialogue_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                scores["dialogue"] += 1.0

        # Instructions detection
        for pattern in self.instruction_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                scores["instructions"] += 1.0

        # Reasoning detection
        for pattern in self.reasoning_patterns:
            if re.search(pattern, text_lower):
                scores["reasoning"] += 1.0

        # Default to narrative if no strong signals
        max_score = max(scores.values())
        if max_score < 0.5:
            return "narrative", 0.6

        # Return highest scoring bucket
        best_bucket = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_bucket] / 3.0)

        return best_bucket, confidence

    def bucket_text(self, text: str, chunk_size: int = 200) -> List[TextSpan]:
        """
        Bucket text into semantic spans.

        Splits text into chunks and classifies each.
        """
        spans = []

        # Split into sentences or chunks
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Classify current chunk
                bucket, confidence = self.classify_span(current_chunk)
                spans.append(
                    TextSpan(
                        text=current_chunk,
                        start=current_start,
                        end=current_start + len(current_chunk),
                        bucket=bucket,
                        confidence=confidence,
                    )
                )
                current_start += len(current_chunk) + 1
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Handle remaining chunk
        if current_chunk:
            bucket, confidence = self.classify_span(current_chunk)
            spans.append(
                TextSpan(
                    text=current_chunk,
                    start=current_start,
                    end=current_start + len(current_chunk),
                    bucket=bucket,
                    confidence=confidence,
                )
            )

        return spans


class EmbeddingBucketer:
    """
    Embedding-based semantic bucketing using sentence transformers.

    More accurate but requires model loading.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.bucket_embeddings = None

    def _load_model(self):
        """Lazy load sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)

                # Create reference embeddings for each bucket
                bucket_descriptions = {
                    "narrative": "A story with characters, plot, and descriptive prose.",
                    "dialogue": "A conversation between people with questions and answers.",
                    "code": "Programming code with functions, variables, and syntax.",
                    "math": "Mathematical equations, formulas, and calculations.",
                    "reasoning": "Logical analysis with because, therefore, and conclusions.",
                    "instructions": "Step-by-step instructions and how-to guides.",
                }

                self.bucket_embeddings = {}
                for bucket, desc in bucket_descriptions.items():
                    self.bucket_embeddings[bucket] = self.model.encode(
                        desc, convert_to_tensor=True
                    )

            except ImportError:
                print("Warning: sentence-transformers not installed, using rule-based")
                return False
        return True

    def classify_span(self, text: str) -> Tuple[str, float]:
        """Classify text using embedding similarity."""
        if not self._load_model():
            # Fallback to rule-based
            return RuleBasedBucketer().classify_span(text)

        # Encode text
        text_embedding = self.model.encode(text, convert_to_tensor=True)

        # Find most similar bucket
        best_bucket = "narrative"
        best_score = -1.0

        for bucket, bucket_emb in self.bucket_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0), bucket_emb.unsqueeze(0)
            ).item()

            if similarity > best_score:
                best_score = similarity
                best_bucket = bucket

        # Normalize confidence
        confidence = (best_score + 1) / 2  # Map [-1, 1] to [0, 1]

        return best_bucket, confidence

    def bucket_text(self, text: str, chunk_size: int = 200) -> List[TextSpan]:
        """Bucket text into semantic spans using embeddings."""
        spans = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                bucket, confidence = self.classify_span(current_chunk)
                spans.append(
                    TextSpan(
                        text=current_chunk,
                        start=current_start,
                        end=current_start + len(current_chunk),
                        bucket=bucket,
                        confidence=confidence,
                    )
                )
                current_start += len(current_chunk) + 1
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            bucket, confidence = self.classify_span(current_chunk)
            spans.append(
                TextSpan(
                    text=current_chunk,
                    start=current_start,
                    end=current_start + len(current_chunk),
                    bucket=bucket,
                    confidence=confidence,
                )
            )

        return spans


def get_bucketer(method: str = "rule") -> "RuleBasedBucketer | EmbeddingBucketer":
    """Get a bucketer instance."""
    if method == "embedding":
        return EmbeddingBucketer()
    return RuleBasedBucketer()


def bucket_tokens(
    text: str,
    tokenizer,
    method: str = "rule",
) -> Dict[str, List[int]]:
    """
    Bucket tokens by semantic category.

    Returns dict mapping bucket names to lists of token indices.
    """
    bucketer = get_bucketer(method)
    spans = bucketer.bucket_text(text)

    # Tokenize full text
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.input_ids
    offsets = encoding.offset_mapping

    # Map tokens to buckets
    bucket_tokens = {bucket: [] for bucket in BUCKET_NAMES}

    for token_idx, (start, end) in enumerate(offsets):
        # Find which span this token belongs to
        for span in spans:
            if start >= span.start and end <= span.end:
                bucket_tokens[span.bucket].append(token_idx)
                break
        else:
            # Default to narrative if no span found
            bucket_tokens["narrative"].append(token_idx)

    return bucket_tokens


def main():
    parser = argparse.ArgumentParser(description="Semantic bucketing for text")
    parser.add_argument("--text", type=str, help="Text to bucket")
    parser.add_argument("--file", type=str, help="File to bucket")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument(
        "--method",
        type=str,
        default="rule",
        choices=["rule", "embedding"],
        help="Bucketing method",
    )
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    args = parser.parse_args()

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        # Demo text
        text = """
        Once upon a time, there was a programmer named Alice.

        "Hello, how are you?" she asked.
        "I'm fine, thanks!" replied Bob.

        def hello_world():
            print("Hello, World!")

        If x = 5 and y = 3, then x + y = 8.

        Let's think step by step. First, we need to consider the problem.
        Therefore, we can conclude that the answer is 42.

        Step 1: Open the terminal.
        Step 2: Type the command.
        Step 3: Press Enter.
        """

    # Bucket text
    bucketer = get_bucketer(args.method)
    spans = bucketer.bucket_text(text, args.chunk_size)

    # Print results
    print(f"Semantic Bucketing ({args.method} method)")
    print("=" * 60)

    bucket_counts = {bucket: 0 for bucket in BUCKET_NAMES}
    for span in spans:
        bucket_counts[span.bucket] += 1
        print(f"\n[{span.bucket.upper()}] (conf={span.confidence:.2f})")
        print(f"  {span.text[:100]}...")

    print("\n" + "=" * 60)
    print("Summary:")
    for bucket, count in bucket_counts.items():
        print(f"  {bucket}: {count} spans")

    # Save output
    if args.output:
        output_data = {
            "method": args.method,
            "spans": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "bucket": s.bucket,
                    "confidence": s.confidence,
                }
                for s in spans
            ],
            "summary": bucket_counts,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
