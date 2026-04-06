#!/usr/bin/env python3
"""Generate embeddings for text using sentence-transformers.

Usage:
    echo "some text" | python3 scripts/embed.py
    python3 scripts/embed.py --text "some text"
    python3 scripts/embed.py --batch input.jsonl > output.jsonl

Output: JSON array of floats (the embedding vector)

Requires: pip install sentence-transformers
Falls back to: simple TF-IDF hash if sentence-transformers not available
"""

import sys
import json
import hashlib


def tfidf_hash_embedding(text, dim=64):
    """Fallback: deterministic hash-based pseudo-embedding.

    Not semantic, but gives consistent vectors for deduplication
    and basic similarity. Better than nothing when sentence-transformers
    is not installed.
    """
    words = text.lower().split()
    vec = [0.0] * dim
    for word in words:
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        for i in range(dim):
            bit = (h >> i) & 1
            vec[i] += 1.0 if bit else -1.0
    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def get_embedder():
    """Try to load sentence-transformers, fall back to hash."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda text: model.encode(text).tolist()
    except Exception:
        return lambda text: tfidf_hash_embedding(text)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate text embeddings for similarity search"
    )
    parser.add_argument("--text", help="Text to embed")
    parser.add_argument("--batch", help="JSONL file with 'text' field per line")
    args = parser.parse_args()

    embed = get_embedder()

    if args.text:
        print(json.dumps(embed(args.text)))
    elif args.batch:
        with open(args.batch) as f:
            for line in f:
                item = json.loads(line)
                vec = embed(item["text"])
                item["embedding"] = vec
                print(json.dumps(item))
    else:
        # Read from stdin
        text = sys.stdin.read().strip()
        if text:
            print(json.dumps(embed(text)))


if __name__ == "__main__":
    main()
