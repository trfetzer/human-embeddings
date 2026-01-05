#!/usr/bin/env python3
"""
Chunk -> classify -> embed interview transcripts.

Designed to work with:
- OpenAI hosted API, OR
- a local OpenAI-compatible endpoint (e.g. Ollama /v1 compatibility layer).

Outputs:
- annotated_chunks.parquet
- embedded_chunks.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

from openai import OpenAI


DIMENSIONS = ["project_management", "engineering_craft", "creativity", "technical_depth", "marketing"]


@dataclass
class Config:
    transcripts_dir: Path
    out_dir: Path
    chunk_tokens: int = 500
    chunk_overlap: int = 250
    tokenizer_hint: str = "gpt-4o-mini"

    classifier_model: str = os.environ.get("CLASSIFIER_MODEL", os.environ.get("OPENAI_MODEL_RESPONSES", "gpt-4o-mini"))
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large"))

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL")  # e.g. http://localhost:11434/v1


CLASSIFIER_SYSTEM = """You label interview snippets into one predominant dimension.

Return STRICT JSON with keys:
- label: one of {dims}
- confidence: float in [0,1]
- rationale: short string

Do not include any other keys or surrounding text.
""".strip()


CLASSIFIER_USER_TEMPLATE = """Snippet:
\"\"\"{snippet}\"\"\"

Choose ONE label from: {dims}.
Return strict JSON only.
""".strip()


def _get_encoder(tokenizer_hint: str):
    if not _HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.encoding_for_model(tokenizer_hint)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_tokens: int, overlap: int, tokenizer_hint: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    enc = _get_encoder(tokenizer_hint)
    if enc is None:
        # Fallback: rough approximation by words
        words = text.split()
        approx_tokens_per_word = 1.3
        chunk_words = max(50, int(chunk_tokens / approx_tokens_per_word))
        overlap_words = max(0, int(overlap / approx_tokens_per_word))
        out = []
        i = 0
        while i < len(words):
            j = min(len(words), i + chunk_words)
            out.append(" ".join(words[i:j]))
            i = j - overlap_words
            if i < 0:
                i = 0
            if j == len(words):
                break
        return out

    toks = enc.encode(text)
    out = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + chunk_tokens)
        out.append(enc.decode(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return out


def load_transcripts(transcripts_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(transcripts_dir / "*.txt")))
    rows = []
    for fp in files:
        p = Path(fp)
        rows.append({"file": p.name, "path": str(p), "text": p.read_text(encoding="utf-8", errors="ignore")})
    return pd.DataFrame(rows)


def make_client(cfg: Config) -> OpenAI:
    if cfg.openai_base_url:
        return OpenAI(api_key=cfg.openai_api_key or "local", base_url=cfg.openai_base_url)
    return OpenAI(api_key=cfg.openai_api_key)


def classify_snippet(client: OpenAI, model: str, snippet: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM.format(dims=", ".join(DIMENSIONS))},
            {"role": "user", "content": CLASSIFIER_USER_TEMPLATE.format(snippet=snippet, dims=", ".join(DIMENSIONS))},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise RuntimeError(f"Model did not return JSON. Raw:\n{content}")
        return json.loads(m.group(0))


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    # OpenAI embeddings API supports batching; many local endpoints do too.
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-dir", type=Path, default=Path("./data/transcripts"))
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs"))
    ap.add_argument("--chunk-tokens", type=int, default=int(os.environ.get("CHUNK_TOKENS", 500)))
    ap.add_argument("--chunk-overlap", type=int, default=int(os.environ.get("CHUNK_OVERLAP", 250)))
    ap.add_argument("--tokenizer-hint", type=str, default=os.environ.get("TOKENIZER_HINT", "gpt-4o-mini"))
    args = ap.parse_args()

    cfg = Config(
        transcripts_dir=args.transcripts_dir,
        out_dir=args.out_dir,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        tokenizer_hint=args.tokenizer_hint,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_transcripts(cfg.transcripts_dir)
    if df.empty:
        raise SystemExit(f"No .txt transcripts found in {cfg.transcripts_dir.resolve()}")

    client = make_client(cfg)

    # Chunk
    chunks = []
    for _, r in df.iterrows():
        for k, ch in enumerate(chunk_text(r["text"], cfg.chunk_tokens, cfg.chunk_overlap, cfg.tokenizer_hint)):
            chunks.append({"file": r["file"], "chunk_id": k, "text": ch})
    chunks_df = pd.DataFrame(chunks)
    if chunks_df.empty:
        raise SystemExit("No chunks created (empty transcripts?)")

    # Classify
    labels = []
    for txt in tqdm(chunks_df["text"].tolist(), desc="Classifying"):
        out = classify_snippet(client, cfg.classifier_model, txt)
        labels.append(out)
    labels_df = pd.json_normalize(labels)
    annotated = pd.concat([chunks_df, labels_df], axis=1)
    annotated_path = cfg.out_dir / "annotated_chunks.parquet"
    annotated.to_parquet(annotated_path, index=False)

    # Embed
    texts = annotated["text"].tolist()
    vecs = []
    batch = 64
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        vecs.append(embed_texts(client, cfg.embedding_model, texts[i:i+batch]))
    vecs = np.vstack(vecs)
    embedded = annotated.copy()
    embedded["embedding"] = list(vecs)

    embedded_path = cfg.out_dir / "embedded_chunks.parquet"
    embedded.to_parquet(embedded_path, index=False)

    print(f"Wrote:\n- {annotated_path}\n- {embedded_path}")


if __name__ == "__main__":
    main()
