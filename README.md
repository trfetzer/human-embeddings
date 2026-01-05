# Human embeddings — team design from interview transcripts

This repo packages the **“human embeddings”** experiment described in the blog post **_Human embeddings_** (Thiemo Fetzer, **5 Jan 2026**) and provides a reproducible notebook + a minimal CLI pipeline for:

1. **Chunking** diarized interview transcripts into ~250–500 token snippets  
2. **Annotating** each snippet into skill/trait dimensions (e.g. *technical depth*, *creativity*, *project management*, *marketing*)  
3. **Embedding** snippets into a high‑dimensional vector space  
4. Constructing **concept vectors** (dimension directions) and running a **combinatorial team allocation** that maximizes “coverage” of the simplex / spanned space.

Blog context and motivation (HR, matching markets, and team formation):  
- **Human embeddings** (source post): https://www.trfetzer.com/human-embeddings/

---

## What’s inside

- `notebooks/team_interview_annotator.ipynb` — end‑to‑end reference notebook 
- `scripts/run_pipeline.py` — chunk → classify → embed → parquet outputs (optional; see below)
- `scripts/team_design.py` — exhaustive team scoring / sampling (optional; see below)
- `data/example_transcripts/demo_transcript.txt` — tiny synthetic example so you can run without private data

Outputs (by default):
- `outputs/annotated_chunks.parquet`
- `outputs/embedded_chunks.parquet`
- figures in `outputs/figures/` (if you run plotting)

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Provide transcripts

Put diarized `.txt` transcripts into:

```text
data/transcripts/
```

(Or start with the synthetic demo transcript in `data/example_transcripts/`.)

### 3) Set model endpoints / keys

You have two main options:

#### A) Local (recommended): Ollama (OpenAI‑compatible endpoint)

If you run a local OpenAI-compatible server (e.g. Ollama’s `/v1` compatibility layer), set:

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"   # any string; Ollama ignores it
export CLASSIFIER_MODEL="gpt-oss:20b"
export EMBEDDING_MODEL="mxbai-embed-large"
```

> Note: model names depend on what you have pulled locally.

#### B) Hosted OpenAI API

```bash
export OPENAI_API_KEY="..."
export CLASSIFIER_MODEL="gpt-4o-mini"
export EMBEDDING_MODEL="text-embedding-3-large"
```

### 4) Run

Notebook:
```bash
jupyter lab
# open notebooks/team_interview_annotator.ipynb
```

CLI pipeline:
```bash
python scripts/run_pipeline.py --transcripts-dir data/transcripts --out-dir outputs
python scripts/team_design.py --embedded-parquet outputs/embedded_chunks.parquet --out-dir outputs
```

---

## Notes on privacy & consent

This workflow is designed to support **self‑sovereign** handling of interview data (transcribe locally; embed locally; keep raw text off third‑party servers). If you are working with real people’s interviews, treat transcripts and embeddings as sensitive data and obtain informed consent.

---

## How to cite / credit

If you use this repo, please cite the accompanying post:

- Thiemo Fetzer, “Human embeddings”, 5 Jan 2026.

---

## License

MIT (see `LICENSE`).
