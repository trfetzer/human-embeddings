# Explainer: what “human embeddings” are doing here

This repository operationalizes a simple idea:

> represent each person as a vector in a latent space, inferred from how they talk about their skills and work,
> then use geometry to design teams.

For the motivation and a narrative walk-through, see the accompanying post:
- https://www.trfetzer.com/human-embeddings/

## Pipeline

1. **Interview → transcript**  
   Interviews are transcribed (ideally locally) and diarized so each speaker can be separated.

2. **Chunk**  
   Each transcript is split into overlapping snippets (e.g. 500 tokens with 250 overlap).

3. **Label snippets into dimensions**  
   A model classifies each chunk into one predominant “dimension” (e.g. technical depth, creativity, project management, marketing).

4. **Embed snippets**  
   Each chunk becomes a high-dimensional vector. This step “fuzzes language” into a geometry where semantic similarity is measurable.

5. **Concept vectors**  
   For each dimension, take the mean direction of all chunks assigned to that label: a “concept vector” that anchors the dimension in the embedding space.

6. **Team design via combinatorics**  
   For small cohorts, enumerate possible allocations into groups of fixed sizes and score allocations by how well the teams cover the conceptual simplex / span the space.

## Why this repo is self-contained

- The notebook and scripts run end-to-end.
- You can run the full pipeline locally using an OpenAI-compatible endpoint (e.g. Ollama) so transcripts never leave your machine.
- A tiny synthetic transcript is included for testing.

## Ethical / practical cautions

Embeddings and inferred “traits” are not ground truth. They reflect prompts, context, and model behavior. Treat results as decision support, not automated hiring or ranking.
