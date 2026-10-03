# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deliberately bare-bones, single-file RAG (Retrieval-Augmented Generation) demo for educational purposes. The entire pipeline — PDF ingestion, chunking, embedding, vector storage, retrieval, and answering — lives in `rag_demo/__main__.py`. There is no abstraction layer by design: the point is to show the mechanics of RAG without framework magic (no LangChain/LlamaIndex). Do not "clean up" the naive chunking or add abstractions unless asked — that's intentional per the README disclaimer.

## Setup & commands

Prerequisites: Python 3.12+, Poetry, Docker (for Postgres/pgvector).

```bash
# Start pgvector in Docker (host port 6432, not 5432, to avoid collisions)
docker run -p 6432:5432 --name pgvector -e POSTGRES_PASSWORD=postgres -d pgvector/pgvector:pg17

# Create DB and load schema
psql -h localhost -p 6432 -U postgres -c "CREATE DATABASE rag_demo;"
psql -h localhost -p 6432 -U postgres rag_demo < schema.sql

# Env vars: copy example_env.txt to .env and set GEMINI_API_KEY and VOYAGE_API_KEY, then
source .env

# Install deps
poetry install
```

Run:
```bash
poetry run python -m rag_demo
# Skip re-embedding (reuse existing DB rows) when iterating on retrieval/answering:
poetry run python -m rag_demo --skip-embedding-step
```

Inspect stored chunks:
```bash
psql -h localhost -p 6432 -U postgres rag_demo -c "SELECT * FROM chunks LIMIT 5;"
```

There are no tests, lint config, or build steps in this repo.

## Architecture

Single script, `rag_demo/__main__.py`, run top-to-bottom:

1. **Embedding & answering via Google's `google-genai` SDK**:
   - Embeddings: model `gemini-embedding-001` (3072 dims).
   - Answering: model `gemini-3.8-flash`.
   - Both go through one client, `genai.Client()`, authed via `GEMINI_API_KEY`.
2. **Reranking via Voyage AI's `voyageai` SDK**: model `rerank-2-lite`, authed via `VOYAGE_API_KEY`. Used only for reranking — embeddings and answering stay on Gemini.
3. **Ingestion** (skipped if `--skip-embedding-step`): reads every file in `data/` with `PyPDF2`, concatenates page text, splits it into fixed-length chunks (`CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 150`, naive `split_string_by_length` — no semantic/overlap-aware chunking), embeds each chunk, and inserts into Postgres.
4. **Storage**: Postgres + `pgvector`. Schema (`schema.sql`) defines one table, `chunks(id, embedding vector(3072), chunk TEXT, tsv tsvector GENERATED ALWAYS AS to_tsvector('english', chunk))` with a GIN index on `tsv` — 3072 dims to match `gemini-embedding-001`'s default output dimension. `__main__.py` truncates this table on every run unless `--skip-embedding-step` is passed.
5. **Retrieval (hybrid search)**: embeds the question, then runs two candidate searches in one SQL statement — pgvector cosine distance (`embedding <=> question_embedding`) and Postgres full-text search (`ts_rank_cd` over `tsv` via `plainto_tsquery`) — each capped at 20 rows, fused via Reciprocal Rank Fusion (`RRF_K = 60`) into a single ranked list of up to 20 candidates.
6. **Reranking**: the fused candidates are sent to Voyage's `rerank()` with the original question; the top 5 by `relevance_score` become the final context chunks.
7. **Answering**: concatenates the top chunks into a context block, builds a prompt, and calls `gemini-3.8-flash` via `generate_content()` — the answer is generated from the context, not extractively lifted from it. The user is optionally shown the raw prompt string via a Y/N input after the answer.

Connection string defaults to `postgresql://postgres:postgres@localhost:6432/rag_demo` (override via `DATABASE_URL`).

When extending this repo, follow the existing single-file, no-framework style unless asked to refactor.
