# BUFANDA — Multimodal Fashion Search Engine

A personal, non-commercial semantic search engine for fashion. Search a clothing dataset by typing a natural language description or uploading a photo, and get visually and semantically similar outfit results back in real time.

Built with Rust, ONNX Runtime, FashionCLIP, Qdrant, and llava:7b.

## Demo

**Text search**
![Text search demo](assets/text_demo.gif)

**Image search**
![Image search demo](assets/image_demo.gif)

---

## How it works

Bufanda uses a dual-vector retrieval strategy. Every image in the dataset is indexed twice — once as an image embedding and once as a text embedding — and both are stored in a Qdrant vector collection. At query time, the same CLIP model encodes either the text query or the uploaded image, and an approximate nearest-neighbor search finds the most semantically similar results using cosine similarity.

```
Query (text or image)
        │
        ▼
  FashionCLIP encoder  (ONNX, runs locally)
        │
        ▼
  512-dim embedding
        │
        ▼
  Qdrant ANN search  (cosine similarity)
        │
        ▼
  Ranked results  →  served via Axum HTTP API  →  browser UI
```

At indexing time, each image also goes through a VLM captioning step (llava:7b via ollama) to produce a natural language description, which is encoded into the `text` vector space. This lets text queries find images even when the raw visual content would not be enough.

---

## Tech stack

| Component | Technology |
|---|---|
| HTTP server | Axum (Rust) |
| ML inference | ONNX Runtime via `ort` |
| Image embeddings | FashionCLIP (`fashion-clip`) |
| Text embeddings | `clip-ViT-B-32-multilingual-v1` |
| Captioning | llava:7b via ollama |
| Vector database | Qdrant Cloud |
| Frontend | Vanilla HTML/CSS/JS |

---

## Models

The models used are from the [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) project:

- **Image encoder**: `fashion-clip` — a CLIP model fine-tuned on fashion data
- **Text encoder**: `clip-ViT-B-32-multilingual-v1` — multilingual CLIP for text embeddings

Both models were downloaded using Python and the `optimum` library from Hugging Face, then exported to ONNX format for use in the Rust inference pipeline. They are not included in this repository and must be exported separately (see setup below).

Both encoders produce 512-dimensional normalized embeddings stored in Qdrant under named vector spaces (`image` and `text`).

---

## Dataset

This project uses the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset for indexing. Only the validation split is used, and only images annotated with `"source": "user"` (real-world photos, not studio/shop shots) are indexed. The indexer shuffles images before processing to improve result diversity, since DeepFashion2 groups consecutive images by garment.

The dataset is **not included** in this repository and is subject to its own license terms — see the DeepFashion2 repository for access instructions.

---

## Project structure

```
src/
  api/        — Axum HTTP handlers (health, text search, image search, image serving)
  bin/
    indexer.rs          — Indexes DeepFashion2 images into Qdrant
    setup.rs            — Creates the Qdrant collection with dual vector spaces
    rebuild_progress.rs — Recovers indexer_progress.log from the live database
  db/         — Qdrant client wrapper (upsert and ANN search)
  engine/     — Connects the AI model and database into a single search interface
  processor/
    image.rs  — Preprocessing pipeline (resize → RGB → CLIP normalization)
    model.rs  — BufandaAI: wraps two ONNX sessions and a tokenizer
  captioner.rs — VLM captioning via ollama HTTP API
  main.rs     — Server entry point
static/
  index.html  — Search UI (text and image modes, masonry results grid)
tests/
  unit_image.rs           — Preprocessing tensor shape and value tests
  unit_model.rs           — Embedding dimension, norm, and semantic sanity tests
  integration_captioner.rs — Captioner integration tests (requires ollama)
  integration_pipeline.rs  — Full DB + engine integration tests
  e2e.rs                  — End-to-end: mini-index → search → HTTP API
```

---

## Setup

### Prerequisites

- Rust (2024 edition)
- [ollama](https://ollama.com/) with `llava:7b` pulled
- A [Qdrant Cloud](https://qdrant.tech/) cluster (or local Qdrant instance)
- Python + `optimum` + `transformers` for exporting the models to ONNX

### 1. Export the models

```python
from optimum.exporters.onnx import main_export

# Text encoder
main_export("clip-ViT-B-32-multilingual-v1", output="models/text/", task="feature-extraction")

# Image encoder
main_export("patrickjohncyh/fashion-clip", output="models/image/", task="feature-extraction")
```

Place the resulting `model.onnx` files and `tokenizer.json` at:

```
models/
  text/
    model.onnx
    tokenizer.json
  image/
    model.onnx
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_api_key
COLLECTION=outfits
PORT=3000
```

### 3. Create the Qdrant collection

```bash
cargo run --bin setup
```

This creates a collection with two named 512-dimensional cosine-similarity vector spaces (`image` and `text`).

### 4. Index the dataset

```bash
cargo run --bin indexer
```

The indexer resumes automatically from `indexer_progress.log` if interrupted. Only `source: "user"` images are indexed. Each image is captioned with llava:7b (ollama must be running), encoded with FashionCLIP, and upserted into Qdrant with retry logic.

If the progress log is lost or out of sync with the database:

```bash
cargo run --bin rebuild_progress
```

### 5. Start the server

```bash
cargo run
```

The server starts on `http://0.0.0.0:3000` by default. Open the URL in your browser to use the search UI.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/search/text?limit=N` | JSON body `{ "query": "..." }` |
| `POST` | `/search/image?limit=N` | `multipart/form-data` with an `image` field |
| `GET` | `/image?path=...` | Serves a dataset image by file path |

`limit` defaults to 20 and is capped at 100.

---

## Running tests

```bash
# Unit tests (no external dependencies)
cargo test --test unit_image
cargo test --test unit_model

# Integration tests (requires Qdrant + ollama)
cargo test --test integration_captioner -- --nocapture
cargo test --test integration_pipeline -- --nocapture

# End-to-end (requires running server + indexed data)
cargo test --test e2e -- --nocapture --test-threads=1
```

---

## Development environment (Nix)

A `flake.nix` is included with a complete dev shell. It sets up the Rust toolchain, ONNX Runtime, OpenSSL, and ollama automatically via `nix develop`.

---

## License

The source code in this repository is released under the [MIT License](LICENSE).

The DeepFashion2 dataset and FashionCLIP models are subject to their own respective licenses. This project is personal and non-commercial.
