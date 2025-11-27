## Movie Search Pipeline

This script builds an in-memory semantic search playground focused on comparing chunking strategies before indexing movie synopses in Qdrant.

### What the script does
- Loads a curated list of sci-fi movie descriptions with metadata (`name`, `author`, `year`).
- Instantiates `SentenceTransformer("all-MiniLM-L6-v2")` for embeddings and a Hugging Face tokenizer to measure chunk sizes.
- Prints token counts per description to highlight anything exceeding the `256`-token threshold used elsewhere in the stack.
- Spins up an in-memory `QdrantClient` and creates a `my_movies` collection with three named vector slots (`fixed`, `sentence`, `semantic`) so each chunking strategy can be queried independently while sharing payload metadata.
- Defines three chunking approaches:
  - **Fixed size**: deterministic blocks of `MAX_TOKENS=40`.
  - **Sentence splitter**: `SentenceSplitter` with configurable overlap to keep adjacent context.
  - **Semantic splitter**: `SemanticSplitterNodeParser` (Hugging Face embeddings + breakpoint percentile) to cut text where semantics change.
- Iterates over every movie description, runs all three chunkers, encodes each chunk with the shared encoder, and uploads the resulting vectors/payloads to Qdrant.
- Runs example searches (`"alien invasion"` using semantic chunks, `"time travel"` using sentence chunks) and prints the top matches with similarity scores.

### Runtime flow

```mermaid
flowchart TD
    A[Load movie descriptions] --> B[Count tokens with AutoTokenizer]
    B --> C{Need chunking?}
    C -->|Yes| D[Fixed-size chunks]
    C -->|Yes| E[Sentence splitter chunks]
    C -->|Yes| F[Semantic splitter chunks]
    D --> G[Encode via SentenceTransformer]
    E --> G
    F --> G
    G --> H[Populate PointStruct payloads]
    H --> I[Upload to Qdrant collection]
    I --> J[Run sample semantic queries]
```

### Key constants
- `model_name = "sentence-transformers/all-MiniLM-L6-v2"` drives both embedding and token counting.
- `MAX_TOKENS = 40` keeps chunks compact so they stay under common LLM context limits.
- `collection_name = "my_movies"` isolates this experiment from other Qdrant collections.

### Running the script

```bash
cd /Users/sahilagarwal/Projects/qdrant-exploration
python movie_search_system/app.py
```

Execution prints token stats, the number of vectors uploaded, and the search hits per chunking strategy, giving a fast way to see how chunk granularity affects retrieval.

