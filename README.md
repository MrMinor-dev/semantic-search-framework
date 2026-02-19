# Semantic Search Framework

**Production retrieval infrastructure: 17,428 embedded chunks with sub-second search and incremental indexing**

A complete pipeline for parsing, chunking, embedding, and retrieving unstructured text at scale — built to make 350+ AI conversations and 500+ documents searchable in under a second. Migrated from local ChromaDB to cloud Supabase pgvector. Hash-based incremental updates run in <10 seconds vs. 5+ minutes for full rebuilds.

---

## The Problem

When an AI agent operates across hundreds of sessions, it generates enormous amounts of institutional knowledge — conversations, decisions, debugging stories, architectural rationale. But without retrieval infrastructure, that knowledge is inaccessible:

- **Brute force doesn't scale.** Loading "everything that might be relevant" into context burns your token budget before the agent does any work. At 373 conversations and 1.2M words, you can't just dump it all in.
- **File-level retrieval is too coarse.** Finding the right *document* isn't enough — you need the right *paragraph* from the right document. A 5,000-word conversation might contain exactly 3 sentences that answer your question.
- **Full rebuilds are unsustainable.** When one document changes, re-embedding 17,000+ chunks wastes 5+ minutes of compute. You need to know what changed and only re-process that.
- **Local solutions don't compose.** A ChromaDB instance running on one machine can't serve a cloud automation platform. The embedding store needs to be accessible from wherever the agent operates.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              SEMANTIC SEARCH PIPELINE                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│  INGEST                                              │
│  ┌────────────────────────────────────────────┐      │
│  │ Source files (Markdown, conversations)      │      │
│  │         ↓                                  │      │
│  │ Parse → Normalize → Chunk (~500 tokens)    │      │
│  │         ↓                                  │      │
│  │ Hash check: content changed?               │      │
│  │   No  → Skip (already indexed)             │      │
│  │   Yes → Delete old chunks → Re-embed       │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  EMBED                                               │
│  ┌────────────────────────────────────────────┐      │
│  │ sentence-transformers (all-MiniLM-L6-v2)   │      │
│  │ 384-dimensional vectors                    │      │
│  │ Batched processing with error handling     │      │
│  └────────────────────────────────────────────┘      │
│                    ↓                                 │
│  STORE                                               │
│  ┌────────────────────────────────────────────┐      │
│  │ Supabase PostgreSQL + pgvector             │      │
│  │ HNSW index (cosine similarity)             │      │
│  │ UNIQUE(file_path, chunk_index) constraint  │      │
│  │ content_hash for smart reindexing          │      │
│  └────────────────────────────────────────────┘      │
│                    ↓                                 │
│  RETRIEVE                                            │
│  ┌────────────────────────────────────────────┐      │
│  │ n8n webhook → HuggingFace embed query      │      │
│  │         ↓                                  │      │
│  │ pgvector similarity search                 │      │
│  │         ↓                                  │      │
│  │ Ranked results with metadata               │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Ingest Layer

Source material comes in two forms: structured Markdown documents (architecture docs, schemas, plans) and unstructured AI conversations (373 conversations, 1.2M words, 224MB). The parser normalizes both into consistent chunks of ~500 tokens with overlap to preserve context at boundaries.

Before embedding, each file's content is hashed (MD5). The hash is compared against what's stored in the database. If unchanged — skip entirely. If changed — delete old chunks and re-embed. This single optimization reduced routine index updates from 5+ minutes to under 10 seconds.

### Embedding Layer

Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace API — 384-dimensional vectors optimized for semantic similarity. Processing is batched with error handling for API rate limits and transient failures.

### Storage Layer

Migrated from local ChromaDB to Supabase PostgreSQL with pgvector extension. The migration solved three problems at once: cloud accessibility (any service can query it), persistence (no local process to keep running), and composability (same database that hosts the application schema).

Key schema decisions:
- **HNSW index** on the embedding column for fast approximate nearest-neighbor search
- **UNIQUE constraint** on `(file_path, chunk_index)` to prevent duplicate chunks
- **content_hash column** enabling the incremental update optimization
- **JSONB metadata** column for flexible file-level attributes (name, folder, size, date)

### Retrieval Layer

A cloud webhook receives search queries, embeds the query text using the same model, and runs a cosine similarity search against pgvector. Results return ranked by relevance with source metadata — allowing the consuming agent to cite where information came from.

The core retrieval query:

```sql
SELECT content, metadata, 1 - (embedding <=> query_embedding) AS similarity
FROM doc_embeddings
WHERE 1 - (embedding <=> query_embedding) > 0.3
ORDER BY similarity DESC
LIMIT 10;
```

The `<=>` operator is pgvector's cosine distance. Subtracting from 1 converts distance to similarity (0 = unrelated, 1 = identical). The `0.3` threshold filters noise — results below that score are semantically unrelated to the query. The query runs in under a second on 17,428 vectors because pgvector uses an HNSW index: approximate nearest-neighbor search, not brute force.

The "search before load" policy: always query the index first, then load full files only when the chunks confirm relevance. This pattern consistently saves 80%+ of the tokens that would be wasted on speculative document loading.

---

## Key Insight

**The migration from local to cloud wasn't a technical upgrade — it was an architectural unlock.**

ChromaDB worked fine as a local embedding store. But it couldn't serve the cloud automation platform (n8n), couldn't be queried from different machines, and required a running Python process. Moving to Supabase pgvector turned the search index from a local tool into a shared service — any workflow, any agent, any interface could query it.

The lesson: retrieval infrastructure only compounds when it's accessible from everywhere the system operates. A search index locked to one machine is a local optimization. A search index on the network is a platform capability.

---

## Debugging Journey

Real problems encountered and solved during development:

**Python version incompatibility.** Python 3.14 was too new for sentence-transformers dependencies. Downgraded to 3.12. Lesson: bleeding-edge runtimes and ML libraries don't mix.

**JSON parsing failures from stdout pollution.** The MCP server communicated via JSON over stdout, but logging statements contaminated the stream. Took hours to diagnose because the errors were intermittent — only triggered when certain log paths fired. Fix: strict separation of logging from communication channels.

**Batching iteration bugs.** The `update_files` function had an off-by-one error in its batch processing loop that silently dropped the last batch when file count wasn't evenly divisible. Only caught by comparing expected vs. actual chunk counts post-index.

**Webhook vs. MCP payload differences.** The same search function received differently-structured inputs depending on whether it was called via n8n webhook or MCP. Intermittent failures traced to payload structure assumptions. Fix: defensive type-checking at the input boundary of every utility function.

---

## Results

| Metric | Value |
|--------|-------|
| Total embedded chunks | **17,428** (14,335 conversation + 3,093 document) |
| Source material | **373 conversations, 1.2M words, 224MB** + **506 files** |
| Retrieval speed | **Sub-second** |
| Incremental update time | **<10 seconds** (vs. 5+ min full rebuild) |
| Chunking strategy | **~500 tokens per chunk** with overlap |
| Vector dimensions | **384** (all-MiniLM-L6-v2) |
| Storage | **Supabase pgvector** (migrated from ChromaDB) |
| Consuming services | **Semantic search webhook, story mining, session context retrieval** |

---

## Applications

This infrastructure enabled several downstream capabilities that wouldn't have been possible with file-level search:

- **Story mining:** 50 semantic queries across the full conversation history extracted 485 evidence chunks for a professional portfolio — finding specific debugging stories, architectural decisions, and quantified outcomes buried across hundreds of sessions.
- **Session continuity:** Agent retrieves relevant historical context without loading entire conversations. "Search before load" policy reduces token consumption dramatically.
- **Audit and compliance:** Searchable history of every decision, rationale, and trade-off across 350+ sessions of autonomous operation.

---

## Built With

- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 via HuggingFace API
- **Vector storage:** Supabase PostgreSQL + pgvector (HNSW indexing)
- **Retrieval:** n8n webhook workflows
- **Indexing:** Python with hash-based change detection
- **Previous:** ChromaDB (local, migrated away)
- **Production scale:** 17,428 chunks, 506 files, 350+ sessions consuming

---

## License

MIT

## Author

**Jordan Waxman** — [mrminor-dev.github.io](https://mrminor-dev.github.io)

14 years operations leadership → building production AI infrastructure. This search framework was developed to solve a real operational problem: making 350+ sessions of AI-generated institutional knowledge retrievable in sub-second time, so the next session starts smarter than the last.
