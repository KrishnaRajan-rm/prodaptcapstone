# System Design and Trade-offs

## Scope

The assistant turns a manager's operational question into retrieved review evidence, structured specialist findings, ranked actions, and a quality-checked report. FastAPI is the public boundary; LangGraph owns workflow routing; the four ADK services own specialist analysis; CrewAI consolidates and checks the report; the Python Quality Gate remains authoritative.

## Data and Retrieval

| Decision | Benefit | Trade-off |
|---|---|---|
| Stream restaurant reviews into Parquet and use DuckDB for aggregate analysis | Trend and satisfaction calculations use broad historical coverage without loading multi-gigabyte JSONL files into memory | Preprocessing takes time and requires local disk space |
| Embed a bounded, stratified review subset while retaining aggregate statistics for the full corpus | Keeps CPU indexing and query-time vector search practical while preserving broad trend coverage | A semantically relevant review outside the embedded subset cannot be returned as a retrieved quote |
| Combine SQLite FTS5/BM25, normalized dense vectors, and reciprocal-rank fusion | Keyword matching supports specific terms while embeddings capture semantic similarity; both indexes run locally | Metadata, FTS, and vector row mappings must stay consistent through ingestion and rebuilds |
| Cross-encode only a bounded candidate set | Improves final ordering without reranking the full corpus | Adds model startup and query latency; retrieval can continue without reranking if the model is unavailable |

## Workflow and Services

LangGraph validates the request, retrieves context, applies the LLM-boundary PII gate, runs classification and trend analysis in parallel, then conditionally routes to root cause and action planning. CrewAI receives only outputs that actually ran and the compact evidence pack. The Quality Gate evaluates grounding and consistency after drafting; it is not replaced by the CrewAI validator.

The specialist services remain separately deployable over HTTP/A2A. This preserves independent health checks and scaling, at the cost of requiring services to be started and making dependency failures visible to the gateway. The gateway does not silently replace an unavailable service with an in-process agent.

## Privacy and Failure Behavior

Gate 1 masks indexed review text; Gate 2 masks the outgoing request and evidence before external LLM calls. Reversible values are request-scoped and decoding occurs only at the authorized response boundary. Trace payloads do not include the raw query.

Mock mode is for tests and local component development. The public analysis endpoint rejects mock mode rather than presenting a deterministic substitute as a live answer. In live mode, provider, retrieval, specialist, and drafting failures propagate as structured HTTP errors with the relevant request ID and cause. The React client clears an earlier result when a new request starts so a failed request cannot appear to have returned stale findings.

## Confidence and Data Limits

Confidence and quality verdict are separate outputs: a high score does not erase a failed grounding, recommendation, or consistency check. Historical trends describe the data actually available and are not claims about current restaurant operations. The indexed Yelp review corpus ends in January 2022; current operational conclusions require newer source data.

## Operational Trade-offs

- Local SQLite, DuckDB, Parquet, and memory-mapped vectors avoid a Docker or hosted vector-database dependency, but require local storage and explicit index lifecycle management.
- A bounded dense index favors laptop-friendly setup; full-corpus statistics remain available through Parquet/DuckDB.
- Separate agent services make failures diagnosable, but local development must start the required services and configure a live LLM provider.
- The UI waits for the completed API response rather than simulating intermediate stages. Its trace is built from measured graph-node execution telemetry.

## References

- [Architecture flow](../README.md#architecture-diagram--data-flow)
- [Architecture decisions](DECISIONS.md)
- [Project requirements](SPEC.md)
- [API and execution flow](../app/api/routes.py)