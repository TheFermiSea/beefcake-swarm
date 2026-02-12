import functools
from datetime import timedelta

import cocoindex
from cocoindex import flow_def, FlowBuilder, DataScope, add_transient_auth_entry
from cocoindex.sources import LocalFile
from cocoindex.functions import DetectProgrammingLanguage, SplitRecursively, EmbedText
from cocoindex.llm import LlmApiType
from cocoindex.targets import Postgres
from cocoindex.index import VectorIndexDef, VectorSimilarityMetric, HnswVectorIndexMethod
from cocoindex.setting import DatabaseConnectionSpec

DB_URL = "postgresql://localhost/cocoindex"

# Register the API key
api_key_ref = add_transient_auth_entry("sk-dummy")

# Register the database connection (local Postgres with pgvector)
db_ref = add_transient_auth_entry(
    DatabaseConnectionSpec(
        url=DB_URL,
        user="briansquires"
    )
)

# --- Transform flow: reuse the same embedding logic for indexing and querying ---
@cocoindex.transform_flow()
def text_to_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[list[float]]:
    return text.transform(
        EmbedText(
            api_type=LlmApiType.OPENAI,
            model="nomic-embed-code.Q8_0",
            address="http://10.0.0.21:8080/v1",
            api_key=api_key_ref,
            output_dimension=3584
        )
    )

@flow_def(name="code_indexing")
def my_flow(builder: FlowBuilder, scope: DataScope):
    # 1. Source: rust-daq repository
    # refresh_interval enables live updates - checks for changes every 30 seconds
    files = builder.add_source(
        LocalFile(
            path="/Users/briansquires/code/rust-daq",
            included_patterns=[
                "**/*.rs", "**/*.toml", "**/*.md", "**/*.py", "**/*.sh",
                "**/*.yaml", "**/*.yml", "**/*.json", "**/*.cfg",
            ],
            excluded_patterns=[
                # Build artifacts
                "target/**", "**/target/**",
                # VCS
                ".git/**", ".worktrees/**", "*.lock",
                # Python environments
                "**/.venv/**", "**/venv/**", "**/node_modules/**",
                "**/site-packages/**", "**/__pycache__/**",
                # AI/tooling metadata (not source code)
                ".planning/**", ".claude/**", ".claude.DANGEROUS.backup/**",
                ".codemachine/**", ".factory/**", ".prompts/**",
                ".beads/**", ".brv/**", ".jules/**", ".pal_venv/**",
                # Generated code
                "**/generated/**", "**/*.pb.rs",
            ]
        ),
        refresh_interval=timedelta(seconds=30),
    )

    # Use 'row' to operate on each file
    row = files.row()

    # 2. Language Detection
    row["language"] = row["filename"].transform(DetectProgrammingLanguage())

    # 3. Chunking - 1000 chars keeps most Rust functions intact
    row["chunks"] = row["content"].transform(
        SplitRecursively(), language=row["language"], chunk_size=1000, chunk_overlap=100
    )

    # 4. Flatten and Embed (Operate on each chunk)
    chunks = row["chunks"].row()

    chunks["embedding"] = text_to_embedding(chunks["text"])

    # 5. Target: Local Postgres
    collector = scope.add_collector()

    collector.collect(
        filename=row["filename"],
        language=row["language"],
        chunk_location=chunks["location"],
        chunk_content=chunks["text"],
        embedding=chunks["embedding"]
    )

    collector.export(
        "postgres_export",
        Postgres(database=db_ref, table_name="code_chunks"),
        primary_key_fields=["filename", "chunk_location"],
        vector_indexes=[
            VectorIndexDef(
                field_name="embedding",
                metric=VectorSimilarityMetric.COSINE_SIMILARITY,
                method=HnswVectorIndexMethod(ef_construction=128)
            )
        ]
    )


# --- Query handler: semantic search over the index ---
@functools.cache
def _connection_pool():
    from psycopg_pool import ConnectionPool
    from pgvector.psycopg import register_vector

    def _configure(conn):
        register_vector(conn)

    return ConnectionPool(DB_URL, configure=_configure)


@my_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["embedding"],
        score="score"
    )
)
def semantic_search(query: str, top_k: int = 10) -> cocoindex.QueryOutput:
    """Search code chunks by semantic similarity."""
    table_name = "code_chunks"
    query_vector = text_to_embedding.eval(query)

    with _connection_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT filename, language, chunk_content, embedding,
                       1.0 - (embedding <=> %s::vector) AS score
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, top_k)
            )
            rows = cur.fetchall()

    return cocoindex.QueryOutput(
        query_info=cocoindex.QueryInfo(
            embedding=query_vector,
            similarity_metric=VectorSimilarityMetric.COSINE_SIMILARITY
        ),
        results=[
            {
                "filename": r[0],
                "language": r[1],
                "chunk_content": r[2],
                "embedding": r[3],
                "score": float(r[4])
            }
            for r in rows
        ]
    )


if __name__ == "__main__":
    import sys

    args = set(sys.argv[1:])
    server = "--server" in args
    live = "--live" in args

    if server:
        print("Starting CocoIndex server on 127.0.0.1:49344 ...")
        print("  Query endpoint: POST /api/semantic_search")
        print("  CocoInsight:    https://cocoindex.io/cocoinsight")
        if live:
            # Server + live: keep index fresh while serving queries
            print("  Live updates:   enabled (30s refresh)")
            with cocoindex.FlowLiveUpdater(
                my_flow,
                cocoindex.FlowLiveUpdaterOptions(print_stats=True)
            ):
                cocoindex.start_server(cocoindex.ServerSettings(
                    address="127.0.0.1:49344",
                    cors_origins=["https://cocoindex.io"]
                ))
        else:
            cocoindex.start_server(cocoindex.ServerSettings(
                address="127.0.0.1:49344",
                cors_origins=["https://cocoindex.io"]
            ))
    elif live:
        # Live mode only: continuously watch for changes
        print("Starting live updater (Ctrl+C to stop)...")
        with cocoindex.FlowLiveUpdater(
            my_flow,
            cocoindex.FlowLiveUpdaterOptions(print_stats=True)
        ) as updater:
            updater.wait()
    else:
        # One-shot update
        my_flow.update(full_reprocess=True, print_stats=True)
