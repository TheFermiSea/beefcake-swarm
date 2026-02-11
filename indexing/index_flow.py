from cocoindex import Flow
from cocoindex.sources import LocalFile
from cocoindex.functions import EmbedText
from cocoindex.llm import LlmApiType
from cocoindex.targets import Postgres
from cocoindex.auth_registry import AuthEntryReference
from cocoindex.setting import DatabaseConnectionSpec

# Initialize the flow
flow = Flow(name="code_indexing")

# 1. Source: Index Python files in the current directory
flow.source(LocalFile(path=".", include=["**/*.py", "**/*.md", "**/*.sh"]))

# 2. Transformation: Embed using the Remote GPU Cluster
# This sends text to your cluster (vasp-02 via pve1 forwarding) for heavy lifting
flow.transform(
    "embedding",
    "content",
    EmbedText(
        api_type=LlmApiType.OPENAI,
        model="nomic-embed-text-v1.5.Q8_0",
        address="http://pve1.tailc46cd0.ts.net:8080/v1",  # Your remote cluster endpoint
        api_key="sk-dummy"  # Required by client, ignored by server
    )
)

# 3. Target: Local Postgres with pgvector
# "postgres" is the default user on Mac Postgres.app usually, 
# but "briansquires" might be the user based on `whoami`. 
# We'll try the standard local unix socket connection or localhost.
flow.target(
    Postgres(
        table_name="code_embeddings",
        # We need to define the connection. 
        # For now, we'll assume a standard local setup. 
        # You might need to adjust user/pass/db in the connection spec.
    )
)
