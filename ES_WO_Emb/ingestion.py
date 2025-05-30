from elasticsearch import Elasticsearch
import json

# Connect to Elastic Cloud
es = Elasticsearch(
    "https://my-elasticsearch-project-c31d65.es.us-central1.gcp.elastic.cloud:443",
    api_key="bzRUUjFKWUJDOFpvVS1UeUlTY046T1VfVFJtTHBDc1M4M0RzZnFkcmhVdw=="
)

index_name = "report_demo"

# Check if the index exists
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "filename": {"type": "text"},
                "text": {"type": "text"},
            }
        }
    )

# Sample documents
with open("index.txt","r",encoding="utf-8") as f:
    docs=json.load(f)

# Index documents
for doc in docs:
    es.index(index=index_name, document=doc)

print("✅ Indexing complete.")
