from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch(
    "https://my-elasticsearch-project-c31d65.es.us-central1.gcp.elastic.cloud:443",
    api_key="bzRUUjFKWUJDOFpvVS1UeUlTY046T1VfVFJtTHBDc1M4M0RzZnFkcmhVdw=="
)

# Replace with your actual index name
index_name = "report_demo"

# Search to view documents (fetch first 10)
response = es.search(index=index_name, query={"match_all": {}}, size=10)

# Print the documents
for hit in response["hits"]["hits"]:
    print(hit["_source"])

print(es.count(index="report_demo_2"))    

# Delete the index (use .options to avoid the deprecation warning)
# es.options(ignore_status=[400, 404]).indices.delete(index="report_demo")
# print("Data Deleted!")
