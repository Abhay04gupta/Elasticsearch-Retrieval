import streamlit as st
# from pymilvus import MilvusClient


@st.cache_resource
# def get_milvus_client(uri: str, token: str = None) -> MilvusClient:
#     client = MilvusClient(uri=uri, token=token)
#     client.using_database("tata_db")  # Switch to tata_db
#     return client


# def create_collection(
#     milvus_client: MilvusClient, collection_name: str, dim: int, drop_old: bool = True
# ):
#     if milvus_client.has_collection(collection_name) and drop_old:
#         milvus_client.drop_collection(collection_name)
#     if milvus_client.has_collection(collection_name):
#         raise RuntimeError(
#             f"Collection {collection_name} already exists. Set drop_old=True to create a new one instead."
#         )
#     return milvus_client.create_collection(
#         collection_name=collection_name,
#         dimension=dim,
#         metric_type="COSINE",
#         consistency_level="Strong",
#         auto_id=True,
#     )

def get_search_results(_es_client, index_name, query_vector, return_fields, k):
    # Elasticsearch script_score-based cosine similarity query
    query = {
        "size": k,
        "_source": return_fields,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    # Execute search
    response = _es_client.search(index=index_name, body=query)

    hits = response["hits"]["hits"]

    # Extract max score for normalization
    max_score = max(hit["_score"] for hit in hits) if hits else 1.0

    results = []
    for hit in hits:
        normalized_score = hit["_score"] / max_score if max_score > 0 else 0.0
        distance = 1.0 - normalized_score  # Lower means more similar

        result = {
            "entity": hit["_source"],
            "distance": distance
        }
        results.append(result)

    return [results]
