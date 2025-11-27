from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("QDRANT_URL"))
print(os.getenv("QDRANT_API_KEY"))

# establish connection to the qdrant cluster
client = QdrantClient(url = os.getenv("QDRANT_URL"), api_key = os.getenv("QDRANT_API_KEY"))

# Create a collection
collection_name = 'my_first_collection'
vector_size = 4
vector_distance = models.Distance.COSINE

if not client.collection_exists(collection_name = collection_name):
    client.create_collection(
        collection_name = collection_name,
        vectors_config = models.VectorParams(
            size = vector_size,
            distance = vector_distance
        )
    )
else:
    print(f"Collection '{collection_name}' already exists, skipping creation.")

print(client.get_collections())

collection_info = client.get_collection(collection_name = collection_name)
payload_schema = getattr(collection_info, "payload_schema", None)

if not payload_schema or "category" not in payload_schema:
    client.create_payload_index(
        collection_name = collection_name,
        field_name = "category",
        field_schema = models.PayloadSchemaType.KEYWORD
    )

# points

# core data entity in qdrant

# 1. ID
# 2. vector data
# 3. payload


points = [
    models.PointStruct(
        id = 1,
        vector = [0.1, 0.2, 0.3, 0.4],
        payload = {"category": "example"}
    ),
    models.PointStruct(
        id = 2,
        vector = [0.2, 0.3, 0.4, 0.5],
        payload={"category": "demo"}
    )
]

client.upsert(
    collection_name = collection_name,
    points = points
)


# similarity search

query_vector = [0.08, 0.14, 0.33, 0.28]

search_results = client.query_points(
    collection_name = collection_name,
    query = query_vector,
    limit = 1
)

print(search_results)

# filter results
print("after filtering results")
search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="test")
        )
    ]
)

search_results = client.query_points(
    collection_name = collection_name,
    query = query_vector,
    query_filter = search_filter,
    limit = 1
)

print(search_results)
