from pymilvus import MilvusClient
from pymilvus import DataType
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to Zilliz Cloud
uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN")

print(uri, token)

milvus_client = MilvusClient(uri=uri, token=token)
print(f"Connected to DB: {milvus_client} successfully")
