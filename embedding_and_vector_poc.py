from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Step 1: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Your data (list of text)
documents = [
    "What is the capital of France?",
    "Machine learning is fun.",
    "The Eiffel Tower is in Paris.",
    "Deep learning requires data."
]

# Step 3: Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

print("type of embeddings: ", type(embeddings))
print(embeddings.tolist())

with open("embeddings.txt", "w") as f:
    for text, emb in zip(documents, embeddings):
        f.write(f"{text}\n{emb.tolist()}\n\n")

# Step 4: Save to FAISS index
dimension = embeddings.shape[1]
print("mgor dimension: ", dimension)
print("mgor list: ", embeddings.shape[0])
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)

index.add(embeddings)

# Optional: Save index and data to disk
faiss.write_index(index, "my_index.faiss")
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Embeddings saved to FAISS index.")
