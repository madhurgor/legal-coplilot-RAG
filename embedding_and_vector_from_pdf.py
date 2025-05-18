from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])


# Example extracted text
pdf_text = extract_text_from_pdf("/Users/mgor/Downloads/case-files/level-2/AHMEDABAD_C-4-2011_27-09-2019.pdf")

# Step 1: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings
embeddings = model.encode(pdf_text, convert_to_numpy=True)

print(embeddings.tolist())
