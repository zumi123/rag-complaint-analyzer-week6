from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Load cleaned data
df = pd.read_csv("data/filtered_complaints.csv")

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
df['chunks'] = df['cleaned_text'].apply(lambda x: splitter.split_text(x))

# Flatten chunks
records = []
for idx, row in df.iterrows():
    for chunk in row['chunks']:
        records.append((row['Product'], row['Consumer complaint narrative'], chunk))

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([r[2] for r in records])

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save vector store
faiss.write_index(index, "vector_store/complaints.index")
