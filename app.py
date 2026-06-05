import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Config
st.set_page_config(page_title="Semantic Search Engine")

st.title("🔍 Semantic Search Engine")

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load Documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

# Generate Embeddings
doc_embeddings = model.encode(documents)

# User Query
query = st.text_input("Enter your search query")

if query:

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:5]

    st.subheader("Top Results")

    for idx in top_indices:
        st.write(f"📄 {documents[idx]}")
        st.write(f"Similarity Score: {similarities[idx]:.4f}")
        st.divider()