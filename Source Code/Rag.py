# Install Required Libraries:
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np


# Load documents
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

file_path = "sample_document.pdf"  # Replace with your PDF

documents = load_documents(file_path)
print(f"Pages Loaded: {len(documents)}")


#  Fixed length chunking
def fixed_chunking(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

fixed_chunks = fixed_chunking(documents)
print(f"Fixed Chunks: {len(fixed_chunks)}")


# semantic chunking
def semantic_chunking(documents):
    semantic_chunks = []
    for doc in documents:
        paragraphs = doc.page_content.split("\n\n")
        for para in paragraphs:
            if len(para.strip()) > 50:
                semantic_chunks.append(para.strip())
    return semantic_chunks

semantic_chunks = semantic_chunking(documents)
print(f"Semantic Chunks: {len(semantic_chunks)}")



# embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_texts_and_embeddings(chunks, is_langchain=True):
    if is_langchain:
        texts = [c.page_content for c in chunks]
    else:
        texts = chunks

    embeddings = embedding_model.encode(texts)
    return texts, np.array(embeddings)

texts, embeddings = get_texts_and_embeddings(fixed_chunks)
print("Embeddings Created")


# vector database (faiss)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

print("FAISS Index Ready")


# keyword retrieval (BM25)
tokenized_corpus = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 Ready")


# vector retrieval 
def vector_retrieval(query, top_k=3):
    q_embed = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(q_embed), top_k)

    results = []
    for idx in indices[0]:
        results.append(texts[idx])
    return results


# keyword retrieval
def keyword_retrieval(query, top_k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(texts[idx])
    return results


# hybrid retrieval
def hybrid_retrieval(query, top_k=5):
    vector_docs = vector_retrieval(query, top_k=3)
    keyword_docs = keyword_retrieval(query, top_k=3)

    combined = list(dict.fromkeys(vector_docs + keyword_docs))
    return combined[:top_k]


# LLM generator
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)


# baseline LLM without rag
def baseline_llm(query):
    prompt = f"Answer the following question:\n\n{query}"
    result = generator(prompt)
    return result[0]["generated_text"]


# RAG response with retrieval
def rag_response(query):
    retrieved_docs = hybrid_retrieval(query)
    context = "\n".join(retrieved_docs)

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt)
    answer = result[0]["generated_text"]
    return answer, context


# Improved truthfulness score
def truthfulness_score(answer, context):
    answer_embedding = embedding_model.encode([answer])
    context_embedding = embedding_model.encode([context])

    similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
    return round(similarity * 100, 2)


# testing and comparison
query = "What is Retrieval-Augmented Generation?"

print("\n==============================")
print("USER QUERY:")
print(query)
print("==============================")

# Baseline
baseline_answer = baseline_llm(query)
print("\nBASELINE LLM ANSWER (No RAG):")
print(baseline_answer)

# RAG
rag_answer, retrieved_context = rag_response(query)
score = truthfulness_score(rag_answer, retrieved_context)

print("\nRAG ANSWER (With Hybrid Retrieval):")
print(rag_answer)

print("\nTRUTHFULNESS SCORE:")
print(f"{score}%")

print("\n==============================")
print("Research-grade RAG system executed successfully")
print("==============================")
