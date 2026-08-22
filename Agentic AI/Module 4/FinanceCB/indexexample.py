"""
rag_faiss.py
Minimal RAG pipeline:
- chunk documents sequentially (overlap)
- embed chunks with SentenceTransformers
- build FAISS index (cosine via IndexFlatIP + IndexIDMap)
- persist index + metadata
- simple retrieve + generate (OpenAI or local HF generator)
"""

import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# -----------------------
# Config / params
# -----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast
CHUNK_SIZE = 300         # tokens/characters per chunk heuristic (we use characters here)
CHUNK_OVERLAP = 50       # overlap between chunks
EMBED_BATCH = 64
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"
ID_DTYPE = np.int64

# -----------------------
# Utilities
# -----------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Sequential chunking with overlap. Uses character-based slicing (simple and deterministic).
    For production you may want token-based chunking (tiktoken) or sentence-boundary aware chunking.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= text_len:
            break
        start = end - overlap  # overlap
    return [c for c in chunks if c]  # drop empty

def prepare_corpus(docs: Dict[str, str]) -> Tuple[List[str], List[dict]]:
    """
    docs: dict mapping doc_id -> document_text
    returns:
      - list of chunk texts (insertion order)
      - metadata list: {id: int, doc_id: str, chunk_index: int, chunk_text: str}
    """
    chunk_texts = []
    metadatas = []
    next_id = 0
    for doc_id, text in docs.items():
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            chunk_texts.append(c)
            metadatas.append({
                "id": int(next_id),
                "doc_id": doc_id,
                "chunk_index": i,
                "text": c
            })
            next_id += 1
    return chunk_texts, metadatas

# -----------------------
# Embedding & FAISS
# -----------------------
class FaissRAG:
    def __init__(self, embed_model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(embed_model_name)
        self.index = None
        self.id_to_meta = {}  # id -> metadata dict

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # returns normalized embeddings (for cosine via inner product)
        embs = self.model.encode(texts, batch_size=EMBED_BATCH, convert_to_numpy=True, show_progress_bar=True)
        # normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embs = embs / norms
        return embs.astype("float32")

    def build_index(self, texts: List[str], metadatas: List[dict], overwrite: bool = True):
        """
        Build FAISS index from chunk texts and metadata.
        We'll use IndexFlatIP (inner product) with normalized vectors => cosine similarity.
        We'll wrap in IndexIDMap so we can persist custom ids that map to metadata.
        """
        embs = self.embed_texts(texts)
        d = embs.shape[1]
        base_index = faiss.IndexFlatIP(d)  # exact, inner-product
        id_index = faiss.IndexIDMap(base_index)  # allow custom integer ids
        ids = np.array([m["id"] for m in metadatas], dtype=ID_DTYPE)
        id_index.add_with_ids(embs, ids)
        self.index = id_index
        # store metadata mapping
        self.id_to_meta = {int(m["id"]): m for m in metadatas}
        print(f"Built index with {self.index.ntotal} vectors (dim={d}).")

    def save(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        if self.index is None:
            raise ValueError("Index is empty.")
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.id_to_meta, f)
        print(f"Saved index -> {index_path}, metadata -> {meta_path}")

    def load(self, index_path=FAISS_INDEX_PATH, meta_path=METADATA_PATH):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.id_to_meta = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.id_to_meta)} metadata entries.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Returns list of (metadata, score) sorted by score desc (cosine).
        """
        if self.index is None:
            raise ValueError("Index not loaded / built.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        q_emb = q_emb.astype("float32")
        scores, ids = self.index.search(q_emb, top_k)
        scores = scores[0]
        ids = ids[0]
        results = []
        for _id, score in zip(ids, scores):
            if _id == -1:
                continue
            meta = self.id_to_meta.get(int(_id), {"id": int(_id), "text": ""})
            results.append((meta, float(score)))
        return results

# -----------------------
# RAG prompt construction
# -----------------------
def construct_rag_prompt(question: str, retrieved: List[Tuple[dict, float]], max_chunks: int = 5) -> str:
    """
    Build a prompt that includes retrieved chunks as context.
    You can adjust the template to the LLM you use.
    """
    header = "You are a helpful assistant. Use the following extracted document snippets to answer the question. If the answer is not contained, say you don't know.\n\n"
    ctxs = []
    for i, (meta, score) in enumerate(retrieved[:max_chunks], start=1):
        ctxs.append(f"---\n[chunk_id: {meta['id']} | doc: {meta.get('doc_id','-')} | score: {score:.4f}]\n{meta['text']}\n")
    context = "\n".join(ctxs)
    prompt = f"{header}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely and cite the chunk_id(s) you used.\n"
    return prompt

# -----------------------
# OPTIONAL: call OpenAI (user must set OPENAI_API_KEY) - example only
# -----------------------
def generate_with_openai(prompt: str, max_tokens: int = 200):
    """
    Example wrapper — user must `pip install openai` and set OPENAI_API_KEY environment variable.
    This function just demonstrates how to send the RAG prompt. Not required for retrieval.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package not installed. `pip install openai`") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable to call OpenAI API.")
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # replace with an available model in your account
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"]

# -----------------------
# OPTIONAL: local generator with HuggingFace (flan-t5-small)
# -----------------------
def generate_with_local_model(prompt: str, max_length: int = 200):
    """
    Uses text2text-generation pipeline. Model download may be large and require GPU/CPU time.
    """
    from transformers import pipeline
    gen = pipeline("text2text-generation", model="google/flan-t5-small", device_map="auto" if os.getenv("USE_GPU") else None)
    out = gen(prompt, max_length=max_length, do_sample=False)[0]["generated_text"]
    return out

# -----------------------
# Example usage / demo
# -----------------------
if __name__ == "__main__":
    # --- 1) prepare a small demo corpus ---
    docs = {
        "doc1": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
        "doc2": "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species.",
        "doc3": "Dogs are domesticated mammals, not natural wild animals. They were originally bred from wolves."
    }

    # --- 2) chunk and prepare metadata ---
    texts, metas = prepare_corpus(docs)
    print(f"Prepared {len(texts)} chunks from {len(docs)} documents.")

    # --- 3) build the FAISS index ---
    rag = FaissRAG()
    rag.build_index(texts, metas)

    # --- optional: persist to disk ---
    rag.save()

    # --- 4) example query & retrieval ---
    question = "Which animal is domesticated mammals?"
    retrieved = rag.retrieve(question, top_k=4)
    for m, s in retrieved:
        print(f"> chunk_id={m['id']} doc={m['doc_id']} score={s:.4f}\n  {m['text'][:160]}...\n")

    # --- 5) construct RAG prompt ---
    prompt = construct_rag_prompt(question, retrieved, max_chunks=3)
    
    # --- 6) (OPTIONAL) generate answer ---
    # Option A: call OpenAI — uncomment if you set OPENAI_API_KEY
    answer = generate_with_openai(prompt)
    print("\nOpenAI answer:\n", answer)

    # Option B: local model (may download weights)
    # answer = generate_with_local_model(prompt)
    # print("\nLocal model answer:\n", answer)
