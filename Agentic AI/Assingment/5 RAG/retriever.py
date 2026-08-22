import pandas as pd
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.cohere import CohereEmbedding

def load_competitor_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic cleaning
    df = df.fillna("")
    # Optional: normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Competitor Name: {row.get('Competitor Name','')}\n"
            f"Product Description: {row.get('Product Description','')}\n"
            f"Marketing Strategy: {row.get('Marketing Strategy','')}\n"
            f"Financial Summary: {row.get('Financial Summary','')}"
        )
        metadata = {"competitor": row.get("Competitor Name","")}
        docs.append(Document(text=text, metadata=metadata))
    return docs

def build_index(docs: List[Document], cohere_api_key: str) -> VectorStoreIndex:
    # Configure Cohere embeddings globally for this index
    Settings.embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0"
    )
    return VectorStoreIndex.from_documents(docs)
    def load_competitor_df(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        # Basic cleaning
        df = df.fillna("")
        # Optional: normalize column names
        df.columns = [c.strip() for c in df.columns]
        return df

    def df_to_documents(df: pd.DataFrame) -> List[Document]:
        docs = []
        for _, row in df.iterrows():
            text = (
                f"Competitor Name: {row.get('Competitor Name','')}\n"
                f"Product Description: {row.get('Product Description','')}\n"
                f"Marketing Strategy: {row.get('Marketing Strategy','')}\n"
                f"Financial Summary: {row.get('Financial Summary','')}"
            )
            metadata = {"competitor": row.get("Competitor Name","")}
            docs.append(Document(text=text, metadata=metadata))
        return docs

    def build_index(docs: List[Document], cohere_api_key: str) -> VectorStoreIndex:
        # Configure Cohere embeddings globally for this index
        Settings.embed_model = CohereEmbedding(
            api_key=cohere_api_key,
            model_name="embed-english-v3.0"
        )
        return VectorStoreIndex.from_documents(docs)
