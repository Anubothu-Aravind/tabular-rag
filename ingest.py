import pandas as pd
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Embedding model (CPU friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma persistent DB - use PersistentClient instead of Client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tabular_data")


def row_to_text(row, columns):
    """
    Convert a table row into a semantic sentence
    """
    return " | ".join(f"{col}: {row[col]}" for col in columns)


def ingest_file(file_path):
    path = Path(file_path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV and Excel files are supported")

    df = df.fillna("")

    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        text = row_to_text(row, df.columns)
        documents.append(text)
        metadatas.append({
            "source": path.name
        })
        ids.append(str(uuid.uuid4()))

    embeddings = embedder.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # No need to call persist() - PersistentClient auto-persists
    print(f"✓ Ingested {len(documents)} rows from {path.name}")


if __name__ == "__main__":
    ingest_file("data/pokemon.csv")     # change file if needed