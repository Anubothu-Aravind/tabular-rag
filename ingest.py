import pandas as pd
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from typing import List, Optional

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


def normalize_column_name(col: str) -> str:
    """
    Normalize column names to snake_case for metadata keys
    Examples: 'Sp. Atk' -> 'sp_atk', 'Type 1' -> 'type_1'
    """
    return col.lower().replace(' ', '_').replace('.', '').replace('-', '_')


def ingest_file(file_path: str, collection_name: Optional[str] = None):
    """
    Dynamically ingest CSV or Excel files into ChromaDB
    
    Args:
        file_path: Path to the file to ingest
        collection_name: Optional custom collection name (defaults to filename without extension)
    """
    path = Path(file_path)

    # Validate file exists
    if not path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return

    # Read file based on extension
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV and Excel files are supported")

    # Use custom collection name or derive from filename
    coll_name = collection_name or path.stem
    coll = client.get_or_create_collection(name=coll_name)

    # Fill NaN values
    df = df.fillna("")

    documents = []
    metadatas = []
    ids = []

    print(f"📊 Processing {len(df)} rows from {path.name}...")
    print(f"📋 Columns: {df.columns.tolist()}")

    for idx, row in df.iterrows():
        # Generate semantic text representation
        text = row_to_text(row, df.columns)
        documents.append(text)

        # Dynamically build metadata from all columns
        meta = {"source": path.name, "row_index": int(idx)}
        
        for col in df.columns:
            # Normalize column name to snake_case for metadata keys
            key = normalize_column_name(col)
            value = row[col]
            
            # Handle different data types intelligently
            if pd.isna(value) or value == "":
                meta[key] = ""
            elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                meta[key] = int(value)
            elif df[col].dtype in ['float64', 'float32', 'float16']:
                meta[key] = float(value)
            elif df[col].dtype == 'bool':
                meta[key] = bool(value)
            else:
                meta[key] = str(value)
        
        metadatas.append(meta)
        ids.append(str(uuid.uuid4()))

    # Generate embeddings with progress indication
    print(f"🔄 Generating embeddings...")
    embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

    # Add to ChromaDB
    print(f"💾 Adding to ChromaDB collection '{coll_name}'...")
    coll.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # No need to call persist() - PersistentClient auto-persists
    print(f"✅ Successfully ingested {len(documents)} rows from {path.name}")
    print(f"   Collection: {coll_name}")
    print(f"   Database: ./chroma_db")
    print()


def ingest_multiple_files(file_paths: List[str], separate_collections: bool = True):
    """
    Ingest multiple files at once
    
    Args:
        file_paths: List of file paths to ingest
        separate_collections: If True, each file gets its own collection.
                            If False, all files go into 'tabular_data' collection.
    """
    print(f"🚀 Starting ingestion of {len(file_paths)} file(s)...\n")
    
    for file_path in file_paths:
        try:
            if separate_collections:
                ingest_file(file_path)
            else:
                # All files go into default 'tabular_data' collection
                path = Path(file_path)
                ingest_file(file_path, collection_name="tabular_data")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}\n")
            continue
    
    print("🎉 Ingestion complete!")


def list_collections():
    """
    List all collections in the ChromaDB
    """
    collections = client.list_collections()
    print(f"\n📚 Available Collections ({len(collections)}):")
    for coll in collections:
        count = coll.count()
        print(f"   - {coll.name}: {count} documents")


if __name__ == "__main__":
    # Check if files are provided via command line
    if len(sys.argv) > 1:
        # Usage: python ingest.py file1.csv file2.xlsx
        files = sys.argv[1:]
        ingest_multiple_files(files, separate_collections=True)
    else:
        # Default behavior: ingest specific files into shared collection
        files_to_ingest = [
            "data/pokemon.csv",
            "data/vehicle_sales_data.csv"
        ]
        
        ingest_multiple_files(files_to_ingest, separate_collections=False)
    
    # Show summary of collections
    list_collections()