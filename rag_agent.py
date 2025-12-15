import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from llm import llm

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma DB - use PersistentClient to match ingest.py
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="tabular_data")

PROMPT = PromptTemplate.from_template("""
You are an expert data analyst.

You are given rows from a table.
Each row is structured as column-value pairs.

Rules:
- Filter rows when required
- Perform calculations if needed
- Compare values across rows
- Explain your reasoning
- If data is insufficient, say so clearly

DATA:
{context}

QUESTION:
{question}

Answer step by step, then give the final answer.
""")


def retrieve_rows(query, k=25):
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


def ask(question: str):
    rows = retrieve_rows(question)
    context = "\n".join(rows)

    prompt = PROMPT.format(
        context=context,
        question=question
    )

    return llm(prompt)


if __name__ == "__main__":
    response = ask(
        "What were the total sales across all years?"
    )
    print(response)