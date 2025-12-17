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

POKEMON_PROMPT = PromptTemplate.from_template("""
You are an expert Pokémon team strategist and data analyst.

You are given rows from a Pokémon dataset.
Each row is structured as column-value pairs (Name, Type 1, Type 2, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Total, Legendary, etc).

Rules:
- Select exactly 6 Pokémon
- Filter rows based on constraints in the question
- Use stats to assign roles (attacker, tank, speed, support)
- Consider type coverage and avoid unnecessary overlaps
- Perform comparisons and calculations when needed
- If the dataset does not support a requirement, say so clearly

DATA:
{context}

QUESTION:
{question}

First explain your reasoning step by step.
Then give the final 6 Pokémon team as a clear list.
""")



def retrieve_rows(query, k=30):
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

def pokemon_ask(question: str):
    rows = retrieve_rows(question)
    context = "\n".join(rows)

    prompt = POKEMON_PROMPT.format(
        context=context,
        question=question
    )

    return llm(prompt)


if __name__ == "__main__":
    # response = ask(
    #     "What were the total sales across all years?"
    # )
    response = ask(
        "Which vehicles had sales over 10,000 units?"
    )
    print(response)
    print("\n" + "="*50 + "\n")
    print()
    print("" + "="*50 + "\n")
    response = pokemon_ask(
    """
    Build a balanced team of 6 Pokémon using this dataset.

    Constraints:
    - Maximize type coverage (avoid repeating primary types when possible)
    - Include at least one defensive Pokémon and one fast attacker
    - Prefer higher total base stats
    - Do not include Legendary Pokémon

    For each Pokémon, explain:
    - Why it was chosen
    - Its role in the team (attacker, tank, support, speed)
    """
)
    print(response)