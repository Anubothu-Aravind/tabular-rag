# Tabular RAG

A Retrieval-Augmented Generation (RAG) system for querying tabular data using natural language. This project uses ChromaDB for vector storage, sentence transformers for embeddings, and Ollama for local LLM inference.

No API keys. No cloud dependency. Everything runs on your machine.


## Features

- **Semantic Search on Tables**: Query CSV / Excel files using natural language

- **Local LLM Inference**: Uses Ollama (llama3) entirely offline

- **Persistent Vector Store**: ChromaDB stores embeddings across runs

- **Tabular Reasoning**: Ask aggregation, filtering, comparison, and trend questions

- **Embedding Visualization (2D & 3D)**: PCA and t-SNE plots to inspect embedding quality and clustering


## Prerequisites

Make sure you have the following installed:

1. **Python 3.10+**
2. **uv** – fast Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Ollama** – local LLM runtime

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```


## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Anubothu-Aravind/tabular-rag.git
cd tabular-rag
```

### 2. Install Dependencies

Create a virtual environment and install dependencies using `uv`:

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Start Ollama

Run Ollama in a separate terminal:

```bash
ollama serve    # Start Ollama service
ollama pull llama3    # Pull the llama3 model (if not already downloaded)
```

### 4. Ingest Your Data

Load CSV or Excel data into ChromaDB:

```bash
python ingest.py
```

**Note**: By default, this ingests `data/vehicle_sales_data.csv`. To ingest a different file, modify the file path in [ingest.py](ingest.py#L58).

### 5. Query Your Data

Run the RAG agent to ask questions:

```bash
python rag_agent.py
```

To ask custom questions, modify the question in [rag_agent.py](rag_agent.py#L51) or use it as a module:

```python
from rag_agent import ask

response = ask("What were the total sales across all years?")
print(response)
```


## Embedding Visualization (2D & 3D)

This project includes tooling to visualize embeddings stored in ChromaDB, which is critical for debugging and validating RAG systems.

### What it does

* Loads embeddings directly from ChromaDB
* Applies:

  * PCA (2D & 3D) for global structure
  * t-SNE (2D & 3D) for local similarity
* Saves plots to disk (no UI popups)

### Run visualization

```bash
python visualize_embeddings.py
```

### Output structure

```
graphs/
├── 2D/
│   ├── pca_2d.png
│   └── tsne_2d.png
└── 3D/
    ├── pca_3d.png
    └── tsne_3d.png
```

Use these plots to:

* Verify semantic clustering
* Detect poor row-to-text formatting
* Debug retrieval issues before blaming the LLM


## 📁 Project Structure

```
tabular-rag/
├── data/
│   └── vehicle_sales_data.csv
├── chroma_db/
├── graphs/
│   ├── 2D/
│   └── 3D/
├── ingest.py
├── rag_agent.py
├── visualize_embeddings.py
├── llm.py
├── requirements.txt
└── README.md
```


## How It Works

### 1. Ingestion (`ingest.py`)

* Reads CSV / Excel files using pandas
* Converts each row into a semantic sentence
* Generates embeddings with `all-MiniLM-L6-v2`
* Stores embeddings in ChromaDB

### 2. Retrieval (`rag_agent.py`)

* Embeds the user query
* Retrieves top-k similar rows from ChromaDB
* Builds a grounded context for the LLM

### 3. Generation (`llm.py`)

* Sends the prompt to Ollama
* Returns a natural language response


## Example Queries

```python
ask("What were the total sales across all years?")
ask("Which vehicles had sales over 10,000 units?")
ask("Compare SUV sales vs Sedan sales in 2023")
ask("What is the sales trend over time?")
```


## Customization

### Change the LLM Model

Edit `llm.py`:

```python
payload = {
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False
}
```


### Adjust Retrieval Depth

In `rag_agent.py`:

```python
def retrieve_rows(query, k=25):
    ...
```


### Add Your Own Data

1. Place your CSV / Excel file in `data/`
2. Update the file path in `ingest.py`
3. Re-run ingestion:

   ```bash
   python ingest.py
   ```


## Troubleshooting

### Cannot connect to Ollama

```text
Error: Cannot connect to Ollama
```

Run:

```bash
ollama serve
```


### Model not found

```text
Error: model 'llama3' not found
```

Run:

```bash
ollama pull llama3
```


### Dependency issues

```bash
uv pip install -r requirements.txt --force-reinstall
```


## Dependencies

Key packages used:

- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `langchain-core` - Prompt templates
- `pandas` - Data manipulation
- `matplotlib` - creating static visualizations
- `scikit-learn` - for predictive data analysis
- `requests` - HTTP client for Ollama

See `requirements.txt` for the full list.


## References

- [Ollama](https://ollama.com/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [uv](https://github.com/astral-sh/uv) for fast Python package management
