# Tabular RAG

A Retrieval-Augmented Generation (RAG) system for querying tabular data using natural language. This project uses ChromaDB for vector storage, sentence transformers for embeddings, and Ollama for local LLM inference.

## Features

- **Semantic Search**: Query CSV/Excel files using natural language
- **Local LLM**: Uses Ollama (llama3) for inference - no API keys needed
- **Vector Storage**: ChromaDB for persistent embedding storage
- **Data Analysis**: Ask complex questions about your tabular data

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.10+**
2. **uv** - Fast Python package manager
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** - Local LLM runtime
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

Using `uv` to create a virtual environment and install dependencies:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all required packages
uv pip install -r requirements.txt
```

### 3. Start Ollama

In a separate terminal, ensure Ollama is running:

```bash
# Start Ollama service
ollama serve

# Pull the llama3 model (if not already downloaded)
ollama pull llama3
```

### 4. Ingest Your Data

Load your CSV or Excel data into the vector database:

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

response = ask("What were the total sales in 2023?")
print(response)
```

## 📁 Project Structure

```
tabular-rag/
├── data/
│   └── vehicle_sales_data.csv    # Sample dataset
├── chroma_db/                     # Vector database (auto-created)
├── ingest.py                      # Data ingestion script
├── rag_agent.py                   # Query interface
├── llm.py                         # Ollama integration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How It Works

1. **Ingestion** ([ingest.py](ingest.py)):
   - Reads CSV/Excel files using pandas
   - Converts each row into a semantic text representation
   - Generates embeddings using `all-MiniLM-L6-v2`
   - Stores embeddings in ChromaDB

2. **Retrieval** ([rag_agent.py](rag_agent.py)):
   - Takes a natural language question
   - Generates query embedding
   - Retrieves top-k similar rows from ChromaDB
   - Passes context + question to LLM

3. **Generation** ([llm.py](llm.py)):
   - Sends prompt to local Ollama instance
   - Returns natural language response

## Example Queries

```python
from rag_agent import ask

# Aggregation
ask("What were the total sales across all years?")

# Filtering
ask("Which vehicles had sales over 10,000 units?")

# Comparison
ask("Compare SUV sales vs Sedan sales in 2023")

# Trend analysis
ask("What's the sales trend for electric vehicles?")
```

## Customization

### Change the LLM Model

Edit [llm.py](llm.py#L11) to use a different Ollama model:

```python
payload = {
    "model": "llama3.2",  # or mistral, codellama, etc.
    "prompt": prompt,
    "stream": False
}
```

### Adjust Retrieval Results

Modify the number of rows retrieved in [rag_agent.py](rag_agent.py#L36):

```python
def retrieve_rows(query, k=25):  # Change k value
    ...
```

### Add Your Own Data

1. Place your CSV/Excel file in the `data/` directory
2. Update the file path in [ingest.py](ingest.py#L58)
3. Run `python ingest.py` to index the new data

## Troubleshooting

### Ollama Connection Error

```
Error: Cannot connect to Ollama
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Missing Model Error

```
Error: model 'llama3' not found
```

**Solution**: Pull the model:
```bash
ollama pull llama3
```

### Import Errors

**Solution**: Reinstall dependencies:
```bash
uv pip install -r requirements.txt --force-reinstall
```

## Dependencies

Key packages used:
- `chromadb` - Vector database
- `sentence-transformers` - Embedding generation
- `langchain-core` - Prompt templates
- `pandas` - Data manipulation
- `requests` - HTTP client for Ollama

See [requirements.txt](requirements.txt) for the complete list.

## References

- [Ollama](https://ollama.com/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [uv](https://github.com/astral-sh/uv) for fast Python package management
