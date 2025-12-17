import chromadb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa
from collections import defaultdict
import json
import requests

# ======================================================
# LLM (Ollama) helper
# ======================================================

def llm(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"⚠ LLM unavailable ({e}), falling back")
        return ""


# ======================================================
# Paths & constants
# ======================================================

DB_PATH = "./chroma_db"
COLLECTION_NAME = "tabular_data"

MAX_LEGEND_ITEMS = 10   # max labels before grouping
BASE_DIR = Path("graphs")
DIR_2D = BASE_DIR / "2D"
DIR_3D = BASE_DIR / "3D"

DIR_2D.mkdir(parents=True, exist_ok=True)
DIR_3D.mkdir(parents=True, exist_ok=True)

# ======================================================
# Load data from Chroma
# ======================================================

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

data = collection.get(include=["embeddings", "documents", "metadatas"])
embeddings = np.array(data["embeddings"])
documents = data["documents"]
metadatas = data["metadatas"]

n = len(embeddings)
print(f"✓ Loaded {n} embeddings")

# ======================================================
# Auto-detect best label key from metadata
# ======================================================

def detect_label_key(metadatas):
    if not metadatas:
        return None

    stats = defaultdict(set)
    for meta in metadatas:
        if not meta:
            continue
        for k, v in meta.items():
            if isinstance(v, (str, bool)):
                stats[k].add(str(v))

    best_key, best_score = None, 0
    for k, values in stats.items():
        if 1 < len(values) < n * 0.6:
            if len(values) > best_score:
                best_key = k
                best_score = len(values)

    return best_key


LABEL_KEY = detect_label_key(metadatas)
print(f"✓ Label source: {LABEL_KEY or 'document-based'}")

# ======================================================
# Extract labels
# ======================================================

def extract_label(doc, meta):
    if LABEL_KEY and meta and LABEL_KEY in meta:
        return str(meta[LABEL_KEY])

    for part in doc.split("|"):
        if ":" in part:
            return part.split(":")[-1].strip()

    return "Unknown"


labels = np.array([
    extract_label(doc, meta)
    for doc, meta in zip(documents, metadatas)
])

unique_labels = sorted(set(labels))
print(f"✓ Found {len(unique_labels)} unique labels")

# ======================================================
# LLM-based grouping (only if needed)
# ======================================================

def group_labels_with_llm(labels):
    unique = sorted(set(labels))

    prompt = f"""
You are given many category labels from a dataset.

Task:
- Group them into a SMALL number of higher-level categories
- Each group must be semantically meaningful
- Do NOT invent new labels
- Return STRICT JSON only

Format:
{{
  "GroupName1": ["labelA", "labelB"],
  "GroupName2": ["labelC"]
}}

Labels:
{unique}
"""

    response = llm(prompt)
    if not response:
        return None

    try:
        groups = json.loads(response)
    except Exception:
        print("⚠ Failed to parse LLM output, skipping grouping")
        return None

    mapping = {}
    for group, members in groups.items():
        for m in members:
            mapping[m] = group

    return mapping


if len(unique_labels) > MAX_LEGEND_ITEMS:
    print(f"⚠ Too many labels ({len(unique_labels)}), grouping enabled")

    label_map = group_labels_with_llm(labels)

    if label_map:
        labels = np.array([
            label_map.get(l, "Other") for l in labels
        ])
        unique_labels = sorted(set(labels))
        print(f"✓ Reduced to {len(unique_labels)} grouped labels")
    else:
        print("⚠ Grouping skipped, legend may be long")

else:
    print("✓ Label count acceptable, no grouping needed")

# ======================================================
# t-SNE perplexity (auto)
# ======================================================

PERPLEXITY = min(30, max(5, n // 10))
print(f"✓ t-SNE perplexity set to {PERPLEXITY}")

# ======================================================
# Color map
# ======================================================

colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# ======================================================
# PCA 2D
# ======================================================

X_pca_2d = PCA(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(unique_labels):
    idx = labels == label
    plt.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1],
                alpha=0.7, label=label)

plt.title("PCA 2D – Embeddings")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(DIR_2D / "pca_2d.png", dpi=300)
plt.close()

# ======================================================
# PCA 3D
# ======================================================

X_pca_3d = PCA(n_components=3).fit_transform(embeddings)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for label in unique_labels:
    idx = labels == label
    ax.scatter(X_pca_3d[idx, 0],
               X_pca_3d[idx, 1],
               X_pca_3d[idx, 2],
               alpha=0.7)

ax.set_title("PCA 3D – Embeddings")
plt.tight_layout()
plt.savefig(DIR_3D / "pca_3d.png", dpi=300)
plt.close()

# ======================================================
# t-SNE 2D
# ======================================================

X_tsne_2d = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    random_state=42,
    init="random"
).fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for label in unique_labels:
    idx = labels == label
    plt.scatter(X_tsne_2d[idx, 0],
                X_tsne_2d[idx, 1],
                alpha=0.7,
                label=label)

plt.title("t-SNE 2D – Embeddings")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(DIR_2D / "tsne_2d.png", dpi=300)
plt.close()

# ======================================================
# t-SNE 3D
# ======================================================

X_tsne_3d = TSNE(
    n_components=3,
    perplexity=PERPLEXITY,
    random_state=42,
    init="random"
).fit_transform(embeddings)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for label in unique_labels:
    idx = labels == label
    ax.scatter(X_tsne_3d[idx, 0],
               X_tsne_3d[idx, 1],
               X_tsne_3d[idx, 2],
               alpha=0.7)

ax.set_title("t-SNE 3D – Embeddings")
plt.tight_layout()
plt.savefig(DIR_3D / "tsne_3d.png", dpi=300)
plt.close()

print(f"✓ Visualizations saved to {BASE_DIR}")
