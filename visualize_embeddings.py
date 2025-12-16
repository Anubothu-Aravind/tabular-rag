import chromadb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path("graphs")
DIR_2D = BASE_DIR / "2D"
DIR_3D = BASE_DIR / "3D"

DIR_2D.mkdir(parents=True, exist_ok=True)
DIR_3D.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load embeddings from Chroma
# -------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="tabular_data")

data = collection.get(include=["embeddings", "documents"])
embeddings = np.array(data["embeddings"])
documents = data["documents"]

print(f"Loaded {len(embeddings)} embeddings")

# -------------------------------
# Extract label (Vehicle Type)
# -------------------------------
def extract_vehicle(doc):
    for part in doc.split("|"):
        if "Vehicle" in part:
            return part.split(":")[1].strip()
    return "Unknown"

labels = np.array([extract_vehicle(doc) for doc in documents])
unique_labels = sorted(set(labels))

# -------------------------------
# Helper: color mapping
# -------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
color_map = dict(zip(unique_labels, colors))

# -------------------------------
# 2D PCA
# -------------------------------
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for label in unique_labels:
    idx = labels == label
    plt.scatter(
        X_pca_2d[idx, 0],
        X_pca_2d[idx, 1],
        label=label,
        alpha=0.75
    )

plt.title("PCA 2D – Tabular Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(DIR_2D / "pca_2d.png", dpi=300)
plt.close()

# -------------------------------
# 3D PCA
# -------------------------------
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(embeddings)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for label in unique_labels:
    idx = labels == label
    ax.scatter(
        X_pca_3d[idx, 0],
        X_pca_3d[idx, 1],
        X_pca_3d[idx, 2],
        label=label,
        alpha=0.75
    )

ax.set_title("PCA 3D – Tabular Embeddings")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.savefig(DIR_3D / "pca_3d.png", dpi=300)
plt.close()

# -------------------------------
# 2D t-SNE
# -------------------------------
tsne_2d = TSNE(
    n_components=2,
    perplexity=5,
    random_state=42,
    init="random"
)

X_tsne_2d = tsne_2d.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for label in unique_labels:
    idx = labels == label
    plt.scatter(
        X_tsne_2d[idx, 0],
        X_tsne_2d[idx, 1],
        label=label,
        alpha=0.75
    )

plt.title("t-SNE 2D – Tabular Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.tight_layout()
plt.savefig(DIR_2D / "tsne_2d.png", dpi=300)
plt.close()

# -------------------------------
# 3D t-SNE
# -------------------------------
tsne_3d = TSNE(
    n_components=3,
    perplexity=5,
    random_state=42,
    init="random"
)

X_tsne_3d = tsne_3d.fit_transform(embeddings)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for label in unique_labels:
    idx = labels == label
    ax.scatter(
        X_tsne_3d[idx, 0],
        X_tsne_3d[idx, 1],
        X_tsne_3d[idx, 2],
        label=label,
        alpha=0.75
    )

ax.set_title("t-SNE 3D – Tabular Embeddings")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_zlabel("Dim 3")
ax.legend()
plt.tight_layout()
plt.savefig(DIR_3D / "tsne_3d.png", dpi=300)
plt.close()

print("✓ Visualizations saved in /graphs/2D and /graphs/3D")
