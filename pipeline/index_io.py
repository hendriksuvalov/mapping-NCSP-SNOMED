"""Load a saved embedding index, embed query texts, and search the index."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def load_index(index_dir: Path, dimensions: int | None = None):
    """Return (concepts, matrix, manifest). With `dimensions`, keep the first N dimensions and re-normalise,
    which for text-embedding-3 models matches requesting that size from the API."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    concepts = pd.read_csv(index_dir / "concepts.csv", dtype=str, keep_default_na=False)
    matrix = np.load(index_dir / "embeddings.npy")
    if dimensions:
        if dimensions > matrix.shape[1]:
            raise ValueError(f"index has {matrix.shape[1]} dimensions, asked for {dimensions}")
        matrix = normalise(matrix[:, :dimensions])
    return concepts, matrix, manifest


def embed_texts(client, deployment: str, texts: list[str], dimensions: int | None = None,
                batch_size: int = 500) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(input=batch, model=deployment)
        if len(response.data) != len(batch):
            raise RuntimeError(f"got {len(response.data)} embeddings for {len(batch)} inputs")
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    matrix = np.asarray(vectors, dtype=np.float32)
    return normalise(matrix[:, :dimensions]) if dimensions else matrix


def search(matrix: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k rows by cosine similarity (rows are unit vectors). Returns (indices, scores), best first."""
    scores = queries @ matrix.T
    k = min(k, matrix.shape[0])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    return top, np.take_along_axis(scores, top, axis=1)
