"""Qdrant: one collection per QA collection, every representation a named vector.

Holding all representations on the same points means every retrieval method
searches exactly the same items. Child chunks for the parent-document
retriever live in a sibling collection, `<name>__chunks`.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    ExtendedPointId,
    Modifier,
    PointIdsList,
    PointStruct,
    PointVectors,
    Record,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ragqa_core.types import Hit, Item

VECTORS = (
    "text_dense",
    "kw_all",
    "kw_individual",
    "kw_topic",
    "kw_method",
    "kw_common",
    "kw_summary",
)
SPARSE = "text_sparse"
CHUNK = "chunk_dense"


def point_id(name: str, item_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{item_id}"))


def chunks_of(name: str) -> str:
    return f"{name}__chunks"


def to_hit(point: ScoredPoint | Record, score: float = 0.0) -> Hit:
    p = point.payload or {}
    return Hit(
        id=p["item_id"],
        text=p["text"],
        score=float(getattr(point, "score", score) or score),
        keywords=p.get("keywords") or {},
        matched=[],
        answer=p.get("answer"),
        explanation=p.get("explanation"),
        meta=p.get("meta") or {},
        summary=p.get("summary"),
    )


class Store:
    def __init__(self, url: str | None, path: Path | None) -> None:
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=":memory:")

    def create(self, name: str, dim: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config={
                    v: VectorParams(size=dim, distance=Distance.COSINE) for v in VECTORS
                },
                sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
            )
        if not self.client.collection_exists(chunks_of(name)):
            self.client.create_collection(
                chunks_of(name),
                vectors_config={CHUNK: VectorParams(size=dim, distance=Distance.COSINE)},
            )

    def drop(self, name: str) -> None:
        for collection in (name, chunks_of(name)):
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)

    def count(self, name: str) -> int:
        return self.client.count(name, exact=True).count

    def upsert_items(
        self,
        name: str,
        items: list[Item],
        dense: np.ndarray,
        sparse: list[SparseVector],
        summaries: list[str | None],
    ) -> None:
        points = [
            PointStruct(
                id=point_id(name, item.id),
                vector={"text_dense": vec.tolist(), SPARSE: sp},
                payload={
                    "item_id": item.id,
                    "text": item.text,
                    "answer": item.answer,
                    "explanation": item.explanation,
                    "meta": item.meta,
                    "summary": summary,
                    "keywords": {},
                },
            )
            for item, vec, sp, summary in zip(items, dense, sparse, summaries, strict=True)
        ]
        self.client.upsert(name, points=points)

    def upsert_chunks(
        self, name: str, parent_ids: list[str], texts: list[str], vecs: np.ndarray
    ) -> None:
        seen: dict[str, int] = {}
        points = []
        for parent, text, vec in zip(parent_ids, texts, vecs, strict=True):
            n = seen[parent] = seen.get(parent, -1) + 1
            points.append(
                PointStruct(
                    id=point_id(name, f"{parent}#{n}"),
                    vector={CHUNK: vec.tolist()},
                    payload={"parent_id": parent, "text": text},
                )
            )
        self.client.upsert(chunks_of(name), points=points)

    def set_keywords(
        self,
        name: str,
        ids: list[str],
        keywords: list[dict],
        vectors: dict[str, np.ndarray],
    ) -> None:
        pids: list[ExtendedPointId] = [point_id(name, i) for i in ids]
        stale = [v for v in VECTORS if v != "text_dense" and v not in vectors]
        if stale:
            self.client.delete_vectors(name, vectors=stale, points=PointIdsList(points=pids))
        if vectors:
            self.client.update_vectors(
                name,
                points=[
                    PointVectors(
                        id=pid, vector={v: vecs[row].tolist() for v, vecs in vectors.items()}
                    )
                    for row, pid in enumerate(pids)
                ],
            )
        for pid, kw in zip(pids, keywords, strict=True):
            self.client.set_payload(name, payload={"keywords": kw}, points=[pid])

    def search(self, name: str, using: str, vector: np.ndarray, limit: int) -> list[Hit]:
        result = self.client.query_points(
            name, query=vector.tolist(), using=using, limit=limit, with_payload=True
        )
        return [to_hit(p) for p in result.points]

    def search_sparse(self, name: str, vector: SparseVector, limit: int) -> list[Hit]:
        result = self.client.query_points(
            name, query=vector, using=SPARSE, limit=limit, with_payload=True
        )
        return [to_hit(p) for p in result.points]

    def search_chunks(self, name: str, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        result = self.client.query_points(
            chunks_of(name), query=vector.tolist(), using=CHUNK, limit=limit, with_payload=True
        )
        return [((p.payload or {})["parent_id"], float(p.score)) for p in result.points]

    def get(self, name: str, ids: list[str]) -> list[Hit]:
        records = self.client.retrieve(name, [point_id(name, i) for i in ids], with_payload=True)
        by_id = {to_hit(r).id: to_hit(r) for r in records}
        return [by_id[i] for i in ids if i in by_id]

    def ids(self, name: str, batch: int = 1024) -> list[str]:
        out: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                name, limit=batch, offset=offset, with_payload=["item_id"]
            )
            out.extend((r.payload or {})["item_id"] for r in records)
            if offset is None:
                return sorted(out)

    def scroll(self, name: str, batch: int = 256) -> Iterator[list[Hit]]:
        offset = None
        while True:
            records, offset = self.client.scroll(
                name, limit=batch, offset=offset, with_payload=True
            )
            if records:
                yield [to_hit(r) for r in records]
            if offset is None:
                return
