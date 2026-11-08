"""Qdrant: one collection per QA collection, every representation a named vector.

Holding all representations on the same points means every retrieval method
searches exactly the same items. Child chunks for the parent-document
retriever live in a sibling collection, `<name>__chunks`.
"""

import hashlib
import json
import uuid
from collections.abc import Collection, Iterator
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    ExtendedPointId,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    Modifier,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    PointVectors,
    Record,
    ScoredPoint,
    SetPayload,
    SetPayloadOperation,
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
    "kw_tags",
)
SPARSE = "text_sparse"
CHUNK = "chunk_dense"


def point_id(name: str, item_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{item_id}"))


def chunks_of(name: str) -> str:
    return f"{name}__chunks"


def rebuild_of(name: str) -> str:
    """The side collection a store is re-embedded into before it switches; its chunks
    live in chunks_of(rebuild_of(name))."""
    return f"{name}__rebuild"


def item_hash(item: Item) -> str:
    """What decides whether an item changed. A text chunk's place in its file, `part` and
    `parts` (a meta with `file` and `part`), is left out: an edit that moves a chunk or
    changes the file's count re-indexes only the chunks whose text changed (the loader
    writes the new place as a payload). Any other item's `part` and `parts` are content
    like the rest of its meta."""
    chunk = "file" in item.meta and "part" in item.meta
    meta = {k: v for k, v in item.meta.items() if not (chunk and k in ("part", "parts"))}
    content = [item.text, item.answer, item.explanation, meta]
    return hashlib.sha1(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def skip_filter(
    skip: tuple[str, ...], excluded: HasIdCondition | FieldCondition | None = None
) -> Filter | None:
    """Points without a meta.source never match must_not, so they are never skipped.
    `excluded` leaves out items by id, as items_excluded or parents_excluded builds it."""
    must_not: list = [] if excluded is None else [excluded]
    if skip:
        must_not.append(FieldCondition(key="meta.source", match=MatchAny(any=list(skip))))
    return Filter(must_not=must_not) if must_not else None


def items_excluded(name: str, exclude: Collection[str]) -> HasIdCondition | None:
    """The items with these ids, by point id: Qdrant needs no payload index for it."""
    if not exclude:
        return None
    return HasIdCondition(has_id=[point_id(name, i) for i in sorted(exclude)])


def parents_excluded(exclude: Collection[str]) -> FieldCondition | None:
    """The chunks of the items with these ids, by the indexed parent_id."""
    if not exclude:
        return None
    return FieldCondition(key="parent_id", match=MatchAny(any=sorted(exclude)))


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
        tags=p.get("tags") or [],
        abstracted=p.get("abstracted"),
    )


class Store:
    def __init__(self, url: str | None, path: Path | None) -> None:
        self.server = bool(url)
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
        self.ensure_indexes(name)

    def ensure_indexes(self, name: str) -> None:
        """Keyword indexes make the source filters fast; local Qdrant ignores them and warns."""
        if not self.server:
            return
        wanted = {name: ["meta.source"], chunks_of(name): ["parent_id", "meta.source"]}
        for collection, keys in wanted.items():
            for key in keys:
                self.client.create_payload_index(
                    collection, field_name=key, field_schema=PayloadSchemaType.KEYWORD
                )

    def drop(self, name: str) -> None:
        """Deletes the store, its chunks, and any rebuild it left part-way."""
        self._delete(name, chunks_of(name), rebuild_of(name), chunks_of(rebuild_of(name)))

    def drop_rebuild(self, name: str) -> None:
        self._delete(rebuild_of(name), chunks_of(rebuild_of(name)))

    def recreate(self, name: str, dim: int) -> None:
        """The store and its chunks, empty, at `dim` dimensions; a rebuild stays."""
        self._delete(name, chunks_of(name))
        self.create(name, dim)

    def _delete(self, *collections: str) -> None:
        for collection in collections:
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)

    def points(self, collection: str, batch: int = 256) -> Iterator[list[tuple[str, dict, dict]]]:
        """Every point as (point id, vectors by name, payload), a batch at a time."""
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection, limit=batch, offset=offset, with_payload=True, with_vectors=True
            )
            if records:
                yield [
                    (
                        str(r.id),
                        dict(r.vector) if isinstance(r.vector, dict) else {},
                        r.payload or {},
                    )
                    for r in records
                ]
            if offset is None:
                return

    def chunk_points(self, name: str, parent_ids: list[str]) -> list[tuple[str, dict]]:
        """(point id, payload) of the chunks of these items, without their vectors."""
        of_parents = Filter(must=[FieldCondition(key="parent_id", match=MatchAny(any=parent_ids))])
        out: list[tuple[str, dict]] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                chunks_of(name),
                limit=1024,
                offset=offset,
                with_payload=True,
                scroll_filter=of_parents,
            )
            out.extend((str(r.id), r.payload or {}) for r in records)
            if offset is None:
                return out

    def write(self, collection: str, points: list[tuple[str, dict, dict]]) -> None:
        """Upserts (point id, vectors by name, payload) with their own ids; a dense vector
        may be a numpy row."""
        if not points:
            return
        self.client.upsert(
            collection,
            points=[
                PointStruct(
                    id=pid,
                    vector={
                        v: vec.tolist() if isinstance(vec, np.ndarray) else vec
                        for v, vec in vectors.items()
                    },
                    payload=payload,
                )
                for pid, vectors, payload in points
            ],
        )

    def copy(self, src: str, dst: str) -> int:
        """Every point of `src`, vectors and payload, into `dst` under the same id."""
        n = 0
        for batch in self.points(src):
            self.write(dst, batch)
            n += len(batch)
        return n

    def exists(self, name: str) -> bool:
        return self.client.collection_exists(name)

    def vector_names(self, name: str) -> set[str]:
        """The dense vectors a collection was created with: older ones lack the newer names."""
        vectors = self.client.get_collection(name).config.params.vectors
        return set(vectors) if isinstance(vectors, dict) else set()

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
        self,
        name: str,
        parent_ids: list[str],
        texts: list[str],
        vecs: np.ndarray,
        sources: list[str | None],
    ) -> None:
        seen: dict[str, int] = {}
        points = []
        for parent, text, vec, source in zip(parent_ids, texts, vecs, sources, strict=True):
            n = seen[parent] = seen.get(parent, -1) + 1
            points.append(
                PointStruct(
                    id=point_id(name, f"{parent}#{n}"),
                    vector={CHUNK: vec.tolist()},
                    payload={"parent_id": parent, "text": text, "meta": {"source": source}},
                )
            )
        self.client.upsert(chunks_of(name), points=points)

    def delete_chunks(self, name: str, parent_ids: list[str]) -> None:
        self.client.delete(
            chunks_of(name),
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="parent_id", match=MatchAny(any=parent_ids))]
                )
            ),
        )

    def set_keywords(
        self,
        name: str,
        ids: list[str],
        keywords: list[dict],
        vectors: dict[str, np.ndarray],
    ) -> None:
        pids: list[ExtendedPointId] = [point_id(name, i) for i in ids]
        # ONLY THE LISTS' VECTORS: TAGS ARE NOT KEYWORDS, AND A VECTOR THE COLLECTION
        # WAS NEVER CREATED WITH CANNOT BE DELETED.
        has = self.vector_names(name)
        stale = [v for v in VECTORS if v not in ("text_dense", "kw_tags", *vectors) and v in has]
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

    def set_tags(
        self, name: str, ids: list[str], tags: list[list[str]], vectors: np.ndarray
    ) -> None:
        """Each item's tags as payload, and their keyword string's embedding as kw_tags."""
        if "kw_tags" not in self.vector_names(name):
            raise ValueError(f"collection '{name}' was made before tags; it has no kw_tags vector")
        pids: list[ExtendedPointId] = [point_id(name, i) for i in ids]
        self.client.update_vectors(
            name,
            points=[
                PointVectors(id=pid, vector={"kw_tags": vec.tolist()})
                for pid, vec in zip(pids, vectors, strict=True)
            ],
        )
        self.send_payloads(name, [(pid, {"tags": t}) for pid, t in zip(pids, tags, strict=True)])

    def search(
        self,
        name: str,
        using: str,
        vector: np.ndarray,
        limit: int,
        skip: tuple[str, ...] = (),
        exclude: Collection[str] = (),
    ) -> list[Hit]:
        """The `limit` nearest items, leaving out skipped sources and excluded item ids."""
        result = self.client.query_points(
            name,
            query=vector.tolist(),
            using=using,
            limit=limit,
            with_payload=True,
            query_filter=skip_filter(skip, items_excluded(name, exclude)),
        )
        return [to_hit(p) for p in result.points]

    def search_sparse(
        self,
        name: str,
        vector: SparseVector,
        limit: int,
        skip: tuple[str, ...] = (),
        exclude: Collection[str] = (),
    ) -> list[Hit]:
        result = self.client.query_points(
            name,
            query=vector,
            using=SPARSE,
            limit=limit,
            with_payload=True,
            query_filter=skip_filter(skip, items_excluded(name, exclude)),
        )
        return [to_hit(p) for p in result.points]

    def search_chunks(
        self,
        name: str,
        vector: np.ndarray,
        limit: int,
        skip: tuple[str, ...] = (),
        exclude: Collection[str] = (),
    ) -> list[tuple[str, float]]:
        """The `limit` nearest chunks as (parent id, score); `exclude` names parent items."""
        result = self.client.query_points(
            chunks_of(name),
            query=vector.tolist(),
            using=CHUNK,
            limit=limit,
            with_payload=True,
            query_filter=skip_filter(skip, parents_excluded(exclude)),
        )
        return [((p.payload or {})["parent_id"], float(p.score)) for p in result.points]

    def get(self, name: str, ids: list[str]) -> list[Hit]:
        records = self.client.retrieve(name, [point_id(name, i) for i in ids], with_payload=True)
        by_id = {to_hit(r).id: to_hit(r) for r in records}
        return [by_id[i] for i in ids if i in by_id]

    def held(self, name: str, ids: list[str]) -> set[str]:
        """Which of these item ids the store holds, read without the payload texts."""
        records = self.client.retrieve(
            name, [point_id(name, i) for i in ids], with_payload=["item_id"]
        )
        return {(r.payload or {})["item_id"] for r in records}

    def ids(self, name: str, batch: int = 1024, skip: tuple[str, ...] = ()) -> list[str]:
        out: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                name,
                limit=batch,
                offset=offset,
                with_payload=["item_id"],
                scroll_filter=skip_filter(skip),
            )
            out.extend((r.payload or {})["item_id"] for r in records)
            if offset is None:
                return sorted(out)

    def scroll(
        self, name: str, batch: int = 256, skip: tuple[str, ...] = ()
    ) -> Iterator[list[Hit]]:
        offset = None
        while True:
            records, offset = self.client.scroll(
                name,
                limit=batch,
                offset=offset,
                with_payload=True,
                scroll_filter=skip_filter(skip),
            )
            if records:
                yield [to_hit(r) for r in records]
            if offset is None:
                return

    def vectors(self, name: str, using: str, limit: int, skip: tuple[str, ...] = ()) -> np.ndarray:
        """Up to `limit` items' `using` vectors, one per row, leaving out skipped sources.
        The items come in point-id order, which the uuid5 ids make an arbitrary sample."""
        records, _ = self.client.scroll(
            name,
            limit=limit,
            with_payload=False,
            with_vectors=[using],
            scroll_filter=skip_filter(skip),
        )
        held = [r.vector for r in records if isinstance(r.vector, dict)]
        rows = [v[using] for v in held if using in v]
        return np.array(rows, dtype=np.float32)

    def hashes(self, name: str, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for start in range(0, len(ids), 1024):
            part = ids[start : start + 1024]
            records = self.client.retrieve(
                name,
                [point_id(name, i) for i in part],
                with_payload=["item_id", "content_hash"],
            )
            for r in records:
                p = r.payload or {}
                out[p["item_id"]] = p.get("content_hash") or ""
        return out

    def source_counts(self, name: str) -> dict[str, int]:
        facet = self.client.facet(name, key="meta.source", limit=10_000, exact=True)
        return {str(h.value): h.count for h in facet.hits}

    def sources_of(self, name: str, ids: list[str]) -> dict[str, str | None]:
        """The source each existing item among the ids carries (None if unlabelled)."""
        out: dict[str, str | None] = {}
        for start in range(0, len(ids), 1024):
            records = self.client.retrieve(
                name,
                [point_id(name, i) for i in ids[start : start + 1024]],
                with_payload=["item_id", "meta.source"],
            )
            for r in records:
                p = r.payload or {}
                out[p["item_id"]] = (p.get("meta") or {}).get("source")
        return out

    def unsourced(self, name: str) -> int:
        """Item points and chunk points that carry no source."""
        empty = Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="meta.source"))])
        return sum(
            self.client.count(c, count_filter=empty, exact=True).count
            for c in (name, chunks_of(name))
        )

    def field_counts(self, name: str) -> dict[str, int]:
        def filled(key: str) -> int:
            f = Filter(must_not=[IsEmptyCondition(is_empty=PayloadField(key=key))])
            return self.client.count(name, count_filter=f, exact=True).count

        return {"explanation": filled("explanation"), "skill": filled("meta.skill")}

    def source_ids(self, name: str, source: str) -> list[str]:
        """The ids of the items in one source, read without the payload texts."""
        by_source = Filter(must=[FieldCondition(key="meta.source", match=MatchValue(value=source))])
        out: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                name, limit=1024, offset=offset, with_payload=["item_id"], scroll_filter=by_source
            )
            out.extend((r.payload or {})["item_id"] for r in records)
            if offset is None:
                return sorted(out)

    def delete_items(self, name: str, ids: list[str]) -> None:
        """Delete these items and their chunks."""
        self.client.delete(
            name, points_selector=PointIdsList(points=[point_id(name, i) for i in ids])
        )
        self.delete_chunks(name, ids)

    def delete_source(self, name: str, source: str) -> None:
        by_source = Filter(must=[FieldCondition(key="meta.source", match=MatchValue(value=source))])
        for collection in (name, chunks_of(name)):
            self.client.delete(collection, points_selector=FilterSelector(filter=by_source))

    def set_payloads(self, name: str, payloads: dict[str, dict]) -> None:
        """Set the given payload keys on each item, by item id; other keys stay."""
        self.send_payloads(name, [(point_id(name, i), p) for i, p in payloads.items()])

    def set_chunk_sources(self, name: str, source_of: dict[str, str]) -> None:
        updates: list[tuple[ExtendedPointId, dict]] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                chunks_of(name), limit=1024, offset=offset, with_payload=["parent_id"]
            )
            for r in records:
                parent = (r.payload or {})["parent_id"]
                if parent in source_of:
                    updates.append((r.id, {"meta": {"source": source_of[parent]}}))
            if offset is None:
                break
        self.send_payloads(chunks_of(name), updates)

    def send_payloads(self, collection: str, updates: list[tuple[ExtendedPointId, dict]]) -> None:
        for start in range(0, len(updates), 256):
            self.client.batch_update_points(
                collection,
                update_operations=[
                    SetPayloadOperation(set_payload=SetPayload(payload=payload, points=[pid]))
                    for pid, payload in updates[start : start + 256]
                ],
            )
