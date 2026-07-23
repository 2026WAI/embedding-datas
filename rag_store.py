"""청크 JSONL을 로컬 BGE-M3 및 SQLite/sqlite-vec로 다루는 공통 기능"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
DEFAULT_CHUNK_DIR = Path("chunk")
DEFAULT_DB_PATH = Path("vector_store/rag.sqlite3")
DEFAULT_MODEL_DIR = Path(".models/bge-m3")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]
    metadata_json: str
    content_hash: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(path: Path, line: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line}: {message}")


def load_chunks(chunk_dir: Path) -> dict[str, Chunk]:
    """청크 루트 아래 모든 chunks.jsonl의 검증 및 로드

    Args:
        chunk_dir: chunks.jsonl 파일 트리의 루트 디렉터리

    Returns:
        청크 ID를 키로 하는 검증 완료 Chunk 객체 사전

    Raises:
        FileNotFoundError: chunk_dir 미존재
        ValueError: JSONL 형식, 필수 필드 또는 청크 ID 오류
    """
    if not chunk_dir.exists():
        raise FileNotFoundError(f"청크 디렉터리가 없습니다: {chunk_dir}")
    catalog: dict[str, Chunk] = {}
    for path in sorted(chunk_dir.rglob("chunks.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line_number, raw in enumerate(file, 1):
                if not (line := raw.strip()):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _error(path, line_number, f"JSONL이 아닙니다: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise _error(path, line_number, "레코드는 JSON 객체여야 합니다.")
                chunk_id, text, metadata = record.get("id"), record.get("text"), record.get("metadata")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise _error(path, line_number, "문자열 id가 필요합니다.")
                if not isinstance(text, str):
                    raise _error(path, line_number, "문자열 text가 필요합니다.")
                if not isinstance(metadata, dict):
                    raise _error(path, line_number, "객체 metadata가 필요합니다.")
                if chunk_id in catalog:
                    raise _error(path, line_number, f"중복 id입니다: {chunk_id}")
                metadata_json = _canonical_json(metadata)
                digest = hashlib.sha256(
                    _canonical_json({"id": chunk_id, "text": text, "metadata": metadata}).encode()
                ).hexdigest()
                catalog[chunk_id] = Chunk(chunk_id, text, metadata, metadata_json, digest)
    return catalog


def _load_sqlite_vec(connection: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError(
            "sqlite-vec가 없습니다. python -m pip install -r requirements.txt를 실행하세요."
        ) from exc
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
    except (AttributeError, sqlite3.Error) as exc:
        raise RuntimeError("sqlite-vec 확장을 SQLite에 로드하지 못했습니다.") from exc
    finally:
        try:
            connection.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


def connect_database(db_path: Path) -> sqlite3.Connection:
    """sqlite-vec 확장 로드 SQLite 연결 생성

    Args:
        db_path: 생성하거나 열 SQLite 데이터베이스 파일 경로

    Returns:
        행 이름 조회 및 sqlite-vec 로드가 완료된 SQLite 연결

    Raises:
        RuntimeError: sqlite-vec 패키지 부재 또는 SQLite 확장 로드 실패
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    _load_sqlite_vec(connection)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """청크 메타데이터 및 1024차원 벡터 검색용 스키마 생성

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
    """
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL,
            source TEXT, source_directory TEXT, item_id TEXT, title TEXT,
            published_date TEXT, detail_url TEXT, section_path_json TEXT NOT NULL,
            chunk_index INTEGER, chunk_type TEXT, input_content_path TEXT,
            content_hash TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        CREATE INDEX IF NOT EXISTS idx_chunks_item_id ON chunks(item_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_published_date ON chunks(published_date);
        CREATE TABLE IF NOT EXISTS store_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[1024] distance_metric=cosine
        );
        """
    )


def rebuild_schema(connection: sqlite3.Connection) -> None:
    """생성 DB 스키마 삭제 및 재생성

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결

    Note:
        data/ 미접근 및 chunk/ 원본 파일 미수정
    """
    connection.executescript(
        "DROP TABLE IF EXISTS vec_chunks; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS store_config;"
    )
    create_schema(connection)


def _config(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM store_config WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def ensure_store_compatible(connection: sqlite3.Connection, model_id: str) -> None:
    dimension, stored_model = _config(connection, "embedding_dimension"), _config(connection, "model_id")
    if dimension is not None and int(dimension) != EMBEDDING_DIMENSION:
        raise RuntimeError("DB 차원이 BGE-M3(1024)와 다릅니다. sync_embeddings.py --rebuild를 실행하세요.")
    if stored_model is not None and stored_model != model_id:
        raise RuntimeError(
            f"DB 모델은 {stored_model}입니다. 현재 모델({model_id})을 쓰려면 sync_embeddings.py --rebuild를 실행하세요."
        )


def _set_config(connection: sqlite3.Connection, model_id: str) -> None:
    connection.executemany(
        "INSERT INTO store_config(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        {"model_id": model_id, "embedding_dimension": str(EMBEDDING_DIMENSION), "distance_metric": "cosine"}.items(),
    )


def _metadata_value(metadata: dict[str, Any], key: str) -> str | int | None:
    value = metadata.get(key)
    return value if value is None or isinstance(value, (str, int)) else str(value)


def _values(chunk: Chunk) -> tuple[Any, ...]:
    metadata = chunk.metadata
    section_path = metadata.get("section_path", [])
    if not isinstance(section_path, list):
        section_path = [str(section_path)]
    return (
        chunk.id, chunk.text, chunk.metadata_json,
        _metadata_value(metadata, "source"), _metadata_value(metadata, "source_directory"),
        _metadata_value(metadata, "item_id"), _metadata_value(metadata, "title"),
        _metadata_value(metadata, "published_date"), _metadata_value(metadata, "detail_url"),
        _canonical_json(section_path), _metadata_value(metadata, "chunk_index"),
        _metadata_value(metadata, "chunk_type"), _metadata_value(metadata, "input_content_path"),
        chunk.content_hash, datetime.now(timezone.utc).isoformat(),
    )


INSERT_SQL = """
INSERT INTO chunks (
    id, text, metadata_json, source, source_directory, item_id, title, published_date,
    detail_url, section_path_json, chunk_index, chunk_type, input_content_path,
    content_hash, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
UPDATE_SQL = """
UPDATE chunks SET id=?, text=?, metadata_json=?, source=?, source_directory=?, item_id=?,
    title=?, published_date=?, detail_url=?, section_path_json=?, chunk_index=?, chunk_type=?,
    input_content_path=?, content_hash=?, updated_at=? WHERE rowid=?
"""


class LocalEmbedder:
    """미리 다운로드한 BGE-M3 기반 정규화 밀집 벡터 생성"""

    def __init__(self, model_dir: Path, device: str | None = None):
        """로컬 BGE-M3 모델 로드

        Args:
            model_dir: download_model.py로 받은 BGE-M3 디렉터리
            device: 사용할 PyTorch 장치, 생략 시 라이브러리 기본값 사용

        Raises:
            FileNotFoundError: 모델 디렉터리 미존재
            RuntimeError: 의존성 부재 또는 1024차원 BGE-M3 모델 아님
        """
        if not model_dir.exists():
            raise FileNotFoundError(
                f"로컬 모델이 없습니다: {model_dir}. python download_model.py --model-dir {model_dir}를 먼저 실행하세요."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers가 없습니다. python -m pip install -r requirements.txt를 실행하세요."
            ) from exc
        kwargs: dict[str, Any] = {"local_files_only": True}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(str(model_dir), **kwargs)
        if self.model.get_embedding_dimension() != EMBEDDING_DIMENSION:
            raise RuntimeError("예상한 1024차원 BGE-M3 모델이 아닙니다.")

    def encode(self, texts: Sequence[str], batch_size: int) -> Any:
        """문자열 목록의 cosine 검색용 float32 정규화 벡터 변환

        Args:
            texts: 임베딩할 청크 본문 또는 검색 질의 목록
            batch_size: 한 번의 모델 추론에 사용할 문자열 수

        Returns:
            행마다 하나의 1024차원 벡터가 담긴 NumPy float32 배열
        """
        return self.model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32", copy=False)


def _batches(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def synchronize(
    connection: sqlite3.Connection, catalog: dict[str, Chunk], model_id: str,
    model_dir: Path, batch_size: int, device: str | None,
) -> dict[str, int]:
    """현재 JSONL 카탈로그와 생성 DB의 완전 동기화

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        catalog: load_chunks가 반환한 청크 ID별 원본 레코드
        model_id: DB 호환성 확인 및 기록용 임베딩 모델 식별자
        model_dir: 로컬 BGE-M3 모델 디렉터리
        batch_size: 모델 추론 배치 크기
        device: 사용할 PyTorch 장치, 생략 시 라이브러리 기본값 사용

    Returns:
        total, created, updated, deleted, unchanged 수를 담은 사전

    Raises:
        RuntimeError: 기존 DB의 모델 또는 벡터 차원 비호환
        FileNotFoundError: 변경분 존재 시 로컬 모델 미존재

    Note:
        모든 DB 변경의 단일 트랜잭션 처리, 임베딩 도중 실패 시 이전 검색
        인덱스 유지
    """
    create_schema(connection)
    ensure_store_compatible(connection, model_id)
    existing = {
        row["id"]: (int(row["rowid"]), str(row["content_hash"]))
        for row in connection.execute("SELECT rowid, id, content_hash FROM chunks")
    }
    removed = [(chunk_id, existing[chunk_id][0]) for chunk_id in existing.keys() - catalog.keys()]
    changed = [chunk for chunk in catalog.values() if chunk.id not in existing or existing[chunk.id][1] != chunk.content_hash]
    created = sum(chunk.id not in existing for chunk in changed)
    updated = len(changed) - created
    unchanged = len(catalog) - len(changed)
    print(
        f"변경사항: 추가 {created:,}, 갱신 {updated:,}, 삭제 {len(removed):,}, "
        f"유지 {unchanged:,} (임베딩 대상 {len(changed):,}개)"
    )
    embedder = LocalEmbedder(model_dir, device) if changed else None

    # 중간 실패 시 기존 검색 인덱스 보존을 위한 모든 변경의 단일 트랜잭션 처리
    with connection:
        for _, rowid in removed:
            connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            connection.execute("DELETE FROM chunks WHERE rowid = ?", (rowid,))
        if changed:
            from tqdm import tqdm

            with tqdm(
                total=len(changed), desc="임베딩 진행", unit="청크", dynamic_ncols=True
            ) as progress:
                for group in _batches(changed, batch_size):
                    assert embedder is not None
                    embeddings = embedder.encode([chunk.text for chunk in group], batch_size)
                    for chunk, embedding in zip(group, embeddings, strict=True):
                        if chunk.id in existing:
                            rowid = existing[chunk.id][0]
                            connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
                            connection.execute(UPDATE_SQL, (*_values(chunk), rowid))
                        else:
                            rowid = int(connection.execute(INSERT_SQL, _values(chunk)).lastrowid)
                        connection.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, embedding))
                    progress.update(len(group))
        _set_config(connection, model_id)
    return {
        "total": len(catalog), "created": sum(chunk.id not in existing for chunk in changed),
        "updated": updated, "deleted": len(removed), "unchanged": unchanged,
    }


def ensure_searchable(connection: sqlite3.Connection, model_id: str) -> None:
    """검색 전 인덱스, 모델 호환성 및 청크 존재 여부 확인

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        model_id: 현재 검색에 사용할 모델 식별자

    Raises:
        RuntimeError: 인덱스 부재, 인덱스 비어 있음 또는 모델 비호환
    """
    names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    if not {"chunks", "vec_chunks"}.issubset(names):
        raise RuntimeError("검색 인덱스가 없습니다. 먼저 python sync_embeddings.py를 실행하세요.")
    ensure_store_compatible(connection, model_id)
    if connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        raise RuntimeError("검색 가능한 청크가 없습니다. chunk/와 동기화 결과를 확인하세요.")


def search(connection: sqlite3.Connection, query_embedding: Any, top_k: int) -> list[dict[str, Any]]:
    """질의 벡터 근접 청크 및 원본 metadata 반환

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        query_embedding: 정규화된 1024차원 float32 질의 벡터
        top_k: 반환할 최근접 청크 수

    Returns:
        id, text, metadata, cosine distance, similarity를 포함한 결과 목록

    Raises:
        ValueError: top_k가 1보다 작은 경우
    """
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    rows = connection.execute(
        """
        WITH knn_matches AS (
            SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?
        )
        SELECT chunks.id, chunks.text, chunks.metadata_json, knn_matches.distance
        FROM knn_matches JOIN chunks ON chunks.rowid = knn_matches.rowid
        ORDER BY knn_matches.distance
        """,
        (query_embedding, top_k),
    ).fetchall()
    return [
        {
            "id": row["id"], "text": row["text"], "metadata": json.loads(row["metadata_json"]),
            "distance": float(row["distance"]), "similarity": 1.0 - float(row["distance"]),
        }
        for row in rows
    ]
