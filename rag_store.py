"""BGE-M3 dense/sparse 임베딩과 SQLite 역색인을 다루는 공통 기능."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import queue
import sqlite3
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Sequence

MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
STORE_FORMAT = "bge-m3-hybrid-v1"
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
    """청크, dense 벡터 및 BGE-M3 sparse 역색인 스키마 생성

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
        CREATE TABLE IF NOT EXISTS sparse_postings (
            token_id TEXT NOT NULL,
            chunk_rowid INTEGER NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (token_id, chunk_rowid)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_sparse_postings_chunk ON sparse_postings(chunk_rowid);
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
        "DROP TABLE IF EXISTS sparse_postings; DROP TABLE IF EXISTS vec_chunks; "
        "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS store_config;"
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
    store_format = _config(connection, "store_format")
    if store_format != STORE_FORMAT and connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]:
        raise RuntimeError("DB 인덱스 형식이 다릅니다. sync_embeddings.py --rebuild로 새로 만드세요.")


def _set_config(connection: sqlite3.Connection, model_id: str) -> None:
    connection.executemany(
        "INSERT INTO store_config(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        {
            "model_id": model_id,
            "embedding_dimension": str(EMBEDDING_DIMENSION),
            "distance_metric": "cosine",
            "store_format": STORE_FORMAT,
        }.items(),
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


class HybridEmbedder:
    """로컬 BGE-M3에서 dense 벡터와 sparse lexical weight를 함께 생성한다."""

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
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding이 없습니다. python -m pip install -r requirements.txt를 실행하세요."
            ) from exc
        self.model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=bool(device and device.startswith("cuda")),
            device=device,
        )

    def encode(self, texts: Sequence[str], batch_size: int) -> tuple[Any, list[dict[str, float]]]:
        """문자열 목록을 dense 벡터와 BGE-M3 sparse token weight로 변환한다.

        Args:
            texts: 임베딩할 청크 본문 또는 검색 질의 목록
            batch_size: 한 번의 모델 추론에 사용할 문자열 수

        Returns:
            행마다 하나의 1024차원 float32 벡터 및 token_id별 가중치 목록
        """
        output = self.model.encode(
            list(texts),
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"].astype("float32", copy=False)
        sparse = [
            {str(token_id): float(weight) for token_id, weight in weights.items() if float(weight) > 0.0}
            for weights in output["lexical_weights"]
        ]
        if dense.ndim != 2 or dense.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError("예상한 1024차원 BGE-M3 dense 벡터가 아닙니다.")
        return dense, sparse


def _batches(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _format_duration(seconds: float) -> str:
    """진행 로그에 표시할 짧은 경과 시간 문자열을 만든다."""
    rounded = max(0, round(seconds))
    minutes, seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _parallel_embedding_worker(
    worker_index: int,
    device: str,
    model_dir: str,
    batch_size: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """한 GPU에 고정되어 배치 임베딩만 수행하는 자식 프로세스 진입점."""
    try:
        embedder = HybridEmbedder(Path(model_dir), device)
    except BaseException:
        result_queue.put(("error", worker_index, None, traceback.format_exc()))
        return
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id, texts = task
        try:
            dense, sparse = embedder.encode(texts, batch_size)
            result_queue.put(("result", worker_index, task_id, dense, sparse))
        except BaseException:
            result_queue.put(("error", worker_index, task_id, traceback.format_exc()))
            return


def _parallel_embed_groups(
    groups: Iterator[Sequence[Chunk]], model_dir: Path, batch_size: int, devices: Sequence[str]
) -> Iterator[tuple[Sequence[Chunk], Any, list[dict[str, float]]]]:
    """여러 GPU worker에서 임베딩하고, 완료되는 순서대로 배치 결과를 반환한다.

    DB 쓰기는 이 함수 밖의 부모 프로세스에서만 실행한다. CUDA는 fork 후 초기화하면
    불안정할 수 있으므로 spawn 프로세스를 사용하고, 각 worker에 하나의 장치를 고정한다.
    """
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue(maxsize=1) for _ in devices]
    workers = [
        context.Process(
            target=_parallel_embedding_worker,
            args=(index, device, str(model_dir), batch_size, task_queue, result_queue),
            name=f"bge-m3-{device}",
        )
        for index, (device, task_queue) in enumerate(zip(devices, task_queues, strict=True))
    ]
    for worker in workers:
        worker.start()

    next_task_id = 0
    pending: dict[int, Sequence[Chunk]] = {}

    def submit(worker_index: int) -> bool:
        nonlocal next_task_id
        try:
            group = next(groups)
        except StopIteration:
            return False
        task_queues[worker_index].put((next_task_id, [chunk.text for chunk in group]))
        pending[next_task_id] = group
        next_task_id += 1
        return True

    try:
        for worker_index in range(len(workers)):
            submit(worker_index)
        while pending:
            try:
                message = result_queue.get(timeout=1)
            except queue.Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    names = ", ".join(f"{worker.name} (exit {worker.exitcode})" for worker in failed)
                    raise RuntimeError(f"병렬 임베딩 worker가 비정상 종료했습니다: {names}")
                continue
            kind, worker_index, task_id, *payload = message
            if kind == "error":
                detail = str(payload[0])
                raise RuntimeError(f"{devices[worker_index]} 임베딩 worker 오류:\n{detail}")
            dense, sparse = payload
            assert task_id is not None
            yield pending.pop(task_id), dense, sparse
            submit(worker_index)
    finally:
        for task_queue in task_queues:
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()


def synchronize(
    connection: sqlite3.Connection, catalog: dict[str, Chunk], model_id: str,
    model_dir: Path,
    batch_size: int,
    device: str | None,
    progress: str = "tqdm",
    devices: Sequence[str] | None = None,
) -> dict[str, int]:
    """현재 JSONL 카탈로그와 생성 DB의 완전 동기화

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        catalog: load_chunks가 반환한 청크 ID별 원본 레코드
        model_id: DB 호환성 확인 및 기록용 임베딩 모델 식별자
        model_dir: 로컬 BGE-M3 모델 디렉터리
        batch_size: 모델 추론 배치 크기
        device: 사용할 PyTorch 장치, 생략 시 라이브러리 기본값 사용
        progress: tqdm 동적 막대 또는 줄 단위 log 진행 출력 방식
        devices: 병렬 임베딩에 고정할 CUDA 장치 목록. 두 개 이상이면 GPU별 worker를 생성

    Returns:
        total, created, updated, deleted, unchanged 수를 담은 사전

    Raises:
        RuntimeError: 기존 DB의 모델 또는 벡터 차원 비호환
        ValueError: 알 수 없는 진행 출력 방식
        FileNotFoundError: 변경분 존재 시 로컬 모델 미존재

    Note:
        모든 DB 변경의 단일 트랜잭션 처리, 임베딩 도중 실패 시 이전 검색
        인덱스 유지
    """
    if progress not in {"tqdm", "log"}:
        raise ValueError("progress는 tqdm 또는 log여야 합니다.")
    if devices and device:
        raise ValueError("device와 devices는 함께 지정할 수 없습니다.")
    if devices and any(not value for value in devices):
        raise ValueError("devices에는 비어 있지 않은 장치 이름이 필요합니다.")
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
    parallel_devices = tuple(devices or ())
    effective_device = parallel_devices[0] if len(parallel_devices) == 1 else device
    embedder = HybridEmbedder(model_dir, effective_device) if changed and len(parallel_devices) < 2 else None
    if changed and len(parallel_devices) >= 2:
        print(
            f"병렬 임베딩 시작: {', '.join(parallel_devices)} "
            f"(GPU별 worker 1개, SQLite 단일 writer)",
            flush=True,
        )

    # 중간 실패 시 기존 검색 인덱스 보존을 위한 모든 변경의 단일 트랜잭션 처리
    with connection:
        for _, rowid in removed:
            connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            connection.execute("DELETE FROM sparse_postings WHERE chunk_rowid = ?", (rowid,))
            connection.execute("DELETE FROM chunks WHERE rowid = ?", (rowid,))
        if changed:
            def index_group(
                group: Sequence[Chunk], dense_embeddings: Any, sparse_embeddings: list[dict[str, float]]
            ) -> None:
                for chunk, dense_embedding, sparse_embedding in zip(
                    group, dense_embeddings, sparse_embeddings, strict=True
                ):
                    if chunk.id in existing:
                        rowid = existing[chunk.id][0]
                        connection.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
                        connection.execute("DELETE FROM sparse_postings WHERE chunk_rowid = ?", (rowid,))
                        connection.execute(UPDATE_SQL, (*_values(chunk), rowid))
                    else:
                        rowid = int(connection.execute(INSERT_SQL, _values(chunk)).lastrowid)
                    connection.execute(
                        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                        (rowid, dense_embedding),
                    )
                    connection.executemany(
                        "INSERT INTO sparse_postings(token_id, chunk_rowid, weight) VALUES (?, ?, ?)",
                        ((token_id, rowid, weight) for token_id, weight in sparse_embedding.items()),
                    )

            def encoded_groups() -> Iterator[tuple[Sequence[Chunk], Any, list[dict[str, float]]]]:
                groups = iter(_batches(changed, batch_size))
                if len(parallel_devices) >= 2:
                    yield from _parallel_embed_groups(groups, model_dir, batch_size, parallel_devices)
                    return
                assert embedder is not None
                for group in groups:
                    dense_embeddings, sparse_embeddings = embedder.encode(
                        [chunk.text for chunk in group], batch_size
                    )
                    yield group, dense_embeddings, sparse_embeddings

            encoded = encoded_groups()
            try:
                if progress == "tqdm":
                    from tqdm import tqdm

                    with tqdm(
                        total=len(changed), desc="임베딩 진행", unit="청크", dynamic_ncols=True
                    ) as progress_bar:
                        for group, dense_embeddings, sparse_embeddings in encoded:
                            index_group(group, dense_embeddings, sparse_embeddings)
                            progress_bar.update(len(group))
                else:
                    started_at, completed = monotonic(), 0
                    for group, dense_embeddings, sparse_embeddings in encoded:
                        index_group(group, dense_embeddings, sparse_embeddings)
                        completed += len(group)
                        elapsed = monotonic() - started_at
                        rate = completed / elapsed if elapsed else 0.0
                        remaining = (len(changed) - completed) / rate if rate else 0.0
                        print(
                            f"임베딩 진행: {completed:,}/{len(changed):,} "
                            f"({completed / len(changed):.1%}) | {rate:.1f} 청크/s | "
                            f"경과 {_format_duration(elapsed)} | ETA {_format_duration(remaining)}",
                            flush=True,
                        )
            finally:
                encoded.close()
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
    if not {"chunks", "vec_chunks", "sparse_postings"}.issubset(names):
        raise RuntimeError("검색 인덱스가 없습니다. 먼저 python sync_embeddings.py를 실행하세요.")
    ensure_store_compatible(connection, model_id)
    if connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        raise RuntimeError("검색 가능한 청크가 없습니다. chunk/와 동기화 결과를 확인하세요.")


def _result(row: sqlite3.Row, **scores: float | int | None) -> dict[str, Any]:
    """DB 행과 검색 점수를 공통 결과 객체로 만든다."""
    result: dict[str, Any] = {
        "id": row["id"],
        "text": row["text"],
        "metadata": json.loads(row["metadata_json"]),
    }
    result.update({key: value for key, value in scores.items() if value is not None})
    return result


def search_dense(connection: sqlite3.Connection, query_embedding: Any, top_k: int) -> list[dict[str, Any]]:
    """sqlite-vec cosine KNN으로 dense 후보를 검색한다."""
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
        _result(
            row,
            distance=float(row["distance"]),
            similarity=1.0 - float(row["distance"]),
            dense_score=1.0 - float(row["distance"]),
        )
        for row in rows
    ]


def search_sparse(
    connection: sqlite3.Connection, query_weights: dict[str, float], top_k: int
) -> list[dict[str, Any]]:
    """BGE-M3 lexical weight의 내적으로 SQLite 역색인을 검색한다.

    질의 token_id와 청크 posting을 조인해 `sum(query_weight * document_weight)`를
    계산한다. 임시 테이블은 연결별로 격리되어 동시 검색 간에 공유되지 않는다.
    """
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if not query_weights:
        return []
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS query_sparse_terms "
        "(token_id TEXT PRIMARY KEY, weight REAL NOT NULL) WITHOUT ROWID"
    )
    connection.execute("DELETE FROM query_sparse_terms")
    connection.executemany(
        "INSERT INTO query_sparse_terms(token_id, weight) VALUES (?, ?)", query_weights.items()
    )
    rows = connection.execute(
        """
        WITH sparse_scores AS (
            SELECT postings.chunk_rowid, SUM(postings.weight * query_terms.weight) AS score
            FROM query_sparse_terms AS query_terms
            JOIN sparse_postings AS postings USING (token_id)
            GROUP BY postings.chunk_rowid
            ORDER BY score DESC, postings.chunk_rowid ASC
            LIMIT ?
        )
        SELECT chunks.id, chunks.text, chunks.metadata_json, sparse_scores.score
        FROM sparse_scores JOIN chunks ON chunks.rowid = sparse_scores.chunk_rowid
        ORDER BY sparse_scores.score DESC, sparse_scores.chunk_rowid ASC
        """,
        (top_k,),
    ).fetchall()
    return [_result(row, sparse_score=float(row["score"])) for row in rows]


def search_hybrid(
    connection: sqlite3.Connection,
    query_embedding: Any,
    query_weights: dict[str, float],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """dense와 sparse 후보를 RRF(Reciprocal Rank Fusion)로 합친다."""
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    candidate_k = max(50, top_k * 5)
    dense_results = search_dense(connection, query_embedding, candidate_k)
    sparse_results = search_sparse(connection, query_weights, candidate_k)
    merged: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, 1):
        fused = merged.setdefault(result["id"], dict(result))
        fused["dense_rank"] = rank
        fused["rrf_score"] = float(fused.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
    for rank, result in enumerate(sparse_results, 1):
        fused = merged.setdefault(result["id"], dict(result))
        fused["sparse_rank"] = rank
        fused["sparse_score"] = result["sparse_score"]
        fused["rrf_score"] = float(fused.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
    return sorted(
        merged.values(), key=lambda item: (-float(item["rrf_score"]), str(item["id"]))
    )[:top_k]


def search(
    connection: sqlite3.Connection,
    query_embedding: Any,
    query_weights: dict[str, float],
    top_k: int,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """지정한 dense, sparse 또는 hybrid 방식으로 청크를 검색한다.

    Args:
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        query_embedding: 정규화된 1024차원 float32 질의 벡터
        query_weights: BGE-M3 sparse lexical weight
        top_k: 반환할 최근접 청크 수
        mode: dense, sparse 또는 hybrid

    Returns:
        id, text, metadata 및 검색 모드별 점수를 포함한 결과 목록

    Raises:
        ValueError: top_k가 1보다 작은 경우
    """
    if mode == "dense":
        return search_dense(connection, query_embedding, top_k)
    if mode == "sparse":
        return search_sparse(connection, query_weights, top_k)
    if mode == "hybrid":
        return search_hybrid(connection, query_embedding, query_weights, top_k)
    raise ValueError("mode는 dense, sparse 또는 hybrid여야 합니다.")
