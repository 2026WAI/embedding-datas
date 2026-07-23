#!/usr/bin/env python3
"""청크 JSONL을 로컬 SQLite/sqlite-vec 검색 인덱스와 동기화하는 명령행 도구"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_store import (
    DEFAULT_CHUNK_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    connect_database,
    load_chunks,
    rebuild_schema,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    """동기화 명령행 인자 해석

    Returns:
        입력 청크, DB, 모델, 배치 및 재구축 설정을 포함한 인자 객체
    """
    parser = argparse.ArgumentParser(description="chunk JSONL과 로컬 벡터 DB를 동기화합니다.")
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR, help="chunks.jsonl 트리의 루트")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB 파일")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="내려받은 BGE-M3 경로")
    parser.add_argument("--model-id", default=MODEL_ID, help="DB에 기록할 모델 식별자")
    parser.add_argument("--batch-size", type=int, default=16, help="BGE-M3 임베딩 배치 크기")
    parser.add_argument("--device", help="예: cpu, cuda, cuda:0. 생략하면 기본 장치 선택")
    parser.add_argument("--rebuild", action="store_true", help="기존 생성 DB를 비우고 전체 재임베딩")
    return parser.parse_args()


def main() -> None:
    """청크 원천 기반 벡터 DB 추가, 갱신 및 삭제 동기화

    Raises:
        SystemExit: 배치 크기가 1보다 작은 경우
    """
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")
    catalog = load_chunks(args.chunk_dir)
    print(f"발견한 청크: {len(catalog):,}개")
    connection = connect_database(args.db_path)
    try:
        if args.rebuild:
            rebuild_schema(connection)
            print("기존 생성 인덱스를 비웠습니다.")
        result = synchronize(connection, catalog, args.model_id, args.model_dir, args.batch_size, args.device)
    finally:
        connection.close()
    print(
        f"동기화 완료: 전체 {result['total']:,}, 추가 {result['created']:,}, "
        f"갱신 {result['updated']:,}, 삭제 {result['deleted']:,}, 유지 {result['unchanged']:,}"
    )


if __name__ == "__main__":
    main()
