#!/usr/bin/env python3
"""동기화된 청크를 대화형 또는 단발성으로 검색하는 명령행 도구"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from rag_store import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    LocalEmbedder,
    connect_database,
    ensure_searchable,
    search,
)


def parse_args() -> argparse.Namespace:
    """검색 명령행 인자 해석

    Returns:
        DB, 모델, 검색 개수, 출력 형식 설정을 포함한 인자 객체
    """
    parser = argparse.ArgumentParser(description="BGE-M3 + sqlite-vec 로컬 청크 검색")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB 파일")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="내려받은 BGE-M3 경로")
    parser.add_argument("--model-id", default=MODEL_ID, help="DB와 비교할 모델 식별자")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="반환할 청크 수")
    parser.add_argument("--device", help="예: cpu, cuda, cuda:0")
    parser.add_argument("--query", help="한 번만 검색하고 종료")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    parser.add_argument("--max-text-chars", type=int, default=1000, help="본문 표시 길이. 0이면 전체")
    return parser.parse_args()


def _preview(text: str, maximum: int) -> str:
    """터미널 표시용 본문 미리보기 생성

    Args:
        text: 원본 청크 텍스트
        maximum: 최대 표시 문자 수, 0이면 축약 없음

    Returns:
        필요 시 생략 표시가 붙은 텍스트
    """
    return text if maximum == 0 or len(text) <= maximum else text[:maximum].rstrip() + "\n… (생략됨)"


def print_results(results: list[dict[str, Any]], maximum: int, as_json: bool) -> None:
    """검색 결과의 일반 텍스트 또는 JSON 출력

    Args:
        results: rag_store.search가 반환한 결과 목록
        maximum: 일반 텍스트 출력의 최대 본문 문자 수
        as_json: JSON 배열 전체 출력 여부
    """
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if not results:
        print("검색 결과가 없습니다.")
        return
    for rank, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"\n[{rank}] {result['id']}  similarity={result['similarity']:.4f}")
        print(f"제목: {metadata.get('title') or '(제목 없음)'}")
        print(
            f"출처: {metadata.get('source') or '-'} / item_id={metadata.get('item_id') or '-'} "
            f"/ type={metadata.get('chunk_type') or '-'}"
        )
        if isinstance(metadata.get("section_path"), list) and metadata["section_path"]:
            print("섹션: " + " > ".join(str(value) for value in metadata["section_path"]))
        if metadata.get("published_date"):
            print(f"발행일: {metadata['published_date']}")
        if metadata.get("detail_url"):
            print(f"원문: {metadata['detail_url']}")
        print("본문:\n" + _preview(result["text"], maximum))


def run_query(
    embedder: LocalEmbedder, connection: sqlite3.Connection, query: str, top_k: int
) -> list[dict[str, Any]]:
    """질의 임베딩 및 단일 벡터 검색 실행

    Args:
        embedder: 로컬 BGE-M3 임베더
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        query: 사용자 입력 검색 질의
        top_k: 반환할 최근접 청크 수

    Returns:
        유사도 순 정렬 검색 결과 목록
    """
    return search(connection, embedder.encode([query], 1)[0], top_k)


def main() -> None:
    """단발성 검색 또는 종료 명령 지원 대화형 검색 세션 실행

    Raises:
        SystemExit: 검색 개수 또는 표시 문자 수 설정 오류
    """
    args = parse_args()
    if args.top_k < 1 or args.max_text_chars < 0:
        raise SystemExit("--top-k는 1 이상, --max-text-chars는 0 이상이어야 합니다.")
    connection = connect_database(args.db_path)
    try:
        ensure_searchable(connection, args.model_id)
        embedder = LocalEmbedder(args.model_dir, args.device)
        if args.query is not None:
            print_results(run_query(embedder, connection, args.query, args.top_k), args.max_text_chars, args.json)
            return
        print("로컬 검색을 시작합니다. 종료: :q, quit, exit")
        while True:
            try:
                query = input("검색> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {":q", "quit", "exit"}:
                break
            if query:
                print_results(run_query(embedder, connection, query, args.top_k), args.max_text_chars, args.json)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
