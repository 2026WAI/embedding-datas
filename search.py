#!/usr/bin/env python3
"""동기화된 청크를 대화형 또는 단발성으로 검색하는 명령행 도구"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from pygments import highlight
    from pygments.formatters import TerminalFormatter
    from pygments.lexers import TextLexer, get_lexer_by_name
    from pygments.util import ClassNotFound
except ImportError:  # pragma: no cover - requirements 설치 전에도 검색은 가능해야 한다.
    highlight = None

from rag_store import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    HybridEmbedder,
    connect_database,
    ensure_searchable,
    search,
)
from sync_embeddings import DEFAULT_CONFIG_PATH, DEFAULTS, load_config


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
FENCED_CODE_START = re.compile(r"^\s*```(?P<language>[A-Za-z0-9_+.-]*)\s*$")


def parse_args() -> argparse.Namespace:
    """검색 명령행 인자 해석

    Returns:
        DB, 모델, 검색 개수, 출력 형식 설정을 포함한 인자 객체
    """
    parser = argparse.ArgumentParser(description="BGE-M3 dense/sparse hybrid 로컬 청크 검색")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB 파일")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="내려받은 BGE-M3 경로")
    parser.add_argument("--model-id", default=MODEL_ID, help="DB와 비교할 모델 식별자")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="반환할 청크 수")
    parser.add_argument("--device", help="예: cpu, cuda, cuda:0")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "dense", "sparse"),
        default="hybrid",
        help="검색 방식 (기본: hybrid; dense 후보를 sparse로 재채점해 가중 결합)",
    )
    parser.add_argument("--query", help="한 번만 검색하고 종료")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="검색 설정 YAML 파일")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="색상 출력 방식 (기본: auto)",
    )
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


def _color_enabled(mode: str) -> bool:
    """색상 출력 여부를 결정한다."""
    return mode == "always" or (mode == "auto" and sys.stdout.isatty())


def _styled(text: str, *codes: str, enabled: bool) -> str:
    """색상 사용이 허용될 때만 ANSI 스타일을 적용한다."""
    return f"{''.join(codes)}{text}{ANSI_RESET}" if enabled else text


def _rule(label: str, character: str, enabled: bool) -> str:
    """결과 영역을 구분하는 읽기 쉬운 수평선 생성."""
    plain = f" {label} "
    line = f"{character * 12}{plain}{character * 12}"
    return _styled(line, ANSI_DIM, enabled=enabled)


def _highlight_code_blocks(text: str, enabled: bool) -> str:
    """Markdown fenced code block만 Pygments로 터미널 하이라이팅한다."""
    if not enabled or highlight is None:
        return text

    rendered: list[str] = []
    code_lines: list[str] = []
    language: str | None = None

    def flush_code() -> None:
        nonlocal code_lines
        if not code_lines:
            return
        code = "".join(code_lines)
        try:
            lexer = get_lexer_by_name(language) if language else TextLexer()
        except ClassNotFound:
            lexer = TextLexer()
        rendered.append(highlight(code, lexer, TerminalFormatter(bg="dark")).rstrip("\n"))
        code_lines = []

    for line in text.splitlines(keepends=True):
        if language is None:
            match = FENCED_CODE_START.match(line.rstrip("\n"))
            if match:
                language = match.group("language")
                rendered.append(_styled(line.rstrip("\n"), ANSI_DIM, enabled=True))
            else:
                rendered.append(line.rstrip("\n"))
        elif line.strip() == "```":
            flush_code()
            rendered.append(_styled(line.rstrip("\n"), ANSI_DIM, enabled=True))
            language = None
        else:
            code_lines.append(line)

    if language is not None:
        flush_code()
    return "\n".join(rendered)


def print_results(
    results: list[dict[str, Any]],
    maximum: int,
    as_json: bool,
    query: str | None = None,
    color: str = "auto",
    elapsed_seconds: float | None = None,
) -> None:
    """검색 결과의 일반 텍스트 또는 JSON 출력

    Args:
        results: rag_store.search가 반환한 결과 목록
        maximum: 일반 텍스트 출력의 최대 본문 문자 수
        as_json: JSON 배열 전체 출력 여부
        query: 일반 텍스트 출력에 표시할 검색 질의
        color: 색상 출력 방식
        elapsed_seconds: 질의 임베딩과 검색에 걸린 시간(초)
    """
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    use_color = _color_enabled(color)
    if query is not None:
        print(_rule("검색 질의", "=", use_color))
        print(_styled(query, ANSI_BOLD, ANSI_CYAN, enabled=use_color))
        if elapsed_seconds is not None:
            print(f"검색 소요 시간 ({elapsed_seconds:.2f}s)")
        print(_rule("검색 결과", "=", use_color))
    if not results:
        print("검색 결과가 없습니다.")
        return
    for rank, result in enumerate(results, 1):
        metadata = result["metadata"]
        print()
        score = (
            f"hybrid={result['hybrid_score']:.4f}"
            if "hybrid_score" in result
            else f"sparse={result['sparse_score']:.4f}"
            if "sparse_score" in result and "similarity" not in result
            else f"similarity={result['similarity']:.4f}"
        )
        print(_rule(f"{rank}위  {score}", "-", use_color))
        print(_styled(result["id"], ANSI_BOLD, ANSI_GREEN, enabled=use_color))
        print(f"제목: {_styled(str(metadata.get('title') or '(제목 없음)'), ANSI_BOLD, enabled=use_color)}")
        print(
            f"출처: {metadata.get('source') or '-'} / item_id={metadata.get('item_id') or '-'} "
            f"/ type={metadata.get('chunk_type') or '-'}"
        )
        if isinstance(metadata.get("section_path"), list) and metadata["section_path"]:
            print("섹션: " + " > ".join(str(value) for value in metadata["section_path"]))
        if metadata.get("published_date"):
            print(f"발행일: {metadata['published_date']}")
        if metadata.get("detail_url"):
            print(f"원문: {_styled(str(metadata['detail_url']), ANSI_CYAN, enabled=use_color)}")
        print(_rule("본문", "-", use_color))
        print(_highlight_code_blocks(_preview(result["text"], maximum), use_color))


def run_query(
    embedder: HybridEmbedder,
    connection: sqlite3.Connection,
    query: str,
    top_k: int,
    mode: str,
    dense_candidates: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[dict[str, Any]]:
    """질의의 dense/sparse 임베딩 생성 및 지정 검색 실행

    Args:
        embedder: 로컬 BGE-M3 임베더
        connection: sqlite-vec 로드가 완료된 SQLite 연결
        query: 사용자 입력 검색 질의
        top_k: 반환할 최근접 청크 수
        mode: dense, sparse 또는 hybrid

    Returns:
        유사도 순 정렬 검색 결과 목록
    """
    dense_embeddings, sparse_embeddings = embedder.encode([query], 1)
    return search(
        connection,
        dense_embeddings[0],
        sparse_embeddings[0],
        top_k,
        mode,
        dense_candidates,
        dense_weight,
        sparse_weight,
    )


def load_search_settings(config_path: Path) -> tuple[int, float, float]:
    """YAML에서 hybrid 후보 수와 가중치를 읽어 검색에 적용한다."""
    if config_path.exists():
        config = load_config(config_path)
    elif config_path != DEFAULT_CONFIG_PATH:
        raise ValueError(f"설정 파일이 없습니다: {config_path}")
    else:
        config = {}
    dense_candidates = int(config.get("hybrid_dense_candidates", DEFAULTS["hybrid_dense_candidates"]))
    dense_weight = float(config.get("hybrid_dense_weight", DEFAULTS["hybrid_dense_weight"]))
    sparse_weight = float(config.get("hybrid_sparse_weight", DEFAULTS["hybrid_sparse_weight"]))
    if dense_weight + sparse_weight == 0:
        raise ValueError("hybrid_dense_weight와 hybrid_sparse_weight의 합은 양수여야 합니다.")
    return dense_candidates, dense_weight, sparse_weight


def main() -> None:
    """단발성 검색 또는 종료 명령 지원 대화형 검색 세션 실행

    Raises:
        SystemExit: 검색 개수 또는 표시 문자 수 설정 오류
    """
    args = parse_args()
    if args.top_k < 1 or args.max_text_chars < 0:
        raise SystemExit("--top-k는 1 이상, --max-text-chars는 0 이상이어야 합니다.")
    try:
        dense_candidates, dense_weight, sparse_weight = load_search_settings(args.config)
    except ValueError as exc:
        raise SystemExit(f"설정 오류: {exc}") from exc
    if args.mode == "hybrid" and dense_candidates < args.top_k:
        raise SystemExit("hybrid_dense_candidates는 --top-k 이상이어야 합니다.")
    connection = connect_database(args.db_path)
    try:
        ensure_searchable(connection, args.model_id)
        embedder = HybridEmbedder(args.model_dir, args.device)
        if args.query is not None:
            started_at = perf_counter()
            results = run_query(
                embedder, connection, args.query, args.top_k, args.mode,
                dense_candidates, dense_weight, sparse_weight,
            )
            print_results(
                results,
                args.max_text_chars,
                args.json,
                query=args.query,
                color=args.color,
                elapsed_seconds=perf_counter() - started_at,
            )
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
                started_at = perf_counter()
                results = run_query(
                    embedder, connection, query, args.top_k, args.mode,
                    dense_candidates, dense_weight, sparse_weight,
                )
                print_results(
                    results,
                    args.max_text_chars,
                    args.json,
                    query=query,
                    color=args.color,
                    elapsed_seconds=perf_counter() - started_at,
                )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
