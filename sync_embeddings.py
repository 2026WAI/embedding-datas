#!/usr/bin/env python3
"""청크 JSONL을 로컬 SQLite/sqlite-vec 검색 인덱스와 동기화하는 명령행 도구"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

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

DEFAULT_CONFIG_PATH = Path("embedding_config.yaml")
DEFAULTS: dict[str, Any] = {
    "chunk_dir": DEFAULT_CHUNK_DIR,
    "db_path": DEFAULT_DB_PATH,
    "model_dir": DEFAULT_MODEL_DIR,
    "model_id": MODEL_ID,
    "batch_size": 16,
    "device": None,
    "progress": "tqdm",
}
PATH_SETTINGS = {"chunk_dir", "db_path", "model_dir"}


def load_config(path: Path) -> dict[str, Any]:
    """embedding_config.yaml을 읽고 동기화 설정으로 변환한다."""
    try:
        with path.open(encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except OSError as exc:
        raise ValueError(f"설정 파일을 읽을 수 없습니다: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 형식이 올바르지 않습니다: {path}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("설정 파일의 최상위 값은 객체여야 합니다.")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"알 수 없는 설정 항목: {', '.join(sorted(unknown))}")

    config: dict[str, Any] = {}
    for key, value in raw.items():
        if key in PATH_SETTINGS:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key}는 비어 있지 않은 경로 문자열이어야 합니다.")
            config[key] = Path(value) if Path(value).is_absolute() else path.parent / value
        elif key == "batch_size":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("batch_size는 1 이상의 정수여야 합니다.")
            config[key] = value
        elif key == "device":
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("device는 문자열 또는 null이어야 합니다.")
            config[key] = value
        elif key == "progress":
            if value not in {"tqdm", "log"}:
                raise ValueError("progress는 tqdm 또는 log여야 합니다.")
            config[key] = value
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{key}는 비어 있지 않은 문자열이어야 합니다.")
        else:
            config[key] = value
    return config


def parse_args() -> argparse.Namespace:
    """동기화 명령행 인자 해석

    Returns:
        입력 청크, DB, 모델, 배치 및 재구축 설정을 포함한 인자 객체
    """
    parser = argparse.ArgumentParser(description="chunk JSONL과 로컬 벡터 DB를 동기화합니다.")
    parser.add_argument("--config", type=Path, help=f"YAML 설정 파일 (기본: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--chunk-dir", type=Path, help="chunks.jsonl 트리의 루트")
    parser.add_argument("--db-path", type=Path, help="SQLite DB 파일")
    parser.add_argument("--model-dir", type=Path, help="내려받은 BGE-M3 경로")
    parser.add_argument("--model-id", help="DB에 기록할 모델 식별자")
    parser.add_argument("--batch-size", type=int, help="BGE-M3 임베딩 배치 크기")
    parser.add_argument("--device", help="예: cpu, cuda, cuda:0. 생략하면 기본 장치 선택")
    parser.add_argument(
        "--progress",
        choices=("tqdm", "log"),
        help="진행 출력 방식. log는 배치마다 새 줄을 flush해 노트북 로그에 적합",
    )
    parser.add_argument("--rebuild", action="store_true", help="기존 생성 DB를 비우고 전체 재임베딩")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """기본값, YAML 설정, 명령행 인자를 우선순위대로 합친다."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    if config_path.exists():
        config = load_config(config_path)
    elif args.config:
        raise ValueError(f"설정 파일이 없습니다: {config_path}")
    else:
        config = {}

    for key, default in DEFAULTS.items():
        command_line_value = getattr(args, key, None)
        setattr(args, key, command_line_value if command_line_value is not None else config.get(key, default))
    args.config = config_path
    return args


def main() -> None:
    """청크 원천 기반 벡터 DB 추가, 갱신 및 삭제 동기화

    Raises:
        SystemExit: 배치 크기가 1보다 작은 경우
    """
    try:
        args = resolve_args(parse_args())
    except ValueError as exc:
        raise SystemExit(f"설정 오류: {exc}") from exc
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")
    catalog = load_chunks(args.chunk_dir)
    print(f"발견한 청크: {len(catalog):,}개")
    connection = connect_database(args.db_path)
    try:
        if args.rebuild:
            rebuild_schema(connection)
            print("기존 생성 인덱스를 비웠습니다.")
        result = synchronize(
            connection,
            catalog,
            args.model_id,
            args.model_dir,
            args.batch_size,
            args.device,
            args.progress,
        )
    finally:
        connection.close()
    print(
        f"동기화 완료: 전체 {result['total']:,}, 추가 {result['created']:,}, "
        f"갱신 {result['updated']:,}, 삭제 {result['deleted']:,}, 유지 {result['unchanged']:,}"
    )


if __name__ == "__main__":
    main()
