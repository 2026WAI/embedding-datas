#!/usr/bin/env python3
"""운영자가 BGE-M3를 한 번 내려받아 로컬에서 사용하기 위한 도구"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_store import DEFAULT_MODEL_DIR, MODEL_ID


def parse_args() -> argparse.Namespace:
    """모델 다운로드 명령행 인자 해석

    Returns:
        모델 ID, 저장 경로, 선택한 revision을 포함한 인자 객체
    """
    parser = argparse.ArgumentParser(description="BGE-M3 모델을 로컬 디렉터리에 내려받습니다.")
    parser.add_argument("--model-id", default=MODEL_ID, help=f"Hugging Face 모델 ID (기본값: {MODEL_ID})")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="저장 디렉터리")
    parser.add_argument("--revision", help="고정할 Hugging Face revision 또는 commit")
    return parser.parse_args()


def main() -> None:
    """BGE-M3 스냅샷의 지정 로컬 디렉터리 저장

    Raises:
        SystemExit: huggingface-hub 패키지 미설치
    """
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub가 없습니다. python -m pip install -r requirements.txt를 실행하세요.") from exc
    path = snapshot_download(repo_id=args.model_id, local_dir=args.model_dir, revision=args.revision)
    print(f"모델 다운로드 완료: {path}")
    print("다음 단계: python sync_embeddings.py")


if __name__ == "__main__":
    main()
