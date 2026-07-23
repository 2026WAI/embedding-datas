# embedding-datas

로컬 hybrid RAG 검색용 임베딩·인덱싱 도구

- 입력 원본: `chunk/**/chunks.jsonl` — 읽기 전용
- 임베딩 모델: BAAI BGE-M3 (`dense` + `sparse`)
- 검색 방식: SQLite 역색인 + sqlite-vec + RRF hybrid fusion
- 생성 경로: `.models/`, `vector_store/`

## 빠른 시작

요구 사항

- Python 3.10+
- GPU 사용 시: 환경에 맞는 PyTorch를 먼저 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 최초 1회: 모델 다운로드
python download_model.py

# 전체 청크 임베딩 및 hybrid 인덱스 생성
python sync_embeddings.py --rebuild

# 대화형 검색
python search.py --top-k 5
```

## 검색

```bash
# 단일 질의
python search.py --query "전자금융사기 피해금 환급 절차" --top-k 3

# RAG 파이프라인용 JSON 결과
python search.py --query "전자금융사기 피해금 환급 절차" --top-k 3 --json
```

- 기본 모드: `hybrid` — dense + sparse 결과를 RRF로 결합
- 비교 모드: `--mode dense`, `--mode sparse`
- 대화형 종료: `:q`, `quit`, `exit`, `Ctrl-D`, `Ctrl-C`
- 색상: 자동 적용 · `--color always` · `--color never`
- 일반 출력: 질의 / 결과 / 본문 구분, Markdown 코드 블록 문법 하이라이팅

## GPU 노트북

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2026WAI/embedding-datas/blob/main/build_embeddings_colab.ipynb)

### Colab

- 파일: `build_embeddings_colab.ipynb`
- 입력: `chunk.7z` 업로드
- 결과: 구조 검증 후 `rag.sqlite3` 다운로드
- 실행: 노트북 셀을 위에서 아래 순서로 실행

### Kaggle

- 파일: `build_embeddings_kaggle.ipynb`
- 입력: 청크 Dataset을 Notebook Input으로 추가
- 설정: GPU 및 Internet 활성화
- 결과: **Save Version** → **Output**에서 `rag.sqlite3` 다운로드
- 진행률: `--progress log` — 배치별 진행률, 처리 속도, 경과 시간, ETA 출력

### 다중 GPU

- GPU별 BGE-M3 임베딩 worker 1개 고정
- 메인 프로세스가 SQLite에 순차 기록

## 설정

- 기본 설정 파일: `embedding_config.yaml`
- 주요 항목: `batch_size`, `device`, 모델 경로, 입출력 경로
- 우선순위: 명령행 옵션 > 설정 파일
- GPU: `device: cuda`
- 메모리 부족: `batch_size` 낮추기
- 다른 설정 파일: `python sync_embeddings.py --config 경로/설정.yaml`

## 동기화

`sync_embeddings.py`는 모든 `chunk/**/chunks.jsonl`을 읽고 `id`, `text`, `metadata`의 SHA-256 해시를 비교합니다.

- 신규 ID: 임베딩 후 추가
- 변경 ID: 벡터·메타데이터 교체
- 삭제 ID: 벡터·메타데이터 삭제
- 미변경 ID: 재임베딩 생략

실행 정보

- 시작 전: 추가 · 갱신 · 삭제 · 유지 · 실제 임베딩 대상 수
- 진행 중: 완료 수 · 처리 속도 · 경과 시간 · ETA
- 단일 원천: `chunks.jsonl`

## 저장소 구조

기본 경로

- 청크: `chunk/`
- 모델: `.models/bge-m3/`
- DB: `vector_store/rag.sqlite3`

경로 변경 옵션

- `--chunk-dir`
- `--model-dir`
- `--db-path`

SQLite 테이블

- `chunks`: 원문 + metadata JSON
- `vec_chunks`: 1024차원 cosine KNN용 vec0
- `sparse_postings`: `token_id → chunk_rowid → weight` sparse 역색인

결과 점수

- hybrid JSON: `rrf_score`, `dense_rank`, `sparse_rank`
- dense: `similarity`
- sparse: `sparse_score`

현재 스토어 형식은 hybrid 전용입니다. 새 인덱스 생성 시에는 아래처럼 재구축하세요.

```bash
python sync_embeddings.py --rebuild
```

## 검증

외부 모델 다운로드 없이 실행합니다.

```bash
python -m unittest discover -s tests -v
```

## 참고

- [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3)
- [sqlite-vec Python 문서](https://alexgarcia.xyz/sqlite-vec/python.html)
