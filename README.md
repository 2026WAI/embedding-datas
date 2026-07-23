# embedding-datas

chunk/**/chunks.jsonl만 원본으로 삼아 BGE-M3의 dense 및 sparse 임베딩을 만들고, 로컬
SQLite 역색인 + sqlite-vec에서 hybrid 검색하는 RAG 검색 기반입니다. chunk/는 읽기만 수행하며,
생성 파일은 .models/와 vector_store/ 입니다.

## 빠른 시작

Python 3.10+ 기준입니다. GPU를 쓸 경우에는 환경에 맞는 PyTorch를 먼저 설치하세요.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    # 운영자가 한 번 실행: BAAI/bge-m3를 프로젝트에 저장
    python download_model.py

    # 새 hybrid 인덱스를 만들며 chunk/의 모든 chunks.jsonl을 임베딩
    python sync_embeddings.py --rebuild

    # 대화형 top-k 검색
    python search.py --top-k 5

검색 중 :q, quit, exit 또는 Ctrl-D/Ctrl-C를 입력하면 종료합니다. 한 번만 검색할 때는
python search.py --query "전자금융사기 피해금 환급 절차" --top-k 3처럼 실행합니다.
RAG 파이프라인에 결과를 전달할 때는 --json을 추가하세요.

기본 검색 방식은 dense와 sparse 결과를 RRF(Reciprocal Rank Fusion)로 결합하는
`hybrid`입니다. 진단·비교를 위해 `--mode dense` 또는 `--mode sparse`도 지정할 수 있습니다.

일반 검색 출력은 질의·결과·본문을 `=`와 `-` 구분선으로 나누며, Markdown 코드 펜스
(예: `python` 언어 지정 블록) 안의 코드는 터미널에서 문법 하이라이팅합니다. 색상은 대화형 터미널에서
자동 적용되고, `--color always` 또는 `--color never`로 강제하거나 끌 수 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2026WAI/embedding-datas/blob/main/build_embeddings_colab.ipynb)

CPU 대신 Colab GPU에서 전체 임베딩을 만들려면 루트의
`build_embeddings_colab.ipynb`를 Colab에서 열어 위에서 아래 순서대로 실행하세요.
`chunk.7z`를 업로드하면 구조를 검증하고, 완료된 `rag.sqlite3`를 다운로드합니다.
Kaggle에서는 청크가 담긴 Dataset을 Input으로 추가한 뒤 `build_embeddings_kaggle.ipynb`를
실행하세요. GPU와 Internet을 켜고, 완료 후 Save Version의
Output 탭에서 `rag.sqlite3`를 내려받을 수 있습니다.

기본 동기화 설정은 `embedding_config.yaml`에 있습니다. `batch_size`, `device`, 모델 및
경로를 그 파일에서 바꿀 수 있으며, 명령행 옵션을 지정하면 그 값이 우선합니다. 예를 들어
GPU를 쓰려면 `device: cuda`, 배치 크기를 바꾸려면 `batch_size: 32`로 편집하세요. 메모리가
부족하면 batch_size를 낮추면 됩니다. 다른 설정 파일은
`python sync_embeddings.py --config 경로/설정.yaml`로 지정할 수 있습니다.

## 동기화 동작

sync_embeddings.py는 모든 chunk/**/chunks.jsonl을 읽고 id, text, metadata 전체의 SHA-256
해시를 비교합니다.

- 없는 ID는 임베딩 후 추가합니다.
- 해시가 바뀐 ID는 벡터와 메타데이터를 교체합니다.
- JSONL에서 없어진 ID는 벡터와 메타데이터 모두 삭제합니다.
- 변하지 않은 ID는 재임베딩하지 않습니다.

실행 시에는 먼저 추가·갱신·삭제·유지 수와 실제 임베딩 대상 수를 출력합니다. 이어지는
`임베딩 진행` 표시줄은 대상 청크 수 기준의 완료 수, 처리 속도, 경과 시간 및 예상 남은
시간을 보여 줍니다.

즉 chunks.jsonl이 DB의 단일 원천입니다. 결과 DB는 전문과 metadata JSON을 보관하는
`chunks`, cosine KNN용 1024차원 `vec_chunks` vec0, 그리고 BGE-M3의
`token_id → chunk_rowid → weight` lexical weight를 보관하는 `sparse_postings` 역색인으로
구성됩니다. 동기화는 바뀐 청크의 dense 벡터와 sparse posting을 함께 교체합니다. hybrid JSON
결과에는 `rrf_score` 및 각 모드의 `dense_rank`/`sparse_rank`가, 단일 모드 결과에는
`similarity` 또는 `sparse_score`가 포함됩니다.

기본 경로는 chunk/, .models/bge-m3/, vector_store/rag.sqlite3입니다. 모두 --chunk-dir,
--model-dir, --db-path 옵션으로 바꿀 수 있습니다. 이 프로젝트의 현재 스토어 형식은
hybrid 전용이므로 새로 임베딩할 때 다음처럼 생성 DB를 재구축하세요. data/에는 접근하지 않고
chunk/는 읽기만 합니다.

    python sync_embeddings.py --rebuild

외부 모델 다운로드 없이 기본 검증을 실행할 수 있습니다.

    python -m unittest discover -s tests -v

참고: [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3), [sqlite-vec Python 문서](https://alexgarcia.xyz/sqlite-vec/python.html)
