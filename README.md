# embedding-datas

chunk/**/chunks.jsonl만 원본으로 삼아 BGE-M3 임베딩을 만들고, 로컬 SQLite +
sqlite-vec에서 검색하는 간단한 RAG 검색 기반입니다. chunk/는 읽기만 수행하며,
생성 파일은 .models/와 vector_store/ 입니다.

## 빠른 시작

Python 3.10+ 기준입니다. GPU를 쓸 경우에는 환경에 맞는 PyTorch를 먼저 설치하세요.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    # 운영자가 한 번 실행: BAAI/bge-m3를 프로젝트에 저장
    python download_model.py

    # chunk/의 모든 chunks.jsonl을 DB와 동기화
    python sync_embeddings.py

    # 대화형 top-k 검색
    python search.py --top-k 5

검색 중 :q, quit, exit 또는 Ctrl-D/Ctrl-C를 입력하면 종료합니다. 한 번만 검색할 때는
python search.py --query "전자금융사기 피해금 환급 절차" --top-k 3처럼 실행합니다.
RAG 파이프라인에 결과를 전달할 때는 --json을 추가하세요.

CPU 대신 Colab GPU에서 전체 임베딩을 만들려면 루트의
`build_embeddings_colab.ipynb`를 Colab에서 열어 위에서 아래 순서대로 실행하세요.
`chunk.7z`를 업로드하면 구조를 검증하고, 완료된 `rag.sqlite3`를 다운로드합니다.

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
chunks 테이블과 cosine KNN용 1024차원 vec_chunks vec0 테이블로 구성됩니다. 검색
결과에는 id, text, metadata 전체, distance, similarity가 포함되며 --json 출력을 RAG
파이프라인에 바로 전달할 수 있습니다.

기본 경로는 chunk/, .models/bge-m3/, vector_store/rag.sqlite3입니다. 모두 --chunk-dir,
--model-dir, --db-path 옵션으로 바꿀 수 있습니다. 모델 또는 차원을 바꿀 때만 다음처럼
생성 DB를 재구축하세요. data/에는 접근하지 않고 chunk/는 읽기만 합니다.

    python sync_embeddings.py --rebuild

외부 모델 다운로드 없이 기본 검증을 실행할 수 있습니다.

    python -m unittest discover -s tests -v

참고: [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3), [sqlite-vec Python 문서](https://alexgarcia.xyz/sqlite-vec/python.html)
