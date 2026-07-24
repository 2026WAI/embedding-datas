from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rag_store import load_chunks, search, search_hybrid, search_sparse
from sync_embeddings import resolve_args


class LoadChunksTests(unittest.TestCase):
    def write_jsonl(self, root: Path, name: str, records: list[dict[str, object]]) -> None:
        path = root / name / "chunks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")

    def test_loads_records_and_hash_tracks_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"id": "source:1:0000", "text": "본문", "metadata": {"source": "source", "item_id": "1"}}
            self.write_jsonl(root, "source/1", [record])
            first = load_chunks(root)
            record["metadata"] = {"source": "source", "item_id": "1", "title": "변경"}
            self.write_jsonl(root, "source/1", [record])
            self.assertNotEqual(first["source:1:0000"].content_hash, load_chunks(root)["source:1:0000"].content_hash)

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"id": "source:1:0000", "text": "본문", "metadata": {}}
            self.write_jsonl(root, "one/1", [record])
            self.write_jsonl(root, "two/1", [record])
            with self.assertRaisesRegex(ValueError, "중복 id"):
                load_chunks(root)


class EmbeddingConfigTests(unittest.TestCase):
    def arguments(self, config: Path) -> Namespace:
        return Namespace(
            config=config,
            chunk_dir=None,
            db_path=None,
            model_dir=None,
            model_id=None,
            batch_size=None,
            device=None,
            devices=None,
            rebuild=False,
        )

    def test_config_paths_are_relative_to_config_file_and_cli_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "embedding_config.yaml"
            config.write_text(
                "chunk_dir: source_chunks\n"
                "db_path: data/index.sqlite3\n"
                "model_dir: models/bge\n"
                "model_id: example/model\n"
                "batch_size: 8\n"
                "device: cuda\n"
                "progress: log\n",
                encoding="utf-8",
            )
            args = self.arguments(config)
            args.batch_size = 32
            resolved = resolve_args(args)

            self.assertEqual(resolved.chunk_dir, root / "source_chunks")
            self.assertEqual(resolved.db_path, root / "data/index.sqlite3")
            self.assertEqual(resolved.model_dir, root / "models/bge")
            self.assertEqual(resolved.model_id, "example/model")
            self.assertEqual(resolved.batch_size, 32)
            self.assertEqual(resolved.device, "cuda")
            self.assertEqual(resolved.progress, "log")
            self.assertEqual(resolved.hybrid_dense_candidates, 200)

    def test_rejects_unknown_config_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "embedding_config.yaml"
            config.write_text("unknown: value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "알 수 없는 설정 항목"):
                resolve_args(self.arguments(config))


class SparseInvertedIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata_json TEXT NOT NULL);
            CREATE TABLE sparse_postings (
                token_id TEXT NOT NULL,
                chunk_rowid INTEGER NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (token_id, chunk_rowid)
            ) WITHOUT ROWID;
            """
        )
        for chunk_id, text in (("a", "첫 청크"), ("b", "둘째 청크"), ("c", "셋째 청크")):
            self.connection.execute(
                "INSERT INTO chunks(id, text, metadata_json) VALUES (?, ?, '{}')", (chunk_id, text)
            )
        self.connection.executemany(
            "INSERT INTO sparse_postings(token_id, chunk_rowid, weight) VALUES (?, ?, ?)",
            [
                ("10", 1, 0.2),
                ("20", 1, 0.8),
                ("10", 2, 0.9),
                ("20", 3, 0.9),
            ],
        )

    def tearDown(self) -> None:
        self.connection.close()

    def test_sparse_search_scores_postings_by_lexical_inner_product(self) -> None:
        results = search_sparse(self.connection, {"10": 0.5, "20": 0.5}, 3)

        self.assertEqual([result["id"] for result in results], ["a", "b", "c"])
        self.assertAlmostEqual(float(results[0]["sparse_score"]), 0.5)
        self.assertAlmostEqual(float(results[1]["sparse_score"]), 0.45)

    def test_sparse_search_restricts_scoring_to_dense_candidates(self) -> None:
        results = search_sparse(self.connection, {"10": 0.5, "20": 0.5}, 3, candidate_ids=["b", "c"])

        self.assertEqual([result["id"] for result in results], ["b", "c"])

    def test_hybrid_reranks_only_dense_candidates_with_weighted_scores(self) -> None:
        dense_results = [
            {"id": "b", "text": "둘째 청크", "metadata": {}, "dense_score": 0.5},
            {"id": "c", "text": "셋째 청크", "metadata": {}, "dense_score": 0.9},
        ]
        with patch("rag_store.search_dense", return_value=dense_results):
            results = search_hybrid(
                self.connection,
                None,
                {"10": 1.0},
                2,
                dense_candidates=2,
                dense_weight=0.2,
                sparse_weight=0.8,
            )

        self.assertEqual([result["id"] for result in results], ["b", "c"])
        self.assertAlmostEqual(float(results[0]["hybrid_score"]), 0.8)

    def test_search_dispatches_sparse_mode(self) -> None:
        results = search(self.connection, None, {"20": 1.0}, 2, mode="sparse")

        self.assertEqual([result["id"] for result in results], ["c", "a"])


if __name__ == "__main__":
    unittest.main()
