from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from rag_store import load_chunks
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
                "device: cuda\n",
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

    def test_rejects_unknown_config_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "embedding_config.yaml"
            config.write_text("unknown: value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "알 수 없는 설정 항목"):
                resolve_args(self.arguments(config))


if __name__ == "__main__":
    unittest.main()
