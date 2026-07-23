from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_store import load_chunks


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


if __name__ == "__main__":
    unittest.main()
