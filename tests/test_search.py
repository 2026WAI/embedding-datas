from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from search import ANSI_RESET, _highlight_code_blocks, print_results


class SearchOutputTests(unittest.TestCase):
    def result(self) -> dict[str, object]:
        return {
            "id": "guide:42:0001",
            "similarity": 0.9876,
            "text": "설명입니다.\n```python\ndef hello():\n    return '안녕'\n```",
            "metadata": {
                "title": "예시 문서",
                "source": "guide",
                "item_id": "42",
                "chunk_type": "content",
            },
        }

    def test_plain_output_separates_query_result_and_body(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_results([self.result()], 1000, False, query="인사 함수", color="never")

        rendered = output.getvalue()
        self.assertIn("============ 검색 질의 ============", rendered)
        self.assertIn("============ 검색 결과 ============", rendered)
        self.assertIn("------------ 본문 ------------", rendered)
        self.assertIn("```python", rendered)
        self.assertNotIn(ANSI_RESET, rendered)

    def test_fenced_code_is_colored_when_available(self) -> None:
        text = "```python\nreturn 1\n```"
        rendered = _highlight_code_blocks(text, enabled=True)

        if "\033[" not in rendered:
            self.skipTest("Pygments가 설치되지 않은 환경")
        self.assertIn("return", rendered)
        self.assertIn(ANSI_RESET, rendered)


if __name__ == "__main__":
    unittest.main()
