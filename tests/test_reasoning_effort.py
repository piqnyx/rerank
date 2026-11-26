"""Уровень размышления модели -- из настройки `reasoning_effort`.

Без уровня 3.5-lite сегодня оценивает шумно: на замере 26.11 из 68 пассажей
ненулевые оценки получили 48, а с low -- около семи, и верх тот же, что у high.
Поэтому уровень есть всегда, low по умолчанию, и задаётся в конфиге, чтобы его
можно было сменить без правки кода. Пустая строка -- поле не уходит вовсе.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerank


class Provider:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def post(self, url, json=None):
        self.calls.append({"url": url, "body": json})
        return Answer(self.answer)


class Answer:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def scores(*pairs):
    text = json.dumps({"scores": [{"index": i, "score": s} for i, s in pairs]})
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def body_with(**settings):
    rerank.CONFIG.clear()
    rerank.CONFIG.update(dict(rerank.DEFAULTS))
    rerank.CONFIG.update(settings)
    provider = Provider(scores((0, 80), (1, 20)))
    asyncio.run(rerank.score_batch(provider, "запрос", ["раз", "два"], [0, 1], "test-model"))
    return provider.calls[0]["body"]


def load_with(settings):
    # Папка рядом с тестами, а не /tmp: на этой машине /tmp живёт в памяти.
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        path = Path(folder) / "config.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        return rerank.load_config(str(path))


class TheLevelInTheBodyTests(unittest.TestCase):
    def tearDown(self):
        rerank.CONFIG.clear()
        rerank.CONFIG.update(dict(rerank.DEFAULTS))

    def test_low_goes_by_default(self):
        self.assertEqual(body_with()["reasoning_effort"], "low")

    def test_the_configured_level_goes(self):
        self.assertEqual(body_with(reasoning_effort="high")["reasoning_effort"], "high")

    def test_an_empty_setting_sends_no_level_at_all(self):
        self.assertNotIn("reasoning_effort", body_with(reasoning_effort=""))


class TheSettingTests(unittest.TestCase):
    def test_the_default_is_low(self):
        self.assertEqual(rerank.DEFAULTS["reasoning_effort"], "low")

    def test_a_known_level_is_taken(self):
        self.assertEqual(load_with({"reasoning_effort": "medium"})["reasoning_effort"], "medium")

    def test_an_empty_level_is_taken(self):
        self.assertEqual(load_with({"reasoning_effort": ""})["reasoning_effort"], "")

    def test_a_misspelt_level_stops_the_start(self):
        # Опечатка иначе ушла бы в каждый запрос и получила бы отказ двери на
        # каждом -- реранк бы молча перестал работать.
        with self.assertRaises(SystemExit) as caught:
            load_with({"reasoning_effort": "hgih"})
        self.assertIn("must be one of", str(caught.exception))
        self.assertIn("hgih", str(caught.exception))

    def test_a_level_that_is_not_a_string_stops_the_start(self):
        with self.assertRaises(SystemExit) as caught:
            load_with({"reasoning_effort": 3})
        self.assertIn("must be one of", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
