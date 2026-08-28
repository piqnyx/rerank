"""Как служба разговаривает с провайдером и что делает с его ответом.

Раньше она ходила в родную гугловую дверь и брала текст из `candidates[0]`
вслепую. Дверь теперь одна, openai-совместимая, а ответ на ней может прийти
успешным и пустым внутри -- отказ, обрыв, пустота, -- и каждый такой случай
должен называться своим словом, иначе из лога не понять, почему пачка не далась.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerank


def setUpModule():
    rerank.CONFIG.clear()
    rerank.CONFIG.update(dict(rerank.DEFAULTS))


class Provider:
    """Провайдер, отвечающий тем, что ему велели, и хранящий, о чём его просили."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    async def post(self, url, json=None):
        self.calls.append({"url": url, "body": json})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return Answer(answer)


class Answer:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def said(text):
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def scores(*pairs):
    return said(json.dumps({"scores": [{"index": i, "score": s} for i, s in pairs]}))


def run(provider, texts=("раз", "два"), indexes=(0, 1)):
    return asyncio.run(
        rerank.score_batch(provider, "запрос", list(texts), list(indexes), "test-model")
    )


class AnswerTextTests(unittest.TestCase):
    def test_an_ordinary_answer_comes_back_whole(self):
        self.assertEqual(rerank.answer_text(said('{"scores": []}')), '{"scores": []}')

    def test_a_fenced_answer_is_unwrapped(self):
        # Провайдеры этой двери оборачивают ответ в ограду даже при заданной
        # схеме, и голый json.loads на этом ломается.
        body = said('```json\n{"scores": []}\n```')
        self.assertEqual(rerank.answer_text(body), '{"scores": []}')

    def test_no_choices_at_all_is_named(self):
        with self.assertRaises(ValueError) as caught:
            rerank.answer_text({"choices": []})
        self.assertIn("no choices", str(caught.exception))

    def test_an_empty_answer_names_the_reason(self):
        body = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        with self.assertRaises(ValueError) as caught:
            rerank.answer_text(body)
        self.assertIn("length", str(caught.exception))

    def test_a_refusal_is_not_read_as_an_answer(self):
        body = {"choices": [{"finish_reason": "content_filter", "message": {}}]}
        with self.assertRaises(ValueError) as caught:
            rerank.answer_text(body)
        self.assertIn("content_filter", str(caught.exception))


class TheRequestTests(unittest.TestCase):
    def test_it_goes_to_the_openai_door(self):
        provider = Provider(scores((0, 80), (1, 20)))
        run(provider)
        self.assertTrue(provider.calls[0]["url"].endswith("/v1beta/openai/chat/completions"))

    def test_the_body_is_an_openai_body(self):
        provider = Provider(scores((0, 80), (1, 20)))
        run(provider)
        body = provider.calls[0]["body"]
        self.assertEqual(body["model"], "test-model")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], rerank.SCORE_SCHEMA)

    def test_the_rubric_travels_as_the_system_message(self):
        provider = Provider(scores((0, 80), (1, 20)))
        run(provider)
        self.assertEqual(provider.calls[0]["body"]["messages"][0]["content"], rerank.RUBRIC)

    def test_strict_is_not_asked_for(self):
        # Строгий режим требует своего подмножества схемы, и на этой двери он не
        # нужен. Просить его -- значит получать 400 на схеме, которая работает.
        provider = Provider(scores((0, 80), (1, 20)))
        run(provider)
        self.assertNotIn("strict", provider.calls[0]["body"]["response_format"]["json_schema"])


class TheAnswerTests(unittest.TestCase):
    def test_scores_come_back_on_a_hundredth_scale(self):
        provider = Provider(scores((0, 100), (1, 0)))
        self.assertEqual(run(provider), {0: 1.0, 1: 0.0})

    def test_an_index_the_model_invented_is_ignored(self):
        provider = Provider(scores((0, 50), (1, 50), (7, 90)))
        self.assertEqual(sorted(run(provider)), [0, 1])

    def test_an_incomplete_set_is_not_taken(self):
        # Неполный ответ снаружи неотличим от «часть пассажей не подошла».
        provider = Provider(scores((0, 50)))
        self.assertEqual(run(provider), {})

    def test_a_repeated_index_is_not_taken(self):
        provider = Provider(scores((0, 50), (0, 90)))
        self.assertEqual(run(provider), {})

    def test_a_refusal_gives_nothing_rather_than_invented_scores(self):
        provider = Provider({"choices": [{"finish_reason": "content_filter", "message": {}}]})
        self.assertEqual(run(provider), {})

    def test_a_batch_that_failed_once_is_tried_again(self):
        rerank.CONFIG["retries"] = 1
        provider = Provider({"choices": []}, scores((0, 40), (1, 60)))
        self.assertEqual(run(provider), {0: 0.4, 1: 0.6})
        self.assertEqual(len(provider.calls), 2)


if __name__ == "__main__":
    unittest.main()
