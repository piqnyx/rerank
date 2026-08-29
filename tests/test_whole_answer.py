"""Что уходит наружу, когда часть пачек не далась.

Сборка ответа не проверялась ничем: код 502 можно было заменить на 200, а
предупреждение выкинуть -- набор оставался зелёным. А цена этому не в коде.

Единственный, кто нас читает -- клиент графити, -- берёт `results` и в `meta`
не смотрит. Укороченный список он поэтому прочтёт как «эти отрывки не подошли»,
что совсем другое утверждение и вполне правдоподобное. Отрывки, которые не
удалось оценить, молча уходят из выдачи -- ровно из тех полусотни кандидатов,
ради которых выдачу и поднимали до полусотни.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerank


class PartialRankingTests(unittest.TestCase):
    def setUp(self):
        rerank.CONFIG.clear()
        rerank.CONFIG.update(dict(rerank.DEFAULTS))
        # По одному отрывку в пачке: так у каждого своя судьба.
        rerank.CONFIG["batch_documents"] = 1
        self.original = rerank.score_batch

    def tearDown(self):
        rerank.score_batch = self.original

    def asked(self, count=3):
        return rerank.parse_request({
            "query": "запрос",
            "documents": [f"отрывок {n}" for n in range(count)],
        })

    def answering(self, skip=()):
        async def score(client, query, texts, group, model):
            return {i: 1.0 - i / 10 for i in group if i not in skip}

        rerank.score_batch = score

    def test_a_complete_ranking_comes_back(self):
        self.answering()
        body, code = asyncio.run(rerank.do_rerank(self.asked()))
        self.assertEqual(code, 200)
        self.assertEqual(len(body["results"]), 3)

    def test_a_partial_ranking_is_refused(self):
        self.answering(skip=(1,))
        body, code = asyncio.run(rerank.do_rerank(self.asked()))
        self.assertEqual(code, 502, "неполную выдачу отдали за полную")
        self.assertNotIn("results", body)

    def test_the_refusal_says_how_much_was_lost(self):
        self.answering(skip=(1,))
        body, _ = asyncio.run(rerank.do_rerank(self.asked()))
        self.assertIn("2 of 3", body["message"])
        self.assertEqual(body["meta"]["scored"], 2)
        self.assertEqual(body["meta"]["documents"], 3)

    def test_nothing_scored_is_refused_too(self):
        self.answering(skip=(0, 1, 2))
        _, code = asyncio.run(rerank.do_rerank(self.asked()))
        self.assertEqual(code, 502)

    def test_top_n_does_not_hide_a_shortfall(self):
        # top_n = 1 всё равно вернул бы один отрывок, и снаружи это выглядело бы
        # как совершенно нормальный ответ.
        self.answering(skip=(1,))
        asked = self.asked()
        asked["top_n"] = 1
        _, code = asyncio.run(rerank.do_rerank(asked))
        self.assertEqual(code, 502)


class TheTwoNumberingsTests(unittest.TestCase):
    """Номер внутри пачки и номер во всём запросе -- разные числа.

    Проверялись они только там, где совпадают: на первой пачке. Модели в
    подсказке даются номера по порядку внутри пачки, и подстановка сквозного
    номера проходила весь набор -- а стоила бы целой пачки, потому что каждый
    вернувшийся номер не прошёл бы проверку границ и был бы отброшен.
    """

    def setUp(self):
        rerank.CONFIG.clear()
        rerank.CONFIG.update(dict(rerank.DEFAULTS))

    def test_a_later_batch_is_numbered_from_zero(self):
        texts = [f"отрывок {n}" for n in range(4)]
        prompt = rerank.build_prompt("запрос", texts, [2, 3])
        self.assertIn('"index": 0', prompt)
        self.assertIn('"index": 1', prompt)
        self.assertNotIn('"index": 2', prompt)
        self.assertNotIn('"index": 3', prompt)

    def test_the_text_is_the_one_the_index_points_at(self):
        texts = [f"отрывок {n}" for n in range(4)]
        prompt = rerank.build_prompt("запрос", texts, [2, 3])
        self.assertIn("отрывок 2", prompt)
        self.assertIn("отрывок 3", prompt)
        self.assertNotIn("отрывок 0", prompt)

    def test_the_count_asked_for_is_the_batch_and_not_the_whole(self):
        texts = [f"отрывок {n}" for n in range(4)]
        prompt = rerank.build_prompt("запрос", texts, [2, 3])
        self.assertIn("Score all 2 passages", prompt)


if __name__ == "__main__":
    unittest.main()
