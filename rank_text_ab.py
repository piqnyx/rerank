#!/usr/bin/env python3
"""
Чем кормить ранжировщик: репликой, репликой с хвостом разговора или всем сразу.
============================================================================

Извлечение и ранжирование задают разные вопросы. Кандидатов ищут по всему
разговору -- иначе не найдётся то, о чём речь шла минуту назад. А оценивают по
реплике, потому что спрошено именно в ней.

У этого есть цена, и она видна на эллипсисе: «а их?» -- два слова, из которых
ранжировщику не понять ничего, и он выдаёт что попало. Напрашивается добавить
хвост разговора. Напрашивается -- не значит верно: короткая реплика даёт очень
чистое разделение, и лишний контекст может его размыть, сделав «относится к
беседе» проще, чем «отвечает на вопрос».

Скрипт не рассуждает об этом, а меряет: один и тот же набор документов
оценивается тремя разными текстами, и видно, что каждый находит и сколько
лишнего приносит.
"""

import argparse
import json
import re
import sys
import urllib.request
from typing import Any, Dict, List, Optional


def remark_of(query: str) -> str:
    """Последняя реплика человека -- то, что спрошено прямо сейчас."""
    parts = re.findall(r"\[[^\]]+\]\s*(.+?)(?=\n\[|\Z)", query, re.S)
    return " ".join((parts[-1] if parts else query).split())


def tail_of(query: str, chars: int) -> str:
    """Хвост разговора вместе с репликой, обрезанный по границе строки."""
    if len(query) <= chars:
        return query
    cut = query[-chars:]
    newline = cut.find("\n")
    return cut[newline + 1:] if newline >= 0 else cut


def score(url: str, query: str, documents: List[str], key: Optional[str]) -> Dict[int, float]:
    body = json.dumps({"model": "cohere/rerank-v3.5", "query": query,
                       "documents": documents, "top_n": len(documents)}).encode()
    headers = {"content-type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with urllib.request.urlopen(
        urllib.request.Request(url, body, headers), timeout=120
    ) as response:
        out = json.load(response)
    return {r["index"]: float(r["relevance_score"]) for r in out.get("results", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнивает тексты для ранжирования")
    parser.add_argument("corpus")
    parser.add_argument("--url", default="http://127.0.0.1:8790/v1/rerank")
    parser.add_argument("--key", default=None)
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--tail", type=int, default=600, help="сколько знаков разговора добавлять")
    args = parser.parse_args()

    cases = json.load(open(args.corpus, encoding="utf-8"))
    if isinstance(cases, dict):
        cases = [cases]

    modes = ["реплика", f"реплика+хвост {args.tail}", "весь разговор"]
    totals = {m: {"found": 0, "extra": 0} for m in modes}
    wanted_total = 0

    for n, case in enumerate(cases):
        query = case["query"]
        documents = [d if isinstance(d, str) else str(d) for d in case["documents"]]
        wanted = set(case.get("relevant") or [])
        wanted_total += len(wanted)
        remark = remark_of(query)

        texts = {
            modes[0]: remark,
            modes[1]: tail_of(query, args.tail),
            modes[2]: query,
        }

        print(f"\n[{n}] реплика ({len(remark)} зн.): {remark[:88]}")
        print(f"     нужно найти: {len(wanted)}")
        for mode in modes:
            try:
                scores = score(args.url, texts[mode], documents, args.key)
            except Exception as error:
                print(f"     {mode:<22} не отработал: {type(error).__name__}")
                continue
            kept = {i for i, v in scores.items() if v >= args.floor}
            found = len(kept & wanted)
            extra = len(kept - wanted)
            totals[mode]["found"] += found
            totals[mode]["extra"] += extra
            top = sorted(scores, key=lambda i: -scores[i])[:1]
            best = f"{scores[top[0]]:.2f} {documents[top[0]][:44]}" if top else "—"
            print(f"     {mode:<22} нашёл {found}/{len(wanted)}, лишних {extra:<3} | верх: {best}")

    print(f"\nИтого при пороге {args.floor} (всего нужного {wanted_total})")
    for mode in modes:
        t = totals[mode]
        print(f"  {mode:<22} нашёл {t['found']:>3}, лишних {t['extra']:>3}")


if __name__ == "__main__":
    main()
