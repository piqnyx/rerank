#!/usr/bin/env python3
"""
Сверка двух реранкеров на одном корпусе.
============================================================================

Смысл не в том, чтобы получить одинаковые числа -- их не будет. Кросс-энкодер и
языковая модель считают по-разному, и совпадения оценка в оценку ждать нечего.
Смысл в том, совпадает ли то, что следует из этих чисел: тот же ли пассаж
оказался первым, тот же ли набор пережил отсечку. Именно это и ломается тихо,
если шкала уехала, и именно это здесь и показывается.

Порог берётся тот же, что стоит у потребителя, и колонка «оставит» отвечает на
единственный вопрос, который в конце концов важен: что дойдёт до модели.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx


def ranks(results: List[Dict[str, Any]], total: int) -> Dict[int, int]:
    """Место каждого документа, считая с единицы. Неоценённые -- в конец."""
    order = [r["index"] for r in sorted(results, key=lambda r: -r["relevance_score"])]
    places = {index: place for place, index in enumerate(order, 1)}
    for i in range(total):
        places.setdefault(i, total)
    return places


def spearman(a: Dict[int, int], b: Dict[int, int]) -> Optional[float]:
    """Согласие порядков. Единица -- порядок тот же, ноль -- ничего общего."""
    n = len(a)
    if n < 2:
        return None
    diff = sum((a[i] - b[i]) ** 2 for i in a)
    return 1 - (6 * diff) / (n * (n * n - 1))


async def call(url: str, payload: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}"} if key else None
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def scores_of(body: Dict[str, Any]) -> Dict[int, float]:
    out = {}
    for row in body.get("results") or []:
        index = row.get("index")
        score = row.get("relevance_score", row.get("score"))
        if isinstance(index, int) and isinstance(score, (int, float)):
            out[index] = float(score)
    return out


async def one_case(case: Dict[str, Any], args) -> Tuple[str, Dict[str, Any]]:
    documents = case["documents"]
    payload = {"query": case["query"], "documents": documents, "top_n": len(documents)}

    theirs_body = await call(args.reference, {**payload, "model": args.reference_model}, args.reference_key)
    ours_body = await call(args.ours, {**payload, "model": args.reference_model}, args.ours_key)

    theirs, ours = scores_of(theirs_body), scores_of(ours_body)
    total = len(documents)
    tr, orr = ranks(theirs_body.get("results") or [], total), ranks(ours_body.get("results") or [], total)

    print(f"\n\033[1mЗапрос:\033[0m {case['query']}")
    print(f"  {'#':>2}  {'эталон':>16}  {'наш':>16}   пассаж")
    print(f"  {'':>2}  {'место  оценка':>16}  {'место  оценка':>16}")
    for index in sorted(range(total), key=lambda i: tr[i]):
        t_score = theirs.get(index)
        o_score = ours.get(index)
        t_keep = "✓" if t_score is not None and t_score >= args.floor else "·"
        o_keep = "✓" if o_score is not None and o_score >= args.floor else "·"
        same = "" if t_keep == o_keep else "  ← расходятся"
        text = documents[index] if isinstance(documents[index], str) else str(documents[index])
        print(
            f"  {index:>2}  {tr[index]:>5} {('—' if t_score is None else f'{t_score:.3f}'):>7} {t_keep}"
            f"  {orr[index]:>5} {('—' if o_score is None else f'{o_score:.3f}'):>7} {o_keep}"
            f"   {text[:52]}{same}"
        )

    kept_t = sum(1 for v in theirs.values() if v >= args.floor)
    kept_o = sum(1 for v in ours.values() if v >= args.floor)
    top1 = "совпал" if (tr and orr and min(tr, key=tr.get) == min(orr, key=orr.get)) else "РАЗНЫЙ"
    rho = spearman(tr, orr)
    print(f"  порядок: ρ={('—' if rho is None else f'{rho:+.2f}')}   первый: {top1}   "
          f"порог {args.floor}: эталон оставит {kept_t}, наш {kept_o}")

    return case["query"], {
        "rho": rho, "top1_same": top1 == "совпал",
        "kept_reference": kept_t, "kept_ours": kept_o, "total": total,
    }


async def main_async(args) -> None:
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    cases = corpus if isinstance(corpus, list) else [corpus]

    summary = []
    for case in cases:
        try:
            summary.append(await one_case(case, args))
        except Exception as error:
            print(f"\n  запрос «{case.get('query')}» не отработал: {type(error).__name__}: {error}")

    if not summary:
        return
    rhos = [s["rho"] for _, s in summary if s["rho"] is not None]
    same_top = sum(1 for _, s in summary if s["top1_same"])
    kept_ref = sum(s["kept_reference"] for _, s in summary)
    kept_our = sum(s["kept_ours"] for _, s in summary)
    print("\n\033[1mИтого\033[0m")
    print(f"  случаев: {len(summary)}")
    print(f"  первый пассаж совпал: {same_top} из {len(summary)}")
    if rhos:
        print(f"  согласие порядков в среднем: ρ={sum(rhos)/len(rhos):+.2f}")
    print(f"  с порогом {args.floor} эталон пропустил бы {kept_ref}, наш {kept_our}")
    if kept_ref and abs(kept_our - kept_ref) / kept_ref > 0.3:
        print("  \033[33mрасхождение по отсечке больше трети — порог придётся пересчитать\033[0m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сверяет наш реранкер с эталонным")
    parser.add_argument("corpus", help="JSON: объект {query, documents} или список таких")
    parser.add_argument("--ours", default="http://127.0.0.1:8790/v1/rerank")
    parser.add_argument("--ours-key", default=os.environ.get("RERANK_API_KEY") or None)
    parser.add_argument("--reference", default="https://openrouter.ai/api/v1/rerank")
    parser.add_argument("--reference-model", default="cohere/rerank-v3.5")
    parser.add_argument("--reference-key", default=os.environ.get("OPENROUTER_API_KEY") or None)
    parser.add_argument("--floor", type=float, default=0.11,
                        help="порог отсечки у потребителя; по умолчанию тот, что стоит в graphiti")
    args = parser.parse_args()
    if not args.reference_key:
        sys.exit("нужен ключ эталона: переменная OPENROUTER_API_KEY или --reference-key")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
