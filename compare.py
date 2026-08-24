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


def flat(scores: Dict[int, float]) -> bool:
    """Все оценки одинаковы -- значит порядка нет вовсе.

    Так выглядит правильный ответ на запрос, которому не подходит ничто. Место
    в таком списке присвоено произвольно, и сравнивать эти места с чужими --
    значит выдать согласие за расхождение. У нас так и вышло: запрос, где обе
    стороны честно не оставили ни одного пассажа, был засчитан как разногласие
    и утянул среднее вниз.
    """
    values = set(round(v, 6) for v in scores.values())
    return len(values) <= 1


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

    # Ничья с обеих сторон -- согласие, а не разногласие.
    tie = flat(theirs) or flat(ours)
    if tie and kept_t == 0 and kept_o == 0:
        rho, top1 = None, "оба пусты"
    else:
        rho = spearman(tr, orr)
        top1 = "совпал" if (tr and orr and min(tr, key=tr.get) == min(orr, key=orr.get)) else "РАЗНЫЙ"

    print(f"  порядок: ρ={('—' if rho is None else f'{rho:+.2f}')}   первый: {top1}   "
          f"порог {args.floor}: эталон оставит {kept_t}, наш {kept_o}")

    truth = case.get("relevant")
    if truth is not None:
        marks = []
        for name, scores in (("эталон", theirs), ("наш", ours)):
            kept = {i for i, v in scores.items() if v >= args.floor}
            hit = len(kept & set(truth))
            marks.append(f"{name}: нашёл {hit} из {len(truth)}, лишних {len(kept) - hit}")
        print("  по разметке — " + ";  ".join(marks))

    return case["query"], {
        "rho": rho, "top1_same": top1 == "совпал", "tie": tie,
        "kept_reference": kept_t, "kept_ours": kept_o, "total": total,
        "truth": set(truth) if truth is not None else None,
        "theirs": theirs, "ours": ours,
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

    # Ничьи из среднего исключаются: там порядка нет ни у кого, и считать по
    # ним согласие -- значит портить среднее артефактом.
    ranked = [s for _, s in summary if not s["tie"]]
    rhos = [s["rho"] for s in ranked if s["rho"] is not None]
    same_top = sum(1 for s in ranked if s["top1_same"])
    kept_ref = sum(s["kept_reference"] for _, s in summary)
    kept_our = sum(s["kept_ours"] for _, s in summary)

    print("\n\033[1mИтого\033[0m")
    print(f"  случаев: {len(summary)}, из них с ничьёй: {len(summary) - len(ranked)}")
    if ranked:
        print(f"  первый пассаж совпал: {same_top} из {len(ranked)}")
    if rhos:
        print(f"  согласие порядков в среднем: ρ={sum(rhos)/len(rhos):+.2f}  (ничьи не в счёт)")
    print(f"  с порогом {args.floor} эталон пропустил бы {kept_ref}, наш {kept_our}")

    labelled = [s for _, s in summary if s["truth"] is not None]
    if not labelled:
        print("\n  Чтобы подобрать порог числом, а не на глаз, разметь корпус:")
        print('  добавь в случай поле "relevant": [индексы пассажей, которые должны пережить отсечку]')
        return

    print("\n\033[1mКакой порог отсекает лучше\033[0m  (по разметке, "
          f"{len(labelled)} случаев из {len(summary)})")
    print(f"  {'порог':>6}  {'эталон: нашёл / лишних':>26}  {'наш: нашёл / лишних':>24}")
    rows = []
    for step in range(1, 20):
        floor = step / 20
        row = {}
        for name in ("theirs", "ours"):
            hit = miss = extra = 0
            for s in labelled:
                kept = {i for i, v in s[name].items() if v >= floor}
                hit += len(kept & s["truth"])
                miss += len(s["truth"] - kept)
                extra += len(kept - s["truth"])
            row[name] = (hit, miss, extra)
        th, tm, te = row["theirs"]
        oh, om, oe = row["ours"]
        # Промах дороже лишнего: потерянный факт исчезает из памяти совсем, а
        # лишний всего лишь занимает место в контексте.
        rows.append((floor, om * 2 + oe, th, te, oh, oe))

    for floor, _, th, te, oh, oe in rows:
        print(f"  {floor:>6.2f}  {f'{th} / {te}':>26}  {f'{oh} / {oe}':>24}")

    # Не первая лучшая точка, а середина полосы, где лучший счёт держится.
    #
    # Край плато -- плохой выбор: шаг в сторону, и качество падает. Корпус
    # всегда мал по сравнению с тем, что придёт в жизни, так что ставить надо
    # подальше от обоих обрывов.
    cheapest = min(r[1] for r in rows)
    plateau = [r[0] for r in rows if r[1] == cheapest]
    if plateau:
        low, high = min(plateau), max(plateau)
        middle = min(plateau, key=lambda f: abs(f - (low + high) / 2))
        span = f"{low:.2f}" if low == high else f"{low:.2f}–{high:.2f}"
        print(f"\n  лучший счёт держится на {span}; ставить стоит в середину: "
              f"\033[1m{middle:.2f}\033[0m")
        if low == high:
            print("  \033[33mполоса шириной в одну точку — качество упадёт от любого сдвига\033[0m")


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
