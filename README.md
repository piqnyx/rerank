# rerank

A rerank endpoint in Cohere's shape, answered by a cheap chat model.

Point a client that already speaks Cohere Rerank at this instead, and it keeps
working: same path, same request fields, same response. What changes is who
does the scoring and what it costs.

## Why

A hosted reranker charges per call, and in a retrieval-augmented setup the call
happens on every question. A tenth of a cent each is a dollar a week and fifteen
a month — not ruinous, and not nothing, for work a free model does acceptably.

## What this is not

It is not a cross-encoder and does not pretend to be one. A cross-encoder pushes
each query-passage pair through a network trained on ranking. Here a
general-purpose model reads the list and assigns numbers by a written rule. They
will not always agree, and `compare.py` exists so that the disagreement is
measured rather than assumed.

## The scale, and why it is what it is

The scores are not invented. They were taken from a live `cohere/rerank-v3.5`
answer, which places relevance like this:

| score | what the passage was |
|---|---|
| 0.84 | states the answer outright |
| 0.73 | answers, in more words |
| 0.16 | same subject, a different property of it |
| 0.10 | subject mentioned in passing, nothing answered |
| 0.09 | same property, a different subject |
| 0.02 | unrelated |

Two things follow. Relevant lands around **0.7–0.85**, not 0.95: the scale does
not saturate. And between "on topic" and "beside the point" there is a gap of
more than four times — roughly 0.16 to 0.73 — which is where a floor belongs.

The rubric in the prompt is anchored to those bands on purpose. A consumer has a
threshold tuned to this distribution; a model told merely to "score 0 to 100"
clusters its answers differently, and the same threshold would quietly come to
mean something else.

## Fidelity

The full request is accepted, not the subset one particular client happens to
send:

- `query`, `documents`, `model`
- `documents` as plain strings **or** objects, with `rank_fields` choosing which
  fields to read and in what order
- `top_n` limits the results returned
- `max_tokens_per_doc` truncates long documents, default 4096
- `return_documents` — documents are returned by default, as OpenRouter does

The response carries `results` with `index` and `relevance_score` sorted
descending, plus `id`, `model`, `usage` and `meta`. Under `meta.backend` are the
model that answered, how many batches it took and how long — useful when a
result looks wrong and the question is why.

Paths: `/rerank`, `/v1/rerank`, `/v2/rerank`.

## The model name is the caller's, not ours

Clients are configured against a real reranker and will keep asking for it by
name. That name is never validated here — refusing a caller for asking exactly
what it was set up to ask would be perverse — and it comes back in the response
unchanged, so the caller sees what it requested.

What answers it is chosen by `model_map`, falling back to `model`:

```json
"model": "gemini-3.5-flash-lite",
"model_map": { "cohere/rerank-v3.5": "gemini-3.5-flash-lite" }
```

An empty map is the normal case: everything is served by `model`. The map earns
its place only when different callers should reach different backing models.
`meta.backend.model` always says which one actually answered.

## Batching

Documents are grouped by count and by length, and the groups are scored one
after another rather than all at once. Sequential on purpose: a burst of
parallel calls against one API key is exactly what exhausts a per-minute quota,
which is the thing this was meant to relieve.

Because the rubric is absolute — each passage judged on its own, not against its
neighbours — scores from different batches are comparable. A rubric that asked
for a ranking instead would not have that property.

## Passages are data

Query and passages reach the model as JSON, not as a numbered list. A passage
comes out of whatever store the caller keeps, which means it is text somebody
else wrote. In a numbered list one newline splits it into two entries, and a
line reading like another entry's marker lets a passage claim someone else's
place in the answer. JSON closes both: the boundaries are declared, newlines are
escaped, and an index cannot be forged.

The rubric says as much in words too — text inside a passage that asks for a
score or claims authority is just more text to judge against the query, and a
passage spending itself on that is answering nothing.

## An answer is complete or it is not an answer

A batch is accepted only when the model returned exactly one score per passage.
A short list or a repeated index sends the batch back for another try.

This is not caution for its own sake. It was found in a live run: the model
named one index twice, the later entry overwrote the earlier one, and the
passage that actually answered the query went from 100 to 0 and sank to fourth
place — while a neighbouring passage lost its score entirely and simply vanished
from the ranking. Nothing about the response looked wrong from outside. A
partial answer reads exactly like "these passages were not relevant", which is a
different statement and a believable one.

## When a batch fails

Nothing is invented. A batch that cannot be scored contributes no scores, the
response says so in `meta.warnings`, and the caller sees fewer results than
documents. Filling the gap with plausible numbers would be worse: they are
indistinguishable from real ones and would pass a threshold on equal terms.

When **nothing** could be scored the answer is `502`, not an empty list with
`200`. An empty success reads as "none of these were relevant" and the caller
walks away satisfied, which is a different statement from "the scorer did not
answer".

## Comparing against the real thing

```bash
OPENROUTER_API_KEY=... python3 compare.py corpus.example.json --floor 0.11
```

Each case prints both rankings side by side, marks what each would keep at the
floor, and points at the rows where they disagree.

Mark up the corpus and the threshold stops being a guess:

```json
{ "query": "...", "documents": ["..."], "relevant": [0, 4] }
```

`relevant` lists the passages that ought to survive. With it, the summary sweeps
the floor from 0.05 to 0.95 and reports, for each, how many wanted passages each
reranker found and how many unwanted came with them — then names the floor that
costs least, counting a lost passage as twice the price of a spare one. A lost
passage disappears from memory; a spare one only takes up room.

Two things the harness deliberately does **not** do. It does not treat the
reference as ground truth — on one of the sample queries the hosted reranker
scored the actual answer third, below a passage about the size of the disk, and
kept nothing at all at the usual floor. And it does not count a query where
nothing is relevant as a disagreement: when every score is equal there is no
order to compare, and both sides keeping nothing is agreement, not conflict.

## Install

```bash
cp config.example.json config.json   # edit the upstream and the model
# Debian and its relatives refuse a system-wide pip (PEP 668). Either take the
# packages from the distribution, or put them in a virtual environment and point
# ExecStart at its python:
sudo apt install python3-fastapi python3-uvicorn python3-httpx
#   or: python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp rerank.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now rerank
curl -s http://127.0.0.1:8790/health
```

`upstream_base_url` points at whatever serves the model. Nothing here holds an
API key for it: the service cannot leak a credential it was never given.

`api_key` guards incoming requests and is empty by default — on the loopback,
asking yourself for a password buys nothing.
