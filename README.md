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

## Batching

Documents are grouped by count and by length, and the groups are scored one
after another rather than all at once. Sequential on purpose: a burst of
parallel calls against one API key is exactly what exhausts a per-minute quota,
which is the thing this was meant to relieve.

Because the rubric is absolute — each passage judged on its own, not against its
neighbours — scores from different batches are comparable. A rubric that asked
for a ranking instead would not have that property.

## When a batch fails

Nothing is invented. A batch that cannot be scored contributes no scores, the
response says so in `meta.warnings`, and the caller sees fewer results than
documents. Filling the gap with plausible numbers would be worse: they are
indistinguishable from real ones and would pass a threshold on equal terms.

## Comparing against the real thing

```bash
OPENROUTER_API_KEY=... python3 compare.py corpus.example.json --floor 0.11
```

Each case prints both rankings side by side, marks what each would keep at the
floor, and points at the rows where they disagree. The summary answers the only
question that matters in the end: with the threshold you actually run, does the
same set of passages survive?

## Install

```bash
cp config.example.json config.json   # edit the upstream and the model
python3 -m pip install -r requirements.txt
cp rerank.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now rerank
curl -s http://127.0.0.1:8790/health
```

`upstream_base_url` points at whatever serves the model. Nothing here holds an
API key for it: the service cannot leak a credential it was never given.

`api_key` guards incoming requests and is empty by default — on the loopback,
asking yourself for a password buys nothing.
