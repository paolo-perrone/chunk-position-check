# chunk-position-check

> Tested by replaying the seven recorded sweeps from Liu et al. and by 25 unit tests, including full sweeps against a local mock model, on Python 3.9 and 3.14. Not yet run against a real model API.

Companion to [What is a Transformer?](https://theaiengineer.substack.com/p/what-is-a-transformer), The AI Engineer.

For engineers running RAG who want to say why a document placed tenth of twenty
helps their model less than the same document placed first or last, measured on
their own model before they reorder chunks or raise top-k on a hunch.

It reads your RAG eval file (question, retrieved chunks, expected answer), moves
the chunk that answers each question through every slot while the other chunks
keep their retrieval order, and asks your model each time. It also asks with no
chunks (closed-book) and with the answer chunk alone (oracle).

It prints accuracy per slot with a 95% interval, both baselines, and one VERDICT
line: where your top chunk belongs and whether to cut k.

First output needs no API key and no GPU: `python3 chunk_position_check.py replay`
replays Liu et al.'s recorded GPT-3.5-Turbo sweep and ends on a known VERDICT line.

## First run: no key, no GPU

```
git clone https://github.com/paolo-perrone/chunk-position-check
cd chunk-position-check
python3 chunk_position_check.py replay
```

Python 3.9 or newer, standard library only, nothing to install.

## Check it before you trust it

The replay must print exactly this. Table 6 of the paper prints 75.8, 57.2,
53.8, 55.4 and 63.2 for the 20-document sweep, and Table 1 prints 56.1
closed-book and 88.3 oracle, so every number below can be checked against the
PDF.

```
chunk_position_check 1.0.0
model    : GPT-3.5-Turbo (0613), 4K window, recorded by Liu et al. 2023 (Tables 1, 5 and 6)
k=10
  slot  1/10  76.8%  [75.2%, 78.4%]  n=2655  ###############################
  slot  5/10  61.2%  [59.3%, 63.0%]  n=2655  ########################
  slot 10/10  62.4%  [60.5%, 64.2%]  n=2655  #########################
k=20  (your k)
  slot  1/20  75.8%  [74.1%, 77.4%]  n=2655  ##############################
  slot  5/20  57.2%  [55.3%, 59.1%]  n=2655  #######################
  slot 10/20  53.8%  [51.9%, 55.7%]  n=2655  ######################  below closed-book
  slot 15/20  55.4%  [53.5%, 57.3%]  n=2655  ######################  below closed-book
  slot 20/20  63.2%  [61.3%, 65.0%]  n=2655  #########################
closed-book (no chunks)           56.1%  [54.2%, 58.0%]  n=2655
oracle (answer chunk alone)       88.3%  [87.0%, 89.5%]  n=2655
middle cost at k=20: slot 10 scores 22.0 points below slot 1
distractor cost at k=20: the best slot scores 12.5 points below the answer chunk alone
VERDICT: put your top chunk in slot 1 (75.8%); slots 10 and 15 score below closed-book (56.1%); cut k from 20 to 10, whose worst slot scores 61.2%, if your retriever still finds the answer chunk in its top 10.
```

`replay --list` shows the seven recorded models. They disagree, which is the
reason to measure your own: `replay --model claude-1.3` puts the top chunk in
slot 20, and `replay --model llama-2-70b-chat` scores slot 20 at 69.5% against
56.8% for slot 1.

`python3 -m unittest discover -s tests` runs 25 tests, including full sweeps
against a mock endpoint that misses any answer chunk sitting in the middle of a
prompt longer than five chunks.

## Run it on your eval file and your model

```
export OPENAI_API_KEY=...
python3 chunk_position_check.py sweep --eval my_eval.jsonl --model gpt-4.1-mini --k 20
```

Any endpoint that serves OpenAI-style `/chat/completions` works, including
vLLM, Ollama, LM Studio and OpenRouter:

```
python3 chunk_position_check.py sweep --eval my_eval.jsonl --model llama3.1:8b \
    --base-url http://localhost:11434/v1 --k 20
```

1. **Your prompt.** `--prompt-template prompt.txt` sends your production prompt;
   it needs `{search_results}` and `{question}`. The default is Liu et al.'s.
   `--system` adds your system prompt.
2. **Your k.** `--k` is the top-k your pipeline uses. The sweep also measures
   half of it, so the verdict can say whether cutting k pays; `--k-grid 5,10,20`
   measures any set you name, and `--k-grid 20` measures your k alone.
3. **The bill.** Calls = questions x (slots measured + 2 baselines), five slots
   per k. The default, 200 questions at `--k 20`, measures k=10 and k=20 in
   2,400 calls; `--k-grid 20` makes 1,400. `--dry-run` prints the count, the
   input tokens still to send and one full prompt before spending anything, and
   `--limit` sets the sample size. On `--eval public` at `--k 20` the dry run
   estimates about 5.0M input tokens.
4. **Speed.** `--workers` sets how many calls run at once, 4 by default. Raise
   it for a hosted API; rate-limit replies are retried with backoff.
5. **Stopping.** Every call lands in `sweep.jsonl` as it returns. Rerun the same
   command and it resumes where it stopped; `report sweep.jsonl` reprints the
   verdict without calling the model. Change the model, the eval file, k, the
   prompt or the sample and it refuses to mix the two runs.
6. **Reasoning models.** OpenAI's reasoning models refuse `max_tokens` and
   temperature 0; the tool switches to `max_completion_tokens` and says when it
   has to sample. Give them `--max-tokens 2000`, since thinking spends the
   budget. `<think>` blocks are cut before scoring, so an answer the model only
   considered is not counted.
7. **Local servers.** Ollama shortens prompts longer than its context setting
   instead of refusing them. Set its context length above your longest prompt;
   the report warns when prompt tokens stop growing with k.
8. **No eval file yet.** `--eval public` downloads Liu et al.'s 20-document
   NaturalQuestions set (about 15 MB, cached in `~/.cache/chunk-position-check`)
   and runs it against your model. At slots 1, 5, 10 and 20 the prompts it sends
   are byte-identical to the paper's own position files, 2,655 of 2,655 each,
   so your model's numbers sit next to the recorded ones.

`sweep` and `report` exit 1 when a slot scores below closed-book beyond noise
and 0 otherwise, so a CI job can rerun the sweep when the prompt or the model
changes.

## The eval file

One JSON object per line. `examples/example_eval.jsonl` has three.

```
{"question": "how many layers does llama 3 70b have", "answers": ["80"], "chunks": ["...", "...", "num_hidden_layers is 80"], "gold": 2}
```

1. `chunks` are your retriever's top chunks, in retrieval order. Each is a
   string or `{"title": ..., "text": ...}`.
2. `answers` are short strings a correct reply contains: a name, a number, a
   date. A full reference sentence never matches, and the report warns when the
   answer chunk alone scores under 50%.
3. `gold` is the 0-based index of the chunk that holds the answer. Leave it out
   and the tool takes the first chunk that contains an expected answer as a
   whole word.
4. Other chunks that also contain the answer are left out of that question's
   prompts, so only one copy moves and the model cannot read a copy that stayed
   behind.
5. A question whose chunks contain no expected answer is a retrieval miss: it
   is left out of the sweep and counted in the recall line.
6. Liu et al.'s own format (`ctxs` with `isgold` and `hasanswer`) loads as it is.

## Every number, by hand

For each question and slot, the tool takes the answer chunk out, keeps the
other chunks in retrieval order, puts the answer chunk back at that slot, keeps
the first k, and asks the model at temperature 0. A reply is correct when a
normalized expected answer appears inside the normalized reply (Liu et al.'s
best-subspan exact match: lowercase, no punctuation, no articles).

1. **Accuracy per slot** is correct calls over calls at that slot: count the
   `"correct": 1.0` lines in `sweep.jsonl`.
2. **The bracket** is the 95% Wilson score interval, formula in `wilson()`.
3. **Closed-book** asks the question with no chunks; **oracle** with the answer
   chunk alone.
4. **Middle cost** is the best slot minus the worst slot at your k.
5. **Distractor cost** is oracle minus the best slot: what the other chunks
   cost even in the best position.
6. **Retriever recall** is the share of questions whose answer chunk sits in
   the retriever's top k, read from `gold`.
7. **Pipeline estimate** is what your pipeline scores at each k measured. Each
   question's answer chunk sits at its retrieval rank (counted from the end
   when the best slot is the last one) and scores the measured curve there,
   along straight lines between measured slots. A question whose answer chunk
   ranks below k, or that the retriever missed, scores the closed-book rate.
   The estimate is the average over every question in your file.
8. **Within noise.** Every slot and closed-book answer the same questions, so
   the tool compares them question by question: the gap's 95% interval is the
   mean of the per-question differences, plus or minus 1.96 standard errors
   (`paired_gap()`). A gap whose interval reaches zero is within noise.
9. **The VERDICT** applies three rules in order. The best slot is the highest
   accuracy at your k, ties to the earlier slot, with the other end slot named
   when their gap is within noise. A slot is dead when it scores below
   closed-book, because there the chunks left the model worse than no chunks.
   The k to run is the one with the highest pipeline estimate, ties to the
   smaller k. Without retrieval ranks (Liu et al.'s format, or a replay), a
   dead slot beyond noise sends k to the largest smaller k measured with no
   dead slot; a dead slot within noise asks for more questions instead.

A replay holds the paper's aggregate numbers only, so it cannot run the
question-by-question comparison: there every slot below closed-book counts as
dead, as the paper reports it, and the end slots are compared by their Wilson
intervals. On your own documents closed-book sits near zero, so no slot dies
there, and the pipeline estimate is what decides k.

## What it does not tell you

1. It moves one answer chunk. Questions that need two chunks together are not
   tested.
2. Best-subspan match credits a reply that contains the answer while hedging
   around it. Read twenty lines of `sweep.jsonl` before you act.
3. The pipeline estimate scores a question whose answer chunk fell out of the
   top k at the closed-book rate. A prompt full of near misses can score below
   that, so treat estimates a point or two apart as a tie.
4. The recorded models are from 2023. Your model is the one that counts, and
   the report names the model version the server says it ran.
5. Small samples flip. A 2026 reproduction (Gabin et al., SIGIR) needed 2,000
   NaturalQuestions questions before the winning chunk order stopped changing
   between random samples. At the default 200, act only on gaps the report does
   not mark within noise.

## Support

Tested on 2026-09-28 on macOS with Python 3.9.6 (the stock `/usr/bin/python3`)
and Python 3.14; CI runs the tests on Python 3.9, 3.12 and 3.13. No
dependencies, so there is no lockfile to drift. Supported through 2027-09-30
for Python 3.9 to 3.14 and any OpenAI-compatible chat endpoint. Open an issue
if a verdict disagrees with what you see in production.

## Credits

The scoring metric, both prompt templates and every recorded number come from
Nelson F. Liu et al., [Lost in the Middle: How Language Models Use Long
Contexts](https://arxiv.org/abs/2307.03172) (TACL 2024), and their
[repository](https://github.com/nelson-liu/lost-in-the-middle), MIT license.
See `LICENSE`.
