#!/usr/bin/env python3
"""chunk_position_check.py: which slot in the prompt does your model actually read?

Moves the chunk that answers each question in your RAG eval file through the slots
of the prompt, keeps every other chunk in its retrieval order, asks your model each
time, and prints accuracy per slot, the closed-book and oracle baselines, and one
VERDICT line: where your top chunk belongs and whether to cut k.

    python3 chunk_position_check.py replay                      # no key, no GPU
    python3 chunk_position_check.py sweep --eval my_eval.jsonl --model gpt-4.1-mini
    python3 chunk_position_check.py report sweep.jsonl

Standard library only. The scoring metric and the default prompt are taken from
Liu et al., Lost in the Middle (github.com/nelson-liu/lost-in-the-middle, MIT
license), so a replay of their recorded numbers and a fresh sweep are comparable.

Exit status: sweep and report exit 1 when a slot scores below closed-book beyond
noise and 0 otherwise, so a CI job can gate a prompt or model change on it.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

__version__ = "1.0.0"

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDED = os.path.join(HERE, "recorded", "liu2023_sweeps.json")
PUBLIC_URL = ("https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/"
              "qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "chunk-position-check")

# Liu et al.'s qa.prompt and closedbook_qa.prompt, verbatim.
QA_PROMPT = ("Write a high-quality answer for the given question using only the provided "
             "search results (some of which might be irrelevant).\n\n{search_results}\n\n"
             "Question: {question}\nAnswer:")
CLOSEDBOOK_PROMPT = "Question: {question}\nAnswer:"


# ------------------------------------------------------------------ scoring (Liu et al.)
def normalize_answer(s):
    """SQuAD normalization, as in Liu et al.'s metrics.py: lowercase, drop punctuation,
    drop the articles a, an and the, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def best_subspan_em(prediction, answers):
    """1.0 when any normalized expected answer appears inside the normalized prediction."""
    pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(a) in pred for a in answers if normalize_answer(a)) else 0.0


def contains_answer(text, answers):
    """Stricter than the metric, for finding answer chunks: the answer must sit on word
    boundaries, so the answer 80 is found in 'is 80 layers' and not in '1980'."""
    padded = f" {normalize_answer(text)} "
    return any(f" {normalize_answer(a)} " in padded for a in answers if normalize_answer(a))


def strip_reasoning(text):
    """Drop <think> blocks that reasoning models served through Ollama or vLLM can leave in
    the reply, so an answer the model only considered while thinking is not scored."""
    return re.sub(r"<think>.*?(?:</think>|\Z)", " ", text or "", flags=re.S).strip()


def wilson(p, n, z=1.96):
    """95% Wilson score interval for an accuracy p (0 to 1) measured on n questions."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_gap(a, b):
    """Accuracy under condition a minus condition b on the questions both answered, with a
    95% interval from the per-question differences. a and b map question id to 0 or 1."""
    common = [q for q in a if q in b]
    n = len(common)
    if n < 2:
        return None
    d = [a[q] - b[q] for q in common]
    m = sum(d) / n
    h = 1.96 * math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return (m, m - h, m + h, n)


# ------------------------------------------------------------------ eval file
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def load_eval(path):
    """Read an eval file into [{qid, question, answers, chunks, gold, gold_rank, clash}].

    Two shapes are accepted, one JSON object per line:
      simple: {"question", "answers": [...], "chunks": [str or {"title","text"}], "gold": int}
              gold is the index of the answer chunk in chunks (your retrieval order);
              when it is missing, the first chunk containing an expected answer is used.
      Liu et al.: {"question", "answers", "ctxs": [{"title","text","isgold","hasanswer"}]}
    Questions whose chunks contain no answer are kept with gold None: they count as
    retrieval misses in the recall line and are left out of the position sweep.
    clash lists the other chunks that also hold an answer; they never serve as
    distractors, so only one copy of the answer moves.
    """
    items = []
    with _open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path} line {i}: not valid JSON ({e})")
            answers = r.get("answers", r.get("answer"))
            if isinstance(answers, str):
                answers = [answers]
            if not r.get("question") or not answers:
                raise SystemExit(f"{path} line {i}: needs a 'question' and a list of 'answers'")
            if "ctxs" in r:
                ctxs = r["ctxs"]
                chunks = [{"title": c.get("title", ""), "text": c.get("text", "")} for c in ctxs]
                gold = next((j for j, c in enumerate(ctxs) if c.get("isgold")), None)
                clash = [j for j, c in enumerate(ctxs) if j != gold and c.get("hasanswer")]
                rank = None          # Liu et al. insert the gold passage; its slot is not a retrieval rank
            elif "chunks" in r:
                chunks = r["chunks"]
                gold = r.get("gold")
                if gold is None:
                    gold = next((j for j, c in enumerate(chunks) if contains_answer(_chunk_text(c), answers)), None)
                elif not (isinstance(gold, int) and 0 <= gold < len(chunks)):
                    raise SystemExit(f"{path} line {i}: gold {gold!r} is not an index into its {len(chunks)} chunks")
                clash = [j for j, c in enumerate(chunks) if j != gold and contains_answer(_chunk_text(c), answers)]
                rank = gold
            else:
                raise SystemExit(f"{path} line {i}: needs 'chunks' (or Liu et al.'s 'ctxs')")
            items.append({"qid": str(r.get("id", i - 1)), "question": r["question"], "answers": answers,
                          "chunks": chunks, "gold": gold, "gold_rank": rank, "clash": clash})
    if not items:
        raise SystemExit(f"{path} holds no questions")
    return items


def _chunk_text(c):
    return c["text"] if isinstance(c, dict) else str(c)


def format_chunk(i, c):
    """Liu et al.'s document line when the chunk has a title, a plain numbered line otherwise."""
    if isinstance(c, dict) and c.get("title"):
        return f"Document [{i}](Title: {c['title']}) {c['text']}"
    return f"Document [{i}] {_chunk_text(c)}"


def arrange(item, k, slot):
    """The k chunks of one prompt: the answer chunk at 1-based `slot`, the first k-1 other
    chunks around it in their original order. None when the item has too few chunks."""
    if item["gold"] is None or not 1 <= slot <= k:
        return None
    skip = set(item.get("clash", ())) | {item["gold"]}
    others = [c for j, c in enumerate(item["chunks"]) if j not in skip]
    if len(others) < k - 1:
        return None
    others = others[:k - 1]
    return others[:slot - 1] + [item["chunks"][item["gold"]]] + others[slot - 1:]


def build_prompt(item, chunks, template=QA_PROMPT):
    docs = "\n".join(format_chunk(i + 1, c) for i, c in enumerate(chunks))
    return template.format(search_results=docs, question=item["question"])


def auto_slots(k):
    """First, three interior quarter points, last: 1, 5, 10, 15, 20 for k=20, the paper's slots."""
    if k <= 5:
        return list(range(1, k + 1))
    return sorted({1, round(k / 4), round(k / 2), round(3 * k / 4), k})


# ------------------------------------------------------------------ model calls
class OverWindow(Exception):
    """The endpoint refused a prompt as longer than the model's context window."""


_COMPAT = {"max_tokens_key": "max_tokens", "temperature": True}
_COMPAT_LOCK = threading.Lock()
_WINDOW = re.compile(r"context length|context window|too long|maximum.{0,40}tokens|reduce the length", re.I)


def _adapt(body, used_key, used_temp):
    """OpenAI's reasoning models reject max_tokens and temperature 0: switch once, say so."""
    low = body.lower()
    with _COMPAT_LOCK:
        if used_key == "max_tokens" and "max_completion_tokens" in low:
            _COMPAT["max_tokens_key"] = "max_completion_tokens"
            return True
        if used_temp and "temperature" in low:
            if _COMPAT["temperature"]:
                _COMPAT["temperature"] = False
                print("note: this model rejects temperature 0, so its replies are sampled and a gap "
                      "of a few points between slots can be noise", file=sys.stderr)
            return True
    return False


def _retry_after(err):
    try:
        return float(err.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None


def call_model(base_url, model, prompt, key, max_tokens=100, system=None, timeout=120, retries=5):
    """One chat-completions call to any OpenAI-compatible endpoint, temperature 0.
    Returns the reply, the model name the server reports, and the token usage."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = base_url.rstrip("/") + "/chat/completions"
    attempt = 0
    while True:
        used_key, used_temp = _COMPAT["max_tokens_key"], _COMPAT["temperature"]
        payload = {"model": model, "messages": messages, used_key: max_tokens}
        if used_temp:
            payload["temperature"] = 0
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return (out["choices"][0]["message"].get("content") or "", out.get("model") or "",
                    out.get("usage") or {})
        except urllib.error.HTTPError as e:
            body = e.read()[:1000].decode("utf-8", "replace")
            if e.code == 400 and _adapt(body, used_key, used_temp):
                continue
            if e.code in (400, 413) and _WINDOW.search(body):
                raise OverWindow(body[:200])
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < retries - 1:
                attempt += 1
                time.sleep(min(_retry_after(e) or 5 * 2 ** (attempt - 1), 120))
                continue
            raise SystemExit(f"model call failed: HTTP {e.code} from {url}: {body[:300]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                attempt += 1
                time.sleep(5 * 2 ** (attempt - 1))
                continue
            raise SystemExit(f"model call failed: {e} ({url})")


# ------------------------------------------------------------------ the verdict
def pct(x):
    return f"{100 * x:.1f}%"


def join_slots(slots):
    s = [str(x) for x in slots]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


def split_dead(curves, closedbook, k, paired=None):
    """Slots at k scoring below closed-book, split into (beyond noise, within noise).
    A slot is beyond noise when the 95% interval of its per-question gap to closed-book
    sits wholly below zero. A replay holds no per-question results, so there every
    slot below closed-book counts as beyond noise, as the paper reports it."""
    dead = sorted(s for s in curves[k] if curves[k][s][0] < closedbook[0])
    if not paired:
        return dead, []
    firm, soft = [], []
    for s in dead:
        g = paired_gap(paired["slots"][(k, s)], paired["cb"])
        (firm if g and g[2] < 0 else soft).append(s)
    return firm, soft


def best_slot(cur):
    """The highest-scoring slot; ties go to the earlier slot."""
    return max(sorted(cur), key=lambda s: (cur[s][0], -s))


def interp(curve, slot):
    """Accuracy at any slot, read along straight lines between the measured slots."""
    xs = sorted(curve)
    if slot <= xs[0]:
        return curve[xs[0]][0]
    for a, b in zip(xs, xs[1:]):
        if slot <= b:
            return curve[a][0] + (curve[b][0] - curve[a][0]) * (slot - a) / (b - a)
    return curve[xs[-1]][0]


def pipeline_estimate(curves, ranks, miss, from_end):
    """Estimated accuracy of the whole pipeline at each measured k. Each question's answer
    chunk sits at its retrieval rank, counted from the end of the prompt when the best slot
    is the last one, and scores the measured curve there; a question whose answer chunk
    ranks below k, or that the retriever missed, scores `miss`, the closed-book rate."""
    est = {}
    for k in curves:
        total = 0.0
        for r in ranks:
            total += interp(curves[k], k - r if from_end else r + 1) if r is not None and r < k else miss
        est[k] = total / len(ranks)
    return est


def verdict(curves, closedbook, k_main, recall=None, paired=None, est=None):
    """The one-line decision from measured accuracies.

    curves: {k: {slot: (accuracy, n)}}; closedbook: (accuracy, n); accuracies 0 to 1.
    paired: per-question results from a sweep ({"cb": {qid: 0/1}, "slots": {(k, slot): {qid: 0/1}}}).
    est: pipeline_estimate() per k, when the eval file carries retrieval ranks.
    Rules, reproducible by hand:
      1. the best slot is the highest accuracy at k_main (ties go to the earlier slot);
         the other end slot is within noise when their gap's interval reaches zero;
      2. a slot is dead when its accuracy is below closed-book, the model with no chunks;
         a dead slot within noise is reported but does not move k;
      3. k: with retrieval ranks and two or more k measured, the k with the highest
         pipeline estimate (ties to the smaller k). Otherwise, with a dead slot beyond
         noise, the largest smaller k measured that has no dead slot, or a rerun at half k.
    """
    cur = curves[k_main]
    cb = closedbook[0]
    best = best_slot(cur)
    parts = [f"put your top chunk in slot {best} ({pct(cur[best][0])})"]
    rival = [s for s in (min(cur), max(cur)) if s != best]
    if rival:
        r = rival[0]
        if paired:
            g = paired_gap(paired["slots"][(k_main, best)], paired["slots"][(k_main, r)])
            close = g is None or g[1] <= 0
        else:
            close = wilson(*cur[r])[1] >= wilson(*cur[best])[0]
        if close:
            parts[0] += f", slot {r} is within noise at {pct(cur[r][0])}"
    k_clause = None
    if est and len(est) >= 2:
        k2 = max(sorted(est), key=lambda k: (round(est[k], 9), -k))
        against = ", ".join(f"{pct(est[k])} at k={k}" for k in sorted(est) if k != k2)
        move = (f"keep k at {k_main}" if k2 == k_main else
                f"cut k from {k_main} to {k2}" if k2 < k_main else f"raise k from {k_main} to {k2}")
        k_clause = f"{move}: with your retriever's ranks it scores an estimated {pct(est[k2])}, against {against}"
    firm, soft = split_dead(curves, closedbook, k_main, paired)
    if not firm and not soft:
        if k_clause:
            parts += [f"every slot beats closed-book ({pct(cb)})", k_clause]
        else:
            parts.append(f"every slot beats closed-book ({pct(cb)}), so keep k at {k_main}")
        return "VERDICT: " + "; ".join(parts) + "."
    if not firm:
        word, verb = ("slot", "scores") if len(soft) == 1 else ("slots", "score")
        parts.append(f"{word} {join_slots(soft)} {verb} below closed-book ({pct(cb)}), but within noise "
                     f"at n={cur[soft[0]][1]}")
        parts.append(k_clause or "rerun with a larger --limit before you cut k")
        return "VERDICT: " + "; ".join(parts) + "."
    word, verb = ("slot", "scores") if len(firm) == 1 else ("slots", "score")
    clause = f"{word} {join_slots(firm)} {verb} below closed-book ({pct(cb)})"
    if soft:
        clause += f", slot{'s' if len(soft) > 1 else ''} {join_slots(soft)} too but within noise"
    parts.append(clause)
    if k_clause:
        parts.append(k_clause)
        return "VERDICT: " + "; ".join(parts) + "."
    smaller = sorted((k for k in curves if k < k_main and min(a for a, _ in curves[k].values()) >= cb),
                     reverse=True)
    if smaller:
        k2 = smaller[0]
        worst = min(a for a, _ in curves[k2].values())
        tail = (f"your retriever finds the answer chunk in its top {k2} for "
                f"{pct(recall[k2])} of questions" if recall and k2 in recall
                else f"if your retriever still finds the answer chunk in its top {k2}")
        joiner = ", and " if recall and k2 in recall else ", "
        parts.append(f"cut k from {k_main} to {k2}, whose worst slot scores {pct(worst)}{joiner}{tail}")
    else:
        parts.append(f"cut k and rerun with --k-grid {max(2, k_main // 2)},{k_main} to find where every slot beats closed-book")
    return "VERDICT: " + "; ".join(parts) + "."


def _row(label, text):
    return f"{label:<17}: {text}"


def render(label, curves, closedbook, oracle, k_main, recall=None, missing=None, paired=None, notes=(), est=None):
    """The printed report: one line per slot per k, the baselines, the middle cost, the verdict."""
    out = [f"chunk_position_check {__version__}", f"model    : {label}"]
    cb = closedbook[0]
    for k in sorted(curves):
        firm, soft = split_dead(curves, closedbook, k, paired)
        out.append(f"k={k}{'  (your k)' if k == k_main else ''}")
        for s in sorted(curves[k]):
            a, n = curves[k][s]
            lo, hi = wilson(a, n)
            bar = "#" * int(round(a * 40))
            flag = ("  below closed-book" if s in firm else
                    "  below closed-book, within noise" if s in soft else "")
            out.append(f"  slot {s:>2}/{k:<2} {pct(a):>6}  [{pct(lo)}, {pct(hi)}]  n={n}  {bar}{flag}")
    lo, hi = wilson(*closedbook)
    out.append(f"closed-book (no chunks)          {pct(cb):>6}  [{pct(lo)}, {pct(hi)}]  n={closedbook[1]}")
    if oracle:
        lo, hi = wilson(*oracle)
        out.append(f"oracle (answer chunk alone)      {pct(oracle[0]):>6}  [{pct(lo)}, {pct(hi)}]  n={oracle[1]}")
    cur = curves[k_main]
    worst = min(cur, key=lambda s: cur[s][0])
    best = best_slot(cur)
    out.append(f"middle cost at k={k_main}: slot {worst} scores {100 * (cur[best][0] - cur[worst][0]):.1f} "
               f"points below slot {best}")
    if oracle:
        out.append(f"distractor cost at k={k_main}: the best slot scores {100 * (oracle[0] - cur[best][0]):.1f} "
                   f"points below the answer chunk alone")
    if recall:
        out.append(_row("retriever recall", ", ".join(f"top {k}: {pct(r)}" for k, r in sorted(recall.items()))))
    if est:
        out.append(_row("pipeline estimate", ", ".join(f"k={k}: {pct(e)}" for k, e in sorted(est.items()))
                        + " (answer chunk at its retrieval rank, misses at the closed-book rate)"))
    if missing:
        out.append(_row("skipped", f"{missing} question(s) whose chunks contain no expected answer"))
    out.extend(notes)
    if oracle and oracle[0] < 0.5:
        out.append(_row("warning", f"the answer chunk alone scores {pct(oracle[0])}, so most misses are not "
                                   "about position; check that each answer is a short string a correct reply contains"))
    out.append(verdict(curves, closedbook, k_main, recall, paired, est))
    return "\n".join(out)


# ------------------------------------------------------------------ replay
def load_recorded():
    with open(RECORDED, encoding="utf-8") as f:
        return json.load(f)


def cmd_replay(a):
    rec = load_recorded()
    if a.list:
        for name, m in rec["models"].items():
            print(f"{name:20s} {m['label']}  k={','.join(sorted(m['sweeps'], key=int))}")
        return 0
    if a.model not in rec["models"]:
        raise SystemExit(f"no recorded sweep for {a.model!r}; try --list")
    m = rec["models"][a.model]
    n = m["n"]
    curves = {int(k): {int(s): (v / 100, n) for s, v in sw.items()} for k, sw in m["sweeps"].items()}
    k_main = a.k if a.k else (20 if 20 in curves else max(curves))
    if k_main not in curves:
        raise SystemExit(f"{a.model} has no recorded sweep at k={k_main}; recorded: {sorted(curves)}")
    print(render(f"{m['label']}, recorded by Liu et al. 2023 ({m['cite']})", curves,
                 (m["closedbook"] / 100, n), (m["oracle"] / 100, n), k_main))
    return 0


# ------------------------------------------------------------------ sweep
def _fetch_public():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(PUBLIC_URL))
    if not os.path.exists(path):
        print(f"downloading Liu et al.'s public 20-document set to {path} (about 15 MB)")
        urllib.request.urlretrieve(PUBLIC_URL, path + ".part")
        os.replace(path + ".part", path)
    return path


def _sha(data):
    return hashlib.sha256(data).hexdigest()[:16]


def _tokens(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else f"{n:.0f}"


def cmd_sweep(a):
    path = _fetch_public() if a.eval == "public" else a.eval
    items = load_eval(path)
    ks = sorted({int(x) for x in a.k_grid.split(",")}) if a.k_grid else sorted({max(2, a.k // 2), a.k})
    if a.k not in ks:
        ks = sorted(ks + [a.k])
    slots = {k: (list(range(1, k + 1)) if a.slots == "all" else
                 auto_slots(k) if a.slots == "auto" else
                 sorted({int(x) for x in a.slots.split(",") if 1 <= int(x) <= k})) for k in ks}
    template = QA_PROMPT
    if a.prompt_template:
        with open(a.prompt_template, encoding="utf-8") as f:
            template = f.read()
        if "{search_results}" not in template or "{question}" not in template:
            raise SystemExit("--prompt-template needs {search_results} and {question} placeholders")
    with open(path, "rb") as f:
        eval_sha = _sha(f.read())
    setup = {"model": a.model, "base_url": a.base_url, "eval_sha256": eval_sha, "k_main": a.k, "ks": ks,
             "slots": {str(k): v for k, v in slots.items()}, "limit": a.limit, "seed": a.seed,
             "max_tokens": a.max_tokens, "system": a.system, "prompt_sha256": _sha(template.encode("utf-8"))}
    usable = [it for it in items if it["gold"] is not None]
    if a.limit and a.limit < len(usable):
        usable = random.Random(a.seed).sample(usable, a.limit)
    jobs, short = [], {k: 0 for k in ks}
    for it in usable:
        jobs.append((it, None, "closedbook", CLOSEDBOOK_PROMPT.format(question=it["question"])))
        jobs.append((it, None, "oracle", build_prompt(it, [it["chunks"][it["gold"]]], template)))
        for k in ks:
            if arrange(it, k, 1) is None:
                short[k] += 1
                continue
            for s in slots[k]:
                jobs.append((it, k, f"slot:{s}", build_prompt(it, arrange(it, k, s), template)))

    out = a.out
    done, header = set(), None
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "header":
                    header = r
                elif r.get("type") == "call":
                    done.add((r["qid"], r["k"], r["cond"]))
        if header is not None and header.get("setup") != setup:
            old = header.get("setup") or {}
            diff = sorted(x for x in set(setup) | set(old) if old.get(x) != setup.get(x))
            raise SystemExit(f"{out} holds a sweep with a different {', '.join(diff)}; "
                             f"pass --out with a new file name to keep both")
    todo = [j for j in jobs if (j[0]["qid"], j[1], j[2]) not in done]

    plan = "; ".join(f"k={k}: slots {', '.join(map(str, slots[k]))}" for k in ks)
    print(f"{len(usable)} questions ({plan}) plus closed-book and oracle: "
          f"{len(jobs)} calls, {len(todo)} still to make")
    skipped = sum(1 for it in items if it["gold"] is None)
    if skipped:
        print(f"{skipped} question(s) skipped: none of their chunks contains an expected answer")
    for k in ks:
        if short[k]:
            print(f"k={k}: {short[k]} question(s) have fewer than {k - 1} other chunks and sit out at this k")
    clashing = sum(1 for it in usable if it["clash"])
    if clashing:
        print(f"{clashing} question(s) hold the answer in more than one chunk; the extra copies are "
              f"left out of their prompts, so only one copy moves")
    if a.dry_run:
        print(f"about {_tokens(sum(len(j[3]) for j in todo) / 4)} input tokens still to send, "
              f"at 4 characters per token")
        first = next((j for j in jobs if j[2].startswith("slot:")), None)
        if first:
            print(f"--- the prompt for slot {first[2][5:]} at k={first[1]}, question {first[0]['qid']} ---")
            print(first[3])
        return 0

    key = os.environ.get(a.api_key_env, "")
    if not key and "api.openai.com" in a.base_url:
        raise SystemExit(f"set {a.api_key_env} before calling {a.base_url}, or pass --api-key-env")
    lock, stop, count = threading.Lock(), threading.Event(), [0]
    with open(out, "a", encoding="utf-8") as f:
        if header is None:
            f.write(json.dumps({"type": "header", "tool": __version__, "model": a.model, "base_url": a.base_url,
                                "eval": os.path.basename(path), "k_main": a.k, "setup": setup,
                                "created": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            for it in items:
                f.write(json.dumps({"type": "item", "qid": it["qid"], "gold_rank": it["gold_rank"],
                                    "has_answer_chunk": it["gold"] is not None}) + "\n")
            f.flush()

        def work(job):
            if stop.is_set():
                return
            it, k, cond, prompt = job
            rec = {"type": "call", "qid": it["qid"], "k": k, "cond": cond}
            try:
                text, served, usage = call_model(a.base_url, a.model, prompt, key, a.max_tokens, a.system)
                rec.update(response=text, correct=best_subspan_em(strip_reasoning(text), it["answers"]),
                           served_model=served, prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"))
            except OverWindow as e:
                rec.update(response=None, correct=None, error=f"over the context window: {e}")
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                count[0] += 1
                if count[0] % 100 == 0:
                    print(f"  {count[0]}/{len(todo)} calls", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
                try:
                    for _ in ex.map(work, todo):
                        pass
                except BaseException:
                    stop.set()
                    raise
        except KeyboardInterrupt:
            raise SystemExit(f"stopped after {count[0]} new calls, all saved in {out}; "
                             f"rerun the same command to resume")
    print(f"wrote {out}")
    return cmd_report(argparse.Namespace(sweep=out))


def cmd_report(a):
    header, items, calls = None, [], []
    with open(a.sweep, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "header":
                header = r
            elif r["type"] == "item":
                items.append(r)
            else:
                calls.append(r)
    if not header or not calls:
        raise SystemExit(f"{a.sweep} holds no sweep yet")
    by, over, ptoks, ctoks, served = {}, 0, {}, 0, set()
    for c in calls:
        if c.get("correct") is None:
            over += 1
            continue
        by.setdefault((c["k"], c["cond"]), {})[c["qid"]] = c["correct"]
        if c.get("prompt_tokens"):
            ptoks.setdefault(c["k"], []).append(c["prompt_tokens"])
        ctoks += c.get("completion_tokens") or 0
        if c.get("served_model"):
            served.add(c["served_model"])
    curves = {}
    for (k, cond), res in by.items():
        if cond.startswith("slot:"):
            curves.setdefault(k, {})[int(cond[5:])] = (sum(res.values()) / len(res), len(res))
    if not curves:
        raise SystemExit(f"{a.sweep} holds no scored slot calls yet"
                         + (f"; {over} calls were longer than the model's context window, so lower --k" if over else ""))
    cbr, orr = by.get((None, "closedbook"), {}), by.get((None, "oracle"), {})
    closedbook = (sum(cbr.values()) / len(cbr), len(cbr)) if cbr else (0.0, 0)
    oracle = (sum(orr.values()) / len(orr), len(orr)) if orr else None
    paired = {"cb": cbr, "slots": {(k, s): by[(k, f"slot:{s}")] for k in curves for s in curves[k]}} if cbr else None
    ranks = [it["gold_rank"] for it in items]
    recall = None
    if items and all(r is not None or not it["has_answer_chunk"] for r, it in zip(ranks, items)) \
            and any(r is not None for r in ranks):
        recall = {k: sum(1 for r in ranks if r is not None and r < k) / len(items) for k in curves}
    missing = sum(1 for it in items if not it["has_answer_chunk"])
    k_main = header["k_main"] if header["k_main"] in curves else max(curves)
    est = None
    if recall and len(curves) >= 2:
        est = pipeline_estimate(curves, ranks, closedbook[0], best_slot(curves[k_main]) > k_main / 2)
    notes = []
    if ptoks:
        total = sum(sum(v) for v in ptoks.values())
        notes.append(_row("tokens", f"{_tokens(total)} prompt, {_tokens(ctoks)} completion over {len(calls)} calls"))
        sized = {k: sum(v) / len(v) for k, v in ptoks.items() if k is not None}
        if len(sized) >= 2 and sized[max(sized)] <= 1.1 * sized[min(sized)]:
            notes.append(_row("warning", f"prompts at k={max(sized)} average {sized[max(sized)]:.0f} tokens against "
                                         f"{sized[min(sized)]:.0f} at k={min(sized)}; the endpoint may be cutting long "
                                         f"prompts, so check its context setting"))
    if over:
        notes.append(_row("left out", f"{over} call(s) longer than the model's context window"))
    if header["k_main"] != k_main:
        notes.append(_row("note", f"k={header['k_main']} has no scored calls, so the verdict reads k={k_main}"))
    label = f"{header['model']} at {header['base_url']} on {header['eval']}"
    if served and served != {header["model"]}:
        label += f", served as {', '.join(sorted(served))}"
    print(render(label, curves, closedbook, oracle, k_main, recall, missing, paired, notes, est))
    firm, _ = split_dead(curves, closedbook, k_main, paired)
    return 1 if firm else 0


# ------------------------------------------------------------------ cli
def main(argv=None):
    p = argparse.ArgumentParser(prog="chunk_position_check.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="replay a recorded sweep from Liu et al. (no key, no GPU)")
    r.add_argument("--model", default="gpt-3.5-turbo")
    r.add_argument("--k", type=int, default=0, help="your k; defaults to 20")
    r.add_argument("--list", action="store_true", help="list the recorded models")
    s = sub.add_parser("sweep", help="run the position sweep on your eval file and your model")
    s.add_argument("--eval", required=True, help="your eval file (.jsonl or .jsonl.gz), or 'public'")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", default="https://api.openai.com/v1",
                   help="any OpenAI-compatible endpoint: vLLM, Ollama (http://localhost:11434/v1), LM Studio")
    s.add_argument("--api-key-env", default="OPENAI_API_KEY", help="name of the env var holding the key")
    s.add_argument("--k", type=int, default=20, help="the top-k your pipeline uses")
    s.add_argument("--k-grid", default="", help="the k values to measure; default half your k and your k")
    s.add_argument("--slots", default="auto", help="auto (1, k/4, k/2, 3k/4, k), all, or a list")
    s.add_argument("--limit", type=int, default=200, help="questions to sample (0 = all)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=4, help="calls in flight at once")
    s.add_argument("--max-tokens", type=int, default=100,
                   help="reply budget; reasoning models need about 2000")
    s.add_argument("--system", default=None, help="optional system prompt")
    s.add_argument("--prompt-template", default=None,
                   help="your production prompt, with {search_results} and {question}")
    s.add_argument("--out", default="sweep.jsonl")
    s.add_argument("--dry-run", action="store_true", help="print the call count and one prompt, call nothing")
    rp = sub.add_parser("report", help="score a sweep file and print the verdict")
    rp.add_argument("sweep")
    a = p.parse_args(argv)
    return {"replay": cmd_replay, "sweep": cmd_sweep, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
