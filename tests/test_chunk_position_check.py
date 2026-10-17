"""Tests for chunk_position_check.py. Standard library only: python3 -m unittest discover -s tests"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import chunk_position_check as cpc  # noqa: E402


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = cpc.main(argv)
    return code, buf.getvalue()


class Metric(unittest.TestCase):
    """The metric is Liu et al.'s best_subspan_em, so replays and sweeps are comparable."""

    def test_normalization(self):
        self.assertEqual(cpc.normalize_answer("The  Eiffel Tower!"), "eiffel tower")
        self.assertEqual(cpc.normalize_answer("An apple, a day"), "apple day")

    def test_subspan_match(self):
        self.assertEqual(cpc.best_subspan_em("It was built by Gustave Eiffel in 1889.", ["gustave eiffel"]), 1.0)
        self.assertEqual(cpc.best_subspan_em("Paris", ["London", "paris"]), 1.0)
        self.assertEqual(cpc.best_subspan_em("I don't know.", ["Warsaw"]), 0.0)
        self.assertEqual(cpc.best_subspan_em("anything", [""]), 0.0)

    def test_answer_chunks_need_word_boundaries(self):
        self.assertTrue(cpc.contains_answer("num_hidden_layers is 80", ["80"]))
        self.assertFalse(cpc.contains_answer("released in 1980", ["80"]))

    def test_reasoning_is_not_scored(self):
        self.assertEqual(cpc.strip_reasoning("<think>maybe Warsaw</think> I do not know."), "I do not know.")
        self.assertEqual(cpc.strip_reasoning("<think>still thinking about Warsaw"), "")

    def test_wilson(self):
        lo, hi = cpc.wilson(0.5, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_paired_gap(self):
        a = {str(q): 0 for q in range(40)}
        b = {str(q): (1 if q % 10 == 0 else 0) for q in range(40)}
        m, lo, hi, n = cpc.paired_gap(a, b)
        self.assertEqual(n, 40)
        self.assertAlmostEqual(m, -0.1)
        self.assertAlmostEqual(lo, -0.1942, places=3)
        self.assertAlmostEqual(hi, -0.0058, places=3)


class Replay(unittest.TestCase):
    """The README's known-good output: the replay of Liu et al.'s GPT-3.5-Turbo sweep."""

    def test_gpt35_verdict(self):
        code, out = run(["replay"])
        self.assertEqual(code, 0)
        self.assertIn("slot 10/20  53.8%", out)
        self.assertIn("below closed-book", out)
        self.assertEqual(out.strip().splitlines()[-1],
                         "VERDICT: put your top chunk in slot 1 (75.8%); slots 10 and 15 score below "
                         "closed-book (56.1%); cut k from 20 to 10, whose worst slot scores 61.2%, "
                         "if your retriever still finds the answer chunk in its top 10.")

    def test_readme_block_is_the_live_output(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        block = re.search(r"## Check it before you trust it.*?```\n(.*?)```", readme, re.S).group(1)
        self.assertEqual(block, run(["replay"])[1])

    def test_verdicts_differ_by_model(self):
        """Hand-checked against Tables 1 and 6 of the paper: the best slot is not always slot 1."""
        expect = {"claude-1.3": "slot 20", "mpt-30b-instruct": "slot 20",
                  "longchat-13b-16k": "slot 1 ", "llama-2-70b-chat": "slot 20",
                  "gpt-3.5-turbo-16k": "slot 1 "}
        for model, slot in expect.items():
            _, out = run(["replay", "--model", model])
            last = out.strip().splitlines()[-1]
            self.assertTrue(last.startswith(f"VERDICT: put your top chunk in {slot}"), (model, last))

    def test_every_recorded_number_is_a_percentage(self):
        rec = cpc.load_recorded()
        for m in rec["models"].values():
            for sweep in m["sweeps"].values():
                for v in sweep.values():
                    self.assertTrue(0 < v < 100)


class Prompts(unittest.TestCase):
    ITEM = {"qid": "q", "question": "who wrote it?", "answers": ["Ada"],
            "chunks": ["c0", "c1 by Ada", "c2", "c3", "c4"], "gold": 1, "gold_rank": 1, "clash": []}

    def test_arrange_moves_only_the_answer_chunk(self):
        self.assertEqual(cpc.arrange(self.ITEM, 4, 1), ["c1 by Ada", "c0", "c2", "c3"])
        self.assertEqual(cpc.arrange(self.ITEM, 4, 4), ["c0", "c2", "c3", "c1 by Ada"])
        self.assertIsNone(cpc.arrange(self.ITEM, 6, 1))   # only 4 other chunks

    def test_a_second_copy_of_the_answer_never_stays_behind(self):
        item = dict(self.ITEM, chunks=["Ada again", "c1 by Ada", "c2", "c3", "c4"], clash=[0])
        self.assertEqual(cpc.arrange(item, 4, 4), ["c2", "c3", "c4", "c1 by Ada"])

    def test_prompt_matches_liu_template(self):
        p = cpc.build_prompt(self.ITEM, [{"title": "T", "text": "body"}])
        self.assertEqual(p, "Write a high-quality answer for the given question using only the provided "
                            "search results (some of which might be irrelevant).\n\n"
                            "Document [1](Title: T) body\n\nQuestion: who wrote it?\nAnswer:")

    def test_interp_and_pipeline_estimate(self):
        curve = {1: (1.0, 40), 5: (0.0, 40), 10: (1.0, 40)}
        self.assertAlmostEqual(cpc.interp(curve, 3), 0.5)
        self.assertAlmostEqual(cpc.interp(curve, 9), 0.8)
        est = cpc.pipeline_estimate({10: curve}, [0, 4, 9, None], 0.25, from_end=False)
        self.assertAlmostEqual(est[10], (1.0 + 0.0 + 1.0 + 0.25) / 4)
        est = cpc.pipeline_estimate({10: curve}, [0, 4, 9, None], 0.25, from_end=True)
        self.assertAlmostEqual(est[10], (1.0 + cpc.interp(curve, 6) + 1.0 + 0.25) / 4)

    def test_auto_slots(self):
        self.assertEqual(cpc.auto_slots(20), [1, 5, 10, 15, 20])
        self.assertEqual(cpc.auto_slots(3), [1, 2, 3])


class EvalFormats(unittest.TestCase):
    def write(self, d, rows):
        p = os.path.join(d, "e.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
        return p

    def test_simple_and_liu_formats(self):
        with tempfile.TemporaryDirectory() as d:
            items = cpc.load_eval(self.write(d, [
                {"question": "q1", "answers": ["x"], "chunks": ["a", "has x", "b", "x again"]},
                {"question": "q2", "answers": ["y"], "chunks": ["a", "b"], "gold": 0},
                {"question": "q3", "answers": ["z"], "chunks": ["a", "b"]},
                {"question": "q4", "answers": ["w"], "ctxs": [
                    {"title": "t", "text": "w here", "isgold": True, "hasanswer": True},
                    {"title": "u", "text": "no", "hasanswer": False}]}]))
            self.assertEqual([it["gold"] for it in items], [1, 0, None, 0])
            self.assertEqual([it["gold_rank"] for it in items], [1, 0, None, None])
            self.assertEqual(items[0]["clash"], [3])

    def test_bad_lines_name_the_line(self):
        with tempfile.TemporaryDirectory() as d:
            for row, msg in [({"question": "q", "answers": ["x"]}, "line 1: needs 'chunks'"),
                             ({"question": "q", "chunks": ["a"]}, "line 1: needs a 'question'"),
                             ({"question": "q", "answers": ["x"], "chunks": ["a"], "gold": 3}, "gold 3"),
                             ("{not json", "line 1: not valid JSON")]:
                with self.assertRaises(SystemExit) as e:
                    cpc.load_eval(self.write(d, [row]))
                self.assertIn(msg, str(e.exception))

    def test_shipped_example_file(self):
        items = cpc.load_eval(os.path.join(ROOT, "examples", "example_eval.jsonl"))
        self.assertEqual([it["gold"] for it in items], [2, 3, None])   # ex2 found by its answer, ex3 a miss


# ------------------------------------------------------------------ live sweeps against a mock model
class Mock(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that behaves like a lost-in-the-middle model.

    u:      answers only when the answer chunk sits in the first two or last two slots,
            or the prompt holds five chunks or fewer; closed-book it knows questions 0, 10, 20, 30.
    noise:  as u, but the middle slots also answer questions 0, 12 and 24, so they land a
            hair under closed-book, a gap 40 questions cannot separate from noise.
    strict: as u, behind the parameter rules of OpenAI's reasoning models, and it thinks
            out loud about the right answer before saying it does not know.
    window: as u, but refuses any prompt holding more than five chunks as too long.
    """
    mode, calls, lock = "u", 0, threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if Mock.mode == "strict" and "max_tokens" in body:
            return self.send(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported "
                                                        "with this model. Use 'max_completion_tokens' instead."}})
        if Mock.mode == "strict" and "temperature" in body:
            return self.send(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0 "
                                                        "with this model. Only the default (1) value is supported."}})
        prompt = body["messages"][-1]["content"]
        docs = re.findall(r"^Document \[(\d+)\] (.*)$", prompt, re.M)
        if Mock.mode == "window" and len(docs) > 5:
            return self.send(400, {"error": {"message": "This model's maximum context length is 4096 tokens."}})
        qn = int(re.search(r"Question: q(\d+)", prompt).group(1))
        answer, miss = f"secret{qn}", "I do not know."
        if not docs:
            text = answer if qn % 10 == 0 else miss
        else:
            k = len(docs)
            slot = next(int(i) for i, t in docs if answer in t)
            edge = k <= 5 or slot <= 2 or slot >= k - 1
            text = answer if edge or (Mock.mode == "noise" and qn in (0, 12, 24)) else miss
        if Mock.mode == "strict" and text == miss:
            text = f"<think>could it be {answer}?</think> {miss}"
        with Mock.lock:
            Mock.calls += 1
        self.send(200, {"model": "mock-2026-09-28", "choices": [{"message": {"content": text}}],
                        "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 4}})

    def send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class LiveSweep(unittest.TestCase):
    # By hand: at k=5 every slot scores 100%, and 19 of 40 answer chunks rank in the top 5, so
    # (19 + 21 x 0.10) / 40 = 52.75%. At k=10, ranks 0 to 9 sit at slots 1 to 10 and score
    # 1, 1, 2/3, 1/3, 0, 0, 0, 0, 1/2, 1 along the measured curve: (16.5 + 6 x 0.10) / 40 = 42.75%.
    U_VERDICT = ("VERDICT: put your top chunk in slot 1 (100.0%), slot 10 is within noise at 100.0%; "
                 "slots 5 and 8 score below closed-book (10.0%); cut k from 10 to 5: with your retriever's "
                 "ranks it scores an estimated 52.8%, against 42.7% at k=10.")

    def setUp(self):
        Mock.mode, Mock.calls = "u", 0
        cpc._COMPAT.update(max_tokens_key="max_tokens", temperature=True)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        self.dir = tempfile.TemporaryDirectory()
        self.eval = os.path.join(self.dir.name, "eval.jsonl")
        with open(self.eval, "w") as f:
            for n in range(40):
                chunks = [f"filler {n}-{j}" for j in range(12)]
                gold = n % 12                           # retrieval rank of the answer chunk
                chunks[gold] = f"the answer is secret{n}"
                f.write(json.dumps({"id": f"q{n}", "question": f"q{n}?", "answers": [f"secret{n}"],
                                    "chunks": chunks, "gold": gold}) + "\n")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.dir.cleanup()
        cpc._COMPAT.update(max_tokens_key="max_tokens", temperature=True)

    def sweep(self, *extra, out="sweep.jsonl", grid="5,10"):
        args = ["sweep", "--eval", self.eval, "--model", "mock", "--base-url", self.url, "--k", "10",
                "--limit", "0", "--out", os.path.join(self.dir.name, out)] + (["--k-grid", grid] if grid else [])
        return run(args + list(extra))

    def test_sweep_report_and_exit_code(self):
        code, text = self.sweep()
        self.assertEqual(code, 1)                              # slots below closed-book beyond noise
        self.assertEqual(Mock.calls, 40 * 12)                  # 40 x (5 slots at k=5, 5 at k=10, 2 baselines)
        self.assertEqual(text.strip().splitlines()[-1], self.U_VERDICT)
        self.assertIn("retriever recall : top 5: 47.5%, top 10: 85.0%", text)
        self.assertIn("pipeline estimate: k=5: 52.8%, k=10: 42.7%", text)
        self.assertIn("served as mock-2026-09-28", text)
        self.assertIn("tokens           : ", text)
        code, text = run(["report", os.path.join(self.dir.name, "sweep.jsonl")])
        self.assertEqual((code, text.strip().splitlines()[-1]), (1, self.U_VERDICT))

    def test_resume_makes_only_the_missing_calls(self):
        self.sweep()
        path = os.path.join(self.dir.name, "sweep.jsonl")
        with open(path) as f:
            lines = f.read().splitlines(True)
        with open(path, "w") as f:
            f.writelines(lines[:-50])                          # as if the run died 50 calls early
        Mock.calls = 0
        code, text = self.sweep()
        self.assertEqual(Mock.calls, 50)
        self.assertEqual(text.strip().splitlines()[-1], self.U_VERDICT)

    def test_a_changed_setup_refuses_to_mix(self):
        self.sweep()
        with self.assertRaises(SystemExit) as e:
            self.sweep("--model", "another")
        self.assertIn("different model", str(e.exception))

    def test_a_gap_inside_noise_does_not_move_k(self):
        Mock.mode = "noise"
        code, text = self.sweep(grid="10")
        self.assertEqual(code, 0)
        self.assertIn("below closed-book, within noise", text)
        self.assertEqual(text.strip().splitlines()[-1],
                         "VERDICT: put your top chunk in slot 1 (100.0%), slot 10 is within noise at 100.0%; "
                         "slots 5 and 8 score below closed-book (10.0%), but within noise at n=40; "
                         "rerun with a larger --limit before you cut k.")

    def test_reasoning_model_parameters_and_thinking(self):
        Mock.mode = "strict"
        code, text = self.sweep("--workers", "1")
        self.assertEqual(cpc._COMPAT, {"max_tokens_key": "max_completion_tokens", "temperature": False})
        self.assertEqual(text.strip().splitlines()[-1], self.U_VERDICT)

    def test_prompts_over_the_window_are_left_out(self):
        Mock.mode = "window"
        code, text = self.sweep()
        self.assertIn("left out         : 200 call(s) longer than the model's context window", text)
        self.assertIn("k=10 has no scored calls, so the verdict reads k=5", text)

    def test_dry_run_makes_no_calls(self):
        code, text = self.sweep("--dry-run", grid="10")
        self.assertEqual((code, Mock.calls), (0, 0))
        self.assertIn("Document [1] the answer is secret", text)     # slot 1 holds the answer chunk
        self.assertIn("Document [10] filler", text)
        self.assertIn("input tokens still to send", text)


if __name__ == "__main__":
    unittest.main()
