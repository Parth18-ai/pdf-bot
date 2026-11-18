"""
eval.py - automatic tester. Runs your test questions through the SAME pipeline
the web app uses and prints PASS / FAIL for each one.

Usage:
    python eval.py ./pdfs tests.json                   (full test, uses the LLM)
    python eval.py ./pdfs tests.json --retrieval-only  (fast: only checks that the
                                                        right PDFs/chunks are found)

tests.json is a list. Each test can have:
  question        (required) the question to ask
  context         who is asking, e.g. {"shift": "9-5", "location": "chennai"}
  expect_source   a PDF file name that MUST be among the sources used
  forbid_source   a PDF file name that must NOT be used (wrong shift!)
  expect_all      words/numbers that must ALL appear in the answer
  expect_any      at least ONE of these must appear in the answer
  forbid          words/numbers that must NOT appear in the answer
  expect_not_found  true -> the right behaviour is "I can't find it"
  expect_clarify  e.g. "shift" -> the app must ASK which shift (no context given)
"""

import asyncio
import glob
import json
import os
import sys
import uuid

from rag_core import (
    NOT_FOUND_TOKEN,
    RagError,
    answer_stream,
    build_index,
    eligible_docs,
    get_collection,
    known_options,
    retrieve,
    same,
    scope_text,
    needed_keys,
)


def canonical_context(raw: dict, options: dict) -> dict:
    """Map '9-5' to the exact value the PDFs use, e.g. '09:00-17:00'."""
    ctx = {}
    for k, v in (raw or {}).items():
        k = k.lower()
        ctx[k] = next((o for o in options.get(k, []) if same(o, v)), str(v).lower())
    return ctx


async def run_test(t, collection, docs, options, retrieval_only):
    """Returns a list of problems (empty list = PASS)."""
    problems = []
    question = t["question"]
    ctx = canonical_context(t.get("context", {}), options)
    eligible = eligible_docs(docs, ctx)
    hits = retrieve(collection, question, docs, eligible)
    asked = [k for k, _ in needed_keys(hits, eligible, docs, ctx)]

    if t.get("expect_clarify"):
        want = t["expect_clarify"].lower()
        if want not in asked:
            problems.append(f"should have asked about '{want}' (it asked: {asked or 'nothing'})")
        return problems
    if asked:
        problems.append(f"app would ask for {asked} (add it to the test's context)")
        return problems
    sources = {h["meta"]["source"] for h in hits}

    if t.get("expect_source") and t["expect_source"] not in sources:
        problems.append(f"missing source {t['expect_source']} (got {sorted(sources)})")
    if t.get("forbid_source") and t["forbid_source"] in sources:
        problems.append(f"used WRONG source {t['forbid_source']}")
    if retrieval_only:
        return problems

    if not hits:
        reply = NOT_FOUND_TOKEN
    else:
        async def ignore(_):
            pass
        reply = await answer_stream(question, hits, ctx, ignore)

    low = reply.lower()
    if t.get("expect_not_found"):
        if NOT_FOUND_TOKEN not in reply:
            problems.append("should have said NOT FOUND")
        return problems
    if NOT_FOUND_TOKEN in reply:
        problems.append("answered NOT FOUND but the info exists")
        return problems
    for w in t.get("expect_all", []):
        if w.lower() not in low:
            problems.append(f"answer is missing '{w}'")
    if t.get("expect_any") and not any(w.lower() in low for w in t["expect_any"]):
        problems.append(f"answer has none of {t['expect_any']}")
    for w in t.get("forbid", []):
        if w.lower() in low:
            problems.append(f"answer wrongly contains '{w}'")
    return problems


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    retrieval_only = "--retrieval-only" in sys.argv
    if len(args) != 2:
        print(__doc__)
        return
    folder, tests_file = args

    collection = get_collection(uuid.uuid4().hex)
    docs = {}
    paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not paths:
        print(f"No PDFs found in {folder}")
        return
    print(f"Indexing {len(paths)} PDFs ...")
    for p in paths:
        try:
            info = build_index(collection, p, os.path.basename(p), docs)
        except RagError as e:
            print(f"  ⚠️ {os.path.basename(p)}: {e}")
            continue
        docs[info["doc_id"]] = info
        print(f"  {info['name']:40s} -> {scope_text(info['scope'])}")

    options = known_options(docs)
    with open(tests_file, encoding="utf-8") as f:
        tests = json.load(f)

    passed = 0
    print(f"\nRunning {len(tests)} tests"
          + (" (retrieval only)" if retrieval_only else "") + "\n")
    for i, t in enumerate(tests, 1):
        try:
            problems = await run_test(t, collection, docs, options, retrieval_only)
        except RagError as e:
            problems = [str(e)]
        if problems:
            print(f"❌ {i}. {t['question']}")
            for p in problems:
                print(f"      - {p}")
        else:
            passed += 1
            print(f"✅ {i}. {t['question']}")

    print(f"\nScore: {passed}/{len(tests)}  ({100 * passed // max(len(tests), 1)}%)")


if __name__ == "__main__":
    asyncio.run(main())
