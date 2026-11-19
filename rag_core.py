"""
rag_core.py - the "brain" (multi-document, any-attribute, hybrid-search version)
"""

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from ollama import AsyncClient, Client

# ----------------------------- settings ---------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
NOT_FOUND_TOKEN = "NOT_FOUND"
CHUNK_WORDS = 180
OVERLAP_WORDS = 40
VEC_K = 20
BM25_K = 20
FINAL_K = 8
MAX_DISTANCE = 1.25
SAMPLE_WORDS = 500
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")
DB_PATH = os.getenv("DB_PATH", "rag_store")
CACHE_DIR = os.getenv("CACHE_DIR", "index_cache")
CACHE_VERSION = 5
IGNORE_VALUES = {
    "all", "everyone", "any", "none", "n/a", "na", "general",
    "null", "unknown", "both", "all employees", ""
}
TOPIC_NOISE_VALUES = {
    "bonus", "bonuses", "leave", "leaves", "benefit", "benefits", "benifits",
    "policy", "policies", "allowance", "allowances", "salary", "pay",
    "overtime", "travel", "remote", "training", "stipend", "insurance"
}
STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been do does did "
    "i me my we our you your it its this that these those what which who how "
    "when where why can could should would will shall at by from as if than "
    "then there their about get got have has had any".split()
)


class RagError(Exception):
    """A friendly error we can show to the user."""


_client = chromadb.PersistentClient(path=DB_PATH)
_COLLECTION = "company_docs"


def get_collection(session_id: str = ""):
    return _client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def sample_embeddings(collection, doc_ids: list, limit: int = 3) -> list:
    if not doc_ids:
        return []
    try:
        res = collection.get(
            where={"doc_id": {"$in": doc_ids}},
            limit=limit,
            include=["embeddings", "documents", "metadatas"],
        )
    except Exception as e:
        raise RagError(f"Could not read embeddings: {e}")

    out = []
    embeddings = res.get("embeddings")
    if embeddings is None:
        embeddings = []
    for cid, emb, meta, doc in zip(
        res.get("ids", []),
        embeddings,
        res.get("metadatas", []),
        res.get("documents", []),
    ):
        vec = list(emb) if emb is not None else []
        out.append({
            "id": cid,
            "dims": len(vec),
            "preview": [round(float(x), 4) for x in vec[:8]],
            "meta": meta or {},
            "text": (doc or "")[:90].replace("\n", " "),
        })
    return out


# --------------------- comparing values ("9-5" == "9 to 5") ---------------
TIME_RE = re.compile(
    r"(?<![\d:.])(\d{1,2})(?::(\d{2}))?\s*(?:(a|p)\.?m\.?)?"
    r"\s*(?:-|–|—|\bto\b|\btill\b|\buntil\b)\s*"
    r"(\d{1,2})(?::(\d{2}))?\s*(?:(a|p)\.?m\.?)?(?![\d:]|[-/]\d)",
    re.IGNORECASE,
)


def _minutes(h, m, ap):
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h * 60 + m


def _time_key(match):
    h1, m1, a1, h2, m2, a2 = match.groups()
    h1, h2 = int(h1), int(h2)
    m1, m2 = int(m1 or 0), int(m2 or 0)
    a1 = a1.lower() if a1 else None
    a2 = a2.lower() if a2 else None
    if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
        return None
    is_24h = h1 > 12 or h2 > 12 or h1 == 0
    if a1 and a2:
        s, e = _minutes(h1, m1, a1), _minutes(h2, m2, a2)
    elif a2:
        e = _minutes(h2, m2, a2)
        s = _minutes(h1, m1, a2)
        if s >= e:
            s = _minutes(h1, m1, "a" if a2 == "p" else "p")
    else:
        s = _minutes(h1, m1, a1)
        e = _minutes(h2, m2, None)
        if e <= s and not is_24h:
            e += 12 * 60
    e %= 24 * 60
    return f"{s // 60:02d}:{s % 60:02d}-{e // 60:02d}:{e % 60:02d}"


def norm(value) -> str:
    v = re.sub(r"\s+", " ", str(value).lower().strip())
    v_compact = re.sub(r"\s*-\s*", "-", v)
    m = TIME_RE.fullmatch(v) or TIME_RE.fullmatch(v_compact)
    if m:
        return _time_key(m) or v_compact
    return v


def same(a, b) -> bool:
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    if re.fullmatch(r"[a-z ]{3,}", na) and re.fullmatch(r"[a-z ]{3,}", nb):
        return na in nb or nb in na
    return False


def _llm_json(system: str, user: str) -> dict:
    try:
        r = Client().chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format="json",
            options={"temperature": 0, "num_ctx": NUM_CTX},
            keep_alive=KEEP_ALIVE,
        )
    except Exception as e:
        raise RagError(f"Could not talk to the model ({e}). Is Ollama running?")
    try:
        data = json.loads(r["message"]["content"])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


KEY_ALIASES = {
    "office": "location", "city": "location", "site": "location", "branch": "location",
    "place": "location", "work_location": "location", "office_location": "location",
    "shifts": "shift", "shift_timing": "shift", "shift_time": "shift", "timing": "shift",
    "timings": "shift", "working_hours": "shift", "hours": "shift", "work_hours": "shift",
    "dept": "department", "team": "department", "designation": "role", "job_role": "role",
}


def clean_scope(raw: dict) -> dict:
    out = {}
    for k, v in (raw or {}).items():
        if not isinstance(v, (str, int, float)):
            continue
        key = re.sub(r"[^a-z]+", "_", str(k).lower()).strip("_")
        val = norm(v)
        key = KEY_ALIASES.get(key, key)
        if key and val and val not in IGNORE_VALUES:
            out[key] = val
    return dict(list(out.items())[:6])


def _sample_text(text: str) -> str:
    words = text.split()
    if len(words) <= SAMPLE_WORDS:
        return " ".join(words)
    part = SAMPLE_WORDS // 3
    mid = len(words) // 2
    return " ".join(words[:part] + ["..."] + words[mid:mid + part]
                    + ["..."] + words[-part:])


def extract_doc_scope(name: str, text: str) -> dict:
    filename_scope = {}
    lower_name = name.lower()
    for loc in [
        "chennai", "mumbai", "bangalore", "bengaluru", "delhi",
        "hyderabad", "pune", "kolkata", "remote", "onsite"
    ]:
        if re.search(rf"(?<![a-z]){re.escape(loc)}(?![a-z])", lower_name):
            filename_scope["location"] = loc
    for shift_keyword in [
        "9-5", "9_5", "9to5", "night", "morning", "evening",
        "rotational", "general", "09:00-17:00"
    ]:
        if shift_keyword in lower_name:
            filename_scope["shift"] = (
                "09:00-17:00" if shift_keyword in ("9-5", "9_5", "9to5") else shift_keyword
            )

    system = (
        "You read parts of a document and decide which specific group it is "
        "written for. Return ONLY JSON.\n"
        "Use short lowercase keys such as shift, location, department, role, "
        "grade, plan, region, year, client.\n"
        "Include a key ONLY if the whole document is clearly limited to ONE value "
        "of it (example: only for the 9-5 shift, only for the Chennai office).\n"
        "If the document applies to everyone, or covers several values, return {}.\n"
        "Write time ranges like 9-5. Never guess."
    )
    user = f"File name: {name}\n\nText:\n{_sample_text(text)}"
    llm_scope = clean_scope(_llm_json(system, user))

    filename_scope.update(llm_scope)
    return clean_scope(filename_scope)


def known_options(docs: dict) -> dict:
    opts = {}
    for d in docs.values():
        for k, v in d["scope"].items():
            if not any(same(v, x) for x in opts.setdefault(k, [])):
                opts[k].append(v)
    return {k: sorted(v) for k, v in opts.items()}


def _is_valid_user_value(key: str, val: str, known_vals: list) -> bool:
    if not val or val in IGNORE_VALUES:
        return False
    if any(same(val, x) for x in known_vals):
        return True
    if val in TOPIC_NOISE_VALUES:
        return False
    return True


def extract_context(question: str, options: dict) -> dict:
    if not options:
        return {}
    ctx = {}

    system = (
        "The user asks questions about documents. The documents come in versions "
        "for different groups.\n"
        f"Known groups: {json.dumps(options)}\n"
        "Extract ONLY attributes the user explicitly states about THEMSELVES "
        "(such as their shift or location). Do NOT extract topic words from the "
        "question like 'bonus', 'leave', 'benefits', 'travel', or 'annual' as a role/grade/plan/department. "
        "Use the exact value from the known list when it matches. "
        'Return ONLY JSON like {"shift": "9-5"}, or {} if nothing is stated.'
    )
    for k, v in _llm_json(system, question).items():
        k = str(k).lower().strip()
        if k in options and isinstance(v, (str, int, float)):
            raw_v = str(v).strip().lower()
            matched_opt = next((o for o in options[k] if same(o, raw_v)), None)
            if matched_opt:
                ctx[k] = matched_opt
            elif _is_valid_user_value(k, raw_v, options[k]):
                ctx[k] = norm(raw_v)

    q = norm(question)
    for k, vals in options.items():
        for v in vals:
            if re.search(rf"(?<!\w){re.escape(norm(v))}(?!\w)", q):
                ctx[k] = v
        for m in TIME_RE.finditer(question):
            key = _time_key(m)
            for v in vals:
                if key and norm(v) == key:
                    ctx[k] = v
    return ctx


def eligible_docs(docs: dict, context: dict) -> list:
    out = []
    for d in docs.values():
        ok = all(
            k not in d["scope"] or same(d["scope"][k], v)
            for k, v in context.items()
        )
        if ok:
            out.append(d)
    return out


def unknown_values(options: dict, context: dict) -> list:
    bad = []
    for k, v in context.items():
        if k in options and not any(same(x, v) for x in options[k]):
            bad.append((k, v))
    return bad


def ambiguous_key(eligible: list, context: dict):
    keys = {k for d in eligible for k in d["scope"]}
    for k in sorted(keys):
        if k in context:
            continue
        values = []
        for d in eligible:
            if k in d["scope"] and not any(same(d["scope"][k], x) for x in values):
                values.append(d["scope"][k])
        if len(values) > 1:
            return k, sorted(values)
    return None


def needed_keys(hits: list, eligible: list, docs: dict, context: dict) -> list:
    if not hits:
        return []
    best = docs[hits[0]["meta"]["doc_id"]]
    best_keys = set(best["scope"])
    needed = []
    for k in sorted(best_keys):
        if k in context:
            continue
        values = []
        for d in eligible:
            dk = set(d["scope"])
            same_kind = dk <= best_keys or best_keys <= dk
            if k in d["scope"] and same_kind and not any(same(d["scope"][k], x) for x in values):
                values.append(d["scope"][k])
        if len(values) > 1:
            needed.append((k, sorted(values)))
    return needed


def top_docs(hits: list, docs: dict, n: int = 3) -> list:
    out = []
    for h in hits[:n]:
        d = docs[h["meta"]["doc_id"]]
        if d not in out:
            out.append(d)
    return out


def scope_text(scope: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in scope.items()) or "everyone"


def _table_to_lines(table) -> list:
    rows = table.extract()
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if len(rows) < 2:
        return [" | ".join(r) for r in rows]
    header = rows[0]
    lines = []
    for row in rows[1:]:
        cells = []
        for i, cell in enumerate(row):
            if not cell:
                continue
            h = header[i] if i < len(header) and header[i] else ""
            cells.append(f"{h}: {cell}" if h else cell)
        if cells:
            lines.append("; ".join(cells))
    return lines


def page_text(page) -> str:
    try:
        tables = page.find_tables().tables
    except Exception:
        tables = []
    if not tables:
        return page.get_text(sort=True)

    rects = [fitz.Rect(t.bbox) for t in tables]
    parts = []
    for b in page.get_text("blocks", sort=True):
        if b[6] != 0:
            continue
        centre = fitz.Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if not any(r.contains(centre) for r in rects):
            parts.append(b[4].strip())
    for t in tables:
        parts.extend(_table_to_lines(t))
    return "\n".join(p for p in parts if p)


def chunk_text(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunk, count = [], 0
    for line in lines:
        w = len(line.split())
        if chunk and count + w > CHUNK_WORDS:
            yield "\n".join(chunk)
            keep, c = [], 0
            for old in reversed(chunk):
                ow = len(old.split())
                if c + ow > OVERLAP_WORDS:
                    break
                keep.insert(0, old)
                c += ow
            chunk, count = keep, c
        chunk.append(line)
        count += w
    if chunk:
        yield "\n".join(chunk)


def _cache_path(doc_id: str) -> Path:
    return Path(CACHE_DIR) / f"{doc_id}.json"


def save_doc_cache(info: dict) -> None:
    try:
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in info.items() if k != "cached"}
        data["v"] = CACHE_VERSION
        _cache_path(info["doc_id"]).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _load_doc_cache(doc_id: str):
    try:
        data = json.loads(_cache_path(doc_id).read_text(encoding="utf-8"))
        return data if data.get("v") == CACHE_VERSION else None
    except Exception:
        return None


def build_index(collection, path: str, name: str, existing: dict) -> dict:
    try:
        with open(path, "rb") as f:
            doc_id = hashlib.sha1(f.read()).hexdigest()[:16]
    except Exception as e:
        raise RagError(f"Could not open {name}: {e}")

    if doc_id in existing:
        return {**existing[doc_id], "cached": True}

    saved = _load_doc_cache(doc_id)
    if saved and saved.get("records"):
        recs = list(saved["records"].values())
        try:
            present = collection.get(ids=[recs[0]["id"]])["ids"]
            if not present:
                collection.add(
                    ids=[r["id"] for r in recs],
                    documents=[r["text"] for r in recs],
                    metadatas=[r["meta"] for r in recs],
                )
        except Exception as e:
            raise RagError(f"Could not load saved index for {name}: {e}")
        saved.pop("v", None)
        saved["scope"] = clean_scope(saved.get("scope", {}))
        saved["name"] = name
        saved["cached"] = True
        return saved

    try:
        pdf = fitz.open(path)
    except Exception as e:
        raise RagError(f"Could not open {name}: {e}")

    pages = [(i + 1, page_text(p)) for i, p in enumerate(pdf)]
    full_text = "\n".join(t for _, t in pages)
    if not full_text.strip():
        raise RagError(f"{name} has no readable text (maybe a scanned PDF).")

    scope = extract_doc_scope(name, full_text)

    ids, texts, metas, records = [], [], [], {}
    for page_no, text in pages:
        for n, piece in enumerate(chunk_text(text)):
            cid = f"{doc_id}-{page_no}-{n}"
            meta = {"doc_id": doc_id, "source": name, "page": page_no}
            ids.append(cid)
            texts.append(piece)
            metas.append(meta)
            records[cid] = {"id": cid, "text": piece, "meta": meta}
    if not ids:
        raise RagError(f"{name} has no readable text.")
    collection.add(ids=ids, documents=texts, metadatas=metas)
    info = {"doc_id": doc_id, "name": name, "scope": scope,
            "chunks": len(ids), "records": records, "cached": False}
    save_doc_cache(info)
    return info


def tokens(text: str) -> list:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def bm25_rank(query: str, records: list, top_n: int) -> list:
    q_terms = set(tokens(query))
    if not q_terms or not records:
        return []
    docs_tokens = [tokens(r["text"]) for r in records]
    n = len(records)
    avg_len = sum(len(t) for t in docs_tokens) / n or 1
    df = Counter()
    for toks in docs_tokens:
        for term in q_terms & set(toks):
            df[term] += 1

    k1, b = 1.5, 0.75
    scored = []
    for rec, toks in zip(records, docs_tokens):
        tf = Counter(toks)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            scored.append((score, rec["id"]))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:top_n]]


def rrf_merge(rankings: list, k: int = 60) -> list:
    scores = Counter()
    for ranking in rankings:
        for rank, rid in enumerate(ranking):
            scores[rid] += 1.0 / (k + rank + 1)
    return [rid for rid, _ in scores.most_common()]


def _split_subquestions(question: str) -> list:
    parts = [p.strip() for p in re.split(r"\b(?:and also|also|and)\b|[?;]", question, flags=re.IGNORECASE) if p.strip()]
    return parts if len(parts) > 1 else [question]


def _retrieve_single(collection, sub_q: str, docs: dict, records: dict, where: dict) -> list:
    try:
        res = collection.query(
            query_texts=[sub_q],
            n_results=min(VEC_K, len(records)),
            where=where,
        )
    except Exception as e:
        raise RagError(f"Search failed: {e}")
    vec_ids, distance = [], {}
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        if dist <= MAX_DISTANCE and cid in records:
            vec_ids.append(cid)
            distance[cid] = dist

    kw_ids = bm25_rank(sub_q, list(records.values()), BM25_K)
    best = rrf_merge([vec_ids, kw_ids])[:FINAL_K]

    hits = []
    for cid in best:
        rec = records[cid]
        doc = docs[rec["meta"]["doc_id"]]
        hits.append({
            "text": rec["text"],
            "meta": {**rec["meta"], "source": doc["name"]},
            "distance": distance.get(cid),
            "scope": docs[rec["meta"]["doc_id"]]["scope"],
        })
    return hits


def retrieve(collection, question: str, docs: dict, eligible: list) -> list:
    if not eligible:
        return []

    records = {}
    for d in eligible:
        records.update(d["records"])

    ids = [d["doc_id"] for d in eligible]
    where = {"doc_id": {"$in": ids}}

    sub_qs = _split_subquestions(question)
    if len(sub_qs) == 1:
        return _retrieve_single(collection, question, docs, records, where)

    # Multi-part question: merge top hits across sub-questions so each part gets its source PDF
    merged = []
    seen_texts = set()
    for sq in sub_qs:
        for h in _retrieve_single(collection, sq, docs, records, where)[:4]:
            key = (h["meta"]["doc_id"], h["meta"]["page"], h["text"][:80])
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(h)
    return merged[:FINAL_K]


def build_prompt(question: str, hits: list, context: dict):
    passages = "\n\n".join(
        f"[{i + 1}] (file: {h['meta']['source']}, page {h['meta']['page']}, "
        f"written for: {scope_text(h['scope'])})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    if context:
        rule = (
            f"The user's situation: {scope_text(context)} "
            f"(note: 09:00-17:00 is the same as 9-5). "
            "Use facts from passages that match this situation or apply to everyone/general. "
            "Only ignore a passage if it explicitly states a conflicting shift or location."
        )
    else:
        rule = "No special situation was given, so answer from the passages as they are."

    system = (
        "You answer questions using ONLY the numbered passages given.\n"
        f"{rule}\n"
        "Be complete: if the passages list several items (allowances, insurance, leave, "
        "perks, bonus, travel, steps, amounts), include ALL of them. Copy numbers, names and dates exactly.\n"
        "Put the passage number like [1] after each fact you use.\n"
        f"If the passages truly have zero relevant information, reply with exactly: {NOT_FOUND_TOKEN}"
    )
    user = f"Passages:\n{passages}\n\nQuestion: {question}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


async def answer_stream(question: str, hits: list, context: dict, on_token) -> str:
    full, started, not_found = "", False, False
    try:
        stream = await AsyncClient().chat(
            model=LLM_MODEL,
            messages=build_prompt(question, hits, context),
            stream=True,
            options={"temperature": 0, "num_ctx": NUM_CTX},
            keep_alive=KEEP_ALIVE,
        )
        async for part in stream:
            tok = part["message"]["content"]
            full += tok
            if started:
                await on_token(tok)
            elif not not_found and len(full.strip()) >= len(NOT_FOUND_TOKEN):
                if full.strip().startswith(NOT_FOUND_TOKEN):
                    not_found = True
                else:
                    started = True
                    await on_token(full)
    except Exception as e:
        raise RagError(f"Could not talk to the model ({e}). Is Ollama running?")

    if not started and not not_found and NOT_FOUND_TOKEN not in full:
        await on_token(full)
    return full


def cited_sources(reply: str, hits: list) -> list:
    """Return all hits cited via [n], plus any other hit whose key terms appear in the reply."""
    used = []
    for num in re.findall(r"\[(\d+)\]", reply):
        i = int(num) - 1
        if 0 <= i < len(hits) and hits[i] not in used:
            used.append(hits[i])

    reply_words = set(re.findall(r"[a-z0-9]{3,}", reply.lower()))
    for h in hits:
        if h in used:
            continue
        passage_words = {
            w for w in re.findall(r"[a-z0-9]{3,}", h["text"].lower())
            if w not in STOPWORDS and w not in {"policy", "employees", "company", "written", "general"}
        }
        # If multiple significant words from a retrieved passage appear in the reply, include its source
        if len(passage_words & reply_words) >= 3:
            used.append(h)

    return used or hits[:1]


def _warm_up():
    try:
        Client().chat(model=LLM_MODEL, messages=[{"role": "user", "content": "hi"}],
                      options={"num_ctx": NUM_CTX, "num_predict": 1},
                      keep_alive=KEEP_ALIVE)
    except Exception:
        pass
    try:
        from chromadb.utils import embedding_functions
        embedding_functions.DefaultEmbeddingFunction()(["warm up"])
    except Exception:
        pass


threading.Thread(target=_warm_up, daemon=True).start()
