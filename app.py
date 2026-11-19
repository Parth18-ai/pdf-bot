"""
app.py - the Chainlit web interface (multi-PDF, scope-aware version)
"""

import re
import uuid

import chainlit as cl

from rag_core import (
    NOT_FOUND_TOKEN,
    RagError,
    ambiguous_key,
    answer_stream,
    build_index,
    cited_sources,
    clean_scope,
    eligible_docs,
    extract_context,
    get_collection,
    known_options,
    norm,
    retrieve,
    same,
    save_doc_cache,
    scope_text,
    top_docs,
    unknown_values,
)

WELCOME = (
    "## 📄 Chat with your PDFs\n"
    "Attach **many PDFs at once** (10 or more is fine).\n\n"
    "Your PDFs can differ by **anything**: shift, location, department, role, "
    "plan, year... I work out what each PDF is *for*. Then tell me about "
    "yourself in your question, like:\n"
    "*\"I work the 9-5 shift in Chennai, what benefits do I get?\"*\n\n"
    "and I will answer **only from the PDFs that match you**.\n\n"
    "Commands: `/docs` · `/scope` · `/reset`"
)

COMMON_LOCATIONS = {
    "pune", "chennai", "mumbai", "bangalore", "bengaluru",
    "delhi", "hyderabad", "kolkata", "noida", "gurgaon", "remote", "onsite"
}


def is_pdf(element) -> bool:
    name = (getattr(element, "name", "") or "").lower()
    mime = (getattr(element, "mime", "") or "").lower()
    return mime == "application/pdf" or name.endswith(".pdf")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("collection", get_collection(uuid.uuid4().hex))
    cl.user_session.set("docs", {})
    cl.user_session.set("context", {})
    cl.user_session.set("pending", None)
    cl.user_session.set("pending_key", None)
    await cl.Message(content=WELCOME).send()


def docs_table(docs: dict) -> str:
    rows = [f"| {d['name']} | {scope_text(d['scope'])} | {d['chunks']} |"
            for d in docs.values()]
    return "| File | Written for | Chunks |\n|---|---|---|\n" + "\n".join(rows)


async def load_pdfs(pdfs) -> None:
    collection = cl.user_session.get("collection")
    docs = cl.user_session.get("docs")
    errors = []
    total = len(pdfs)

    progress = cl.Message(content=f"⏳ Reading your PDFs (0 of {total})...")
    await progress.send()

    for n, element in enumerate(pdfs, 1):
        progress.content = f"⏳ Reading PDF {n} of {total}: **{element.name}**"
        await progress.update()
        try:
            info = await cl.make_async(build_index)(
                collection, element.path, element.name, docs
            )
        except RagError as e:
            errors.append(f"⚠️ {element.name}: {e}")
            continue
        docs[info["doc_id"]] = info

    await progress.remove()

    text = "✅ **PDFs processed**\n\n" + docs_table(docs)
    text += ("\n\nIf a **Written for** value looks wrong, fix it with "
             "`/scope file.pdf key=value`.")
    if errors:
        text += "\n\n" + "\n".join(errors)
    await cl.Message(content=text).send()


async def handle_command(text: str) -> None:
    docs = cl.user_session.get("docs")
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd == "/docs":
        msg = docs_table(docs) if docs else "No PDFs yet. Attach some. 📎"
        await cl.Message(content=msg).send()

    elif cmd == "/reset":
        cl.user_session.set("context", {})
        cl.user_session.set("pending", None)
        cl.user_session.set("pending_key", None)
        await cl.Message(content="Okay, I forgot what you told me about yourself.").send()

    elif cmd == "/scope":
        rest = parts[1] if len(parts) > 1 else ""
        target = next((d for d in docs.values()
                       if rest.lower().startswith(d["name"].lower())), None)
        if not target:
            await cl.Message(
                content="Usage: `/scope file.pdf shift=9-5, location=chennai` "
                        "(or `/scope file.pdf none`). Use `/docs` to see file names."
            ).send()
            return
        spec = rest[len(target["name"]):].strip()
        raw = {}
        if spec.lower() != "none":
            for pair in spec.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    raw[k.strip()] = v.strip()
        target["scope"] = clean_scope(raw)
        save_doc_cache(target)
        await cl.Message(
            content=f"Updated **{target['name']}** → written for: "
                    f"{scope_text(target['scope'])}"
        ).send()

    else:
        await cl.Message(content="Unknown command. Try `/docs`, `/scope`, `/reset`.").send()


@cl.on_message
async def on_message(message: cl.Message):
    question = (message.content or "").strip()

    if question.startswith("/"):
        await handle_command(question)
        return

    pdfs = [e for e in (message.elements or []) if is_pdf(e)]
    if pdfs:
        await load_pdfs(pdfs)
        if not question:
            await cl.Message(
                content="Now ask your question and tell me about yourself. "
                        "Example: *I work the 9-5 shift, what benefits do I get?*"
            ).send()
            return
    if not question:
        return

    docs = cl.user_session.get("docs")
    collection = cl.user_session.get("collection")
    if not docs:
        await cl.Message(content="Please attach your PDFs first. 📎").send()
        return

    options = known_options(docs)
    context = dict(cl.user_session.get("context") or {})
    pending = cl.user_session.get("pending")
    pending_key = cl.user_session.get("pending_key")
    skip_ask = False

    if pending and question.lower() in ("all", "any", "skip"):
        question, skip_ask = pending, True
        cl.user_session.set("pending", None)
        cl.user_session.set("pending_key", None)
    else:
        found = await cl.make_async(extract_context)(question, options)
        if not found and pending:
            matched = False
            if pending_key and pending_key in options:
                for v in options[pending_key]:
                    if same(v, question) or norm(v) in norm(question):
                        found = {pending_key: v}
                        matched = True
                        break
            if not matched:
                for k, vals in options.items():
                    for v in vals:
                        if same(v, question) or norm(v) in norm(question):
                            found = {k: v}
                            matched = True
                            break
                    if matched:
                        break
            if not found and pending_key:
                found = {pending_key: norm(question)}

        if found:
            context.update(found)
            cl.user_session.set("context", context)
            if pending:
                question = pending
            cl.user_session.set("pending", None)
            cl.user_session.set("pending_key", None)

    if "location" in options and "location" not in context:
        known_locs = {norm(x) for x in options["location"]}
        for word in re.findall(r"[a-z]{3,}", question.lower()):
            if word in COMMON_LOCATIONS and word not in known_locs:
                await cl.Message(
                    content=f"⚠️ No uploaded PDF covers **location = {word}**.\n\n"
                            f"Available locations: {', '.join(options['location'])}\n\n"
                            "Try one of those or type `/reset`."
                ).send()
                return

    bad = unknown_values(options, context)
    if bad:
        k, v = bad[0]
        await cl.Message(
            content=f"⚠️ No PDF covers **{k} = {v}**.\n\n"
                    f"I have: {', '.join(options[k])}\n\n"
                    "Tell me the right one, or type `/reset`."
        ).send()
        context.pop(k, None)
        cl.user_session.set("context", context)
        return

    eligible = eligible_docs(docs, context)

    try:
        hits = await cl.make_async(retrieve)(collection, question, docs, eligible)
    except RagError as e:
        await cl.Message(content=f"⚠️ {e}").send()
        return

    if not hits:
        await cl.Message(
            content="I couldn't find anything about that in the matching PDFs."
        ).send()
        return

    amb = None if skip_ask else ambiguous_key(top_docs(hits, docs), context)
    if amb:
        key, values = amb
        cl.user_session.set("pending", question)
        cl.user_session.set("pending_key", key)
        await cl.Message(
            content=f"Your PDFs differ by **{key}**: {', '.join(values)}.\n\n"
                    f"Which **{key}** applies to you? "
                    f"(or type `all` to search every PDF)"
        ).send()
        return

    msg = cl.Message(content="")

    async def on_token(text: str):
        await msg.stream_token(text)

    try:
        reply = await answer_stream(question, hits, context, on_token)
    except RagError as e:
        await cl.Message(content=f"⚠️ {e}").send()
        return

    if reply.strip().startswith(NOT_FOUND_TOKEN):
        await cl.Message(
            content="The matching PDFs don't seem to answer that question."
        ).send()
        return

    lines = []
    for h in cited_sources(reply, hits):
        snippet = h["text"][:150].replace("\n", " ")
        lines.append(
            f"📄 **{h['meta']['source']}**, page {h['meta']['page']} "
            f"· for: {scope_text(h['scope'])}\n> {snippet}..."
        )
    footer = "\n\n---\n"
    if context:
        footer += f"🔎 Answered for: **{scope_text(context)}**\n\n"
    footer += "\n\n".join(lines)
    await msg.stream_token(footer)
    await msg.send()
