"""
app.py  -  the Chainlit web interface.
"""

import chainlit as cl

from rag_core import (
    NOT_FOUND_TOKEN,
    RagError,
    answer_stream,
    build_index,
    pick_source,
    retrieve,
)

WELCOME = (
    "## 📄 Chat with your PDF\n"
    "Click the **attach** button next to the message box and add a PDF. "
    "You can type your question in the same message, or ask later.\n\n"
    "I answer **only from the document** and show you the exact source page."
)


def is_pdf(element) -> bool:
    name = (getattr(element, "name", "") or "").lower()
    mime = (getattr(element, "mime", "") or "").lower()
    return mime == "application/pdf" or name.endswith(".pdf")


async def load_pdf_into_session(element) -> bool:
    error = None
    cached = False

    async with cl.Step(name=f"Reading {element.name}") as step:
        try:
            collection, cached = await cl.make_async(build_index)(element.path, element.name)
        except RagError as e:
            error = e
            step.output = f"⚠️ {e}"
        else:
            cl.user_session.set("collection", collection)
            cl.user_session.set("pdf_name", element.name)
            step.output = "Already indexed." if cached else "Indexed and ready."

    if error:
        raise error
    return cached


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("collection", None)
    cl.user_session.set("pdf_name", None)
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = (message.content or "").strip()

    pdfs = [e for e in (message.elements or []) if is_pdf(e)]
    if pdfs:
        try:
            await load_pdf_into_session(pdfs[-1])
        except RagError as e:
            await cl.Message(content=f"⚠️ {e}").send()
            return

        if not question:
            await cl.Message(
                content=f"✅ **{pdfs[-1].name}** is ready. Ask me anything about it."
            ).send()
            return

    if not question:
        return

    collection = cl.user_session.get("collection")
    if collection is None:
        await cl.Message(
            content="Please attach a PDF first using the attach button next to the message box. 📎"
        ).send()
        return

    try:
        hits = await cl.make_async(retrieve)(collection, question)
    except RagError as e:
        await cl.Message(content=f"⚠️ {e}").send()
        return

    if not hits:
        await cl.Message(content="I couldn't find anything about that in this PDF.").send()
        return

    msg = cl.Message(content="")

    async def on_token(text: str):
        await msg.stream_token(text)

    try:
        reply = await answer_stream(question, hits, on_token)
    except RagError as e:
        await cl.Message(content=f"⚠️ {e}").send()
        return

    if NOT_FOUND_TOKEN in reply:
        await cl.Message(content="The PDF doesn't seem to answer that question.").send()
        return

    _, best = pick_source(reply, hits)
    snippet = best["text"][:250].replace("\n", " ")
    source = (
        f"\n\n---\n📄 **Source:** {best['meta']['source']}, page {best['meta']['page']} "
        f"· distance {best['distance']:.2f}\n> {snippet}..."
    )
    await msg.stream_token(source)
    await msg.send()