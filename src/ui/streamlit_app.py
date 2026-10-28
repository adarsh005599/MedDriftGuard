"""Premium local Streamlit presentation UI for MedDriftGuard demo mode."""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

import streamlit as st

from src.ui.chat_client import ChatApiError, send_chat_message

API_BASE_URL = os.getenv("MEDDRIFTGUARD_API_URL", "http://127.0.0.1:8000")
DISCLAIMER = "General health information only — not a diagnosis or a replacement for clinical care."
EXAMPLES = (
    "What home care is mentioned for a cough?",
    "Khansi ke liye ghar par kya dekhbhal ki ja sakti hai?",
    "मुझे सिरदर्द के बारे में सामान्य जानकारी चाहिए।",
    "What lifestyle information is available for acidity?",
)


def _apply_design_system() -> None:
    """Apply static styling only; user content is never interpolated into HTML."""
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        :root { --bg:#080D18; --panel:#0D1423; --card:#121C2E; --border:#24334A; --cyan:#56D8E8; --text:#F3F6FC; --muted:#9CAEC5; --warning:#F4BE68; }
        .stApp { background:radial-gradient(circle at 76% 5%,#102e3d 0%,var(--bg) 32rem); color:var(--text); font-family:Inter,Segoe UI,sans-serif; }
        [data-testid="stHeader"] { background:rgba(8,13,24,.75); border-bottom:1px solid rgba(36,51,74,.45); }
        [data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--border); }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.25rem; }
        h1,h2,h3 { font-family:Manrope,Inter,sans-serif !important; letter-spacing:-.035em; color:var(--text) !important; }
        h1 { font-size:clamp(2rem,4vw,3.25rem) !important; margin-bottom:.3rem !important; }
        p,li,[data-testid="stCaptionContainer"] { color:var(--muted); line-height:1.65; }
        .mdg-brand { display:flex; gap:.65rem; align-items:center; margin-bottom:1.35rem; }.mdg-mark { width:2.25rem;height:2.25rem;display:grid;place-items:center;color:var(--cyan);border:1px solid #2d6a78;border-radius:.75rem;background:linear-gradient(135deg,#163b4a,#121c2e);font-weight:800; }.mdg-name { color:var(--text);font-family:Manrope,sans-serif;font-weight:800;font-size:1.15rem; }.mdg-sub { color:var(--muted);font-size:.75rem; }
        .eyebrow,.assistant-label { color:var(--cyan);font-size:.74rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase; }.demo-badge,.status-badge { display:inline-block;border-radius:999px;padding:.33rem .7rem;font-size:.72rem;font-weight:700;letter-spacing:.06em; }.demo-badge { color:#1a1406;background:var(--warning); }.status-badge { color:#baf7df;border:1px solid #29795d;background:#103126; }
        .hero-card,.feature-card { background:rgba(18,28,46,.9);border:1px solid var(--border);border-radius:1rem;box-shadow:0 10px 30px rgba(0,0,0,.16); }.hero-card { padding:1.35rem;margin:1.5rem 0 1.1rem;background:linear-gradient(115deg,rgba(18,28,46,.96),rgba(22,59,74,.70)); }.feature-card { padding:1rem;min-height:8.5rem; }.feature-card b { display:block;color:var(--text);margin-bottom:.35rem; }
        .stButton button { width:100%;border:1px solid #2b7180;border-radius:.7rem;background:#123442;color:#eafcff;font-weight:600;transition:background .18s ease,transform .18s ease; }.stButton button:hover { border-color:var(--cyan);background:#174959;transform:translateY(-1px);color:white; }.stButton button:focus-visible { outline:3px solid rgba(86,216,232,.35);outline-offset:2px; }
        [data-testid="stChatInput"] { border:1px solid #2a5968;border-radius:1rem;background:var(--card);box-shadow:0 8px 28px rgba(0,0,0,.22); }.footer-note { border-top:1px solid var(--border);color:var(--muted);font-size:.79rem;margin-top:2.6rem;padding:1rem 0 2rem; }
        @media (prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none !important; } } @media (max-width:700px) { .hero-card { padding:1rem; } h1 { font-size:2.15rem !important; } }
        </style>""",
        unsafe_allow_html=True,
    )


def _new_conversation() -> None:
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []


def _visible_answer(answer: str) -> str:
    """Source cards replace raw inline chunk IDs in the presentation UI."""
    return re.sub(r"\s*\[[^\]]+\]", "", answer).strip()


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown(
        f"<p class='assistant-label'>Evidence used · {len(sources)} source(s)</p>",
        unsafe_allow_html=True,
    )
    for source in sources:
        title = " — ".join(
            part for part in (source.get("topic", "").title(), source.get("section", "")) if part
        )
        with st.expander(title or source["chunk_id"], expanded=False):
            st.caption(f"{source['source']} · Source ID: {source['chunk_id']}")
            if source.get("excerpt"):
                st.write(source["excerpt"])
            details = []
            if source.get("topic"):
                details.append(f"Topic: {source['topic']}")
            if source.get("license"):
                details.append(f"Reuse terms: {source['license']}")
            if details:
                st.caption(" · ".join(details))
            st.link_button("Open original source", source["url"], use_container_width=False)


def _render_assistant(item: dict[str, Any]) -> None:
    st.markdown("<p class='assistant-label'>MedDriftGuard</p>", unsafe_allow_html=True)
    st.markdown(_visible_answer(item["content"]))
    if item.get("abstained"):
        st.warning("Safe abstention: no relevant evidence was available for this response.")
    _render_sources(item.get("sources", []))
    with st.expander("Copy response", expanded=False):
        st.code(_visible_answer(item["content"]), language=None)


def _welcome_screen() -> str | None:
    st.markdown(
        "<p class='eyebrow'>Evidence-grounded health information</p>", unsafe_allow_html=True
    )
    st.title("Clear health information,\nbuilt around evidence.")
    st.markdown(
        "MedDriftGuard retrieves from a curated local knowledge base, cites the evidence it uses, and can safely abstain when evidence is missing or irrelevant."
    )
    st.markdown(
        "<div class='hero-card'><b style='color:#F3F6FC'>Designed for clarity. Built for safety.</b><br><span style='color:#9CAEC5'>Ask in English, हिन्दी, or Roman Hinglish. This demo provides general health information only.</span></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for column, (title, body) in zip(
        columns,
        (
            ("Evidence first", "Every displayed source comes from the current retrieval result."),
            (
                "Safe abstention",
                "When evidence is missing or unrelated, the assistant does not guess.",
            ),
            ("Demo honestly", "Generation and verification are deterministic demo components."),
        ),
        strict=True,
    ):
        with column:
            st.markdown(
                f"<div class='feature-card'><b>{title}</b><span>{body}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("#### Try an evidence-backed question")
    selected = None
    example_columns = st.columns(2)
    for index, example in enumerate(EXAMPLES):
        with example_columns[index % 2]:
            if st.button(example, key=f"example_{index}"):
                selected = example
    return selected


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='mdg-brand'><div class='mdg-mark'>M</div><div><div class='mdg-name'>MedDriftGuard</div><div class='mdg-sub'>Evidence-grounded health information</div></div></div>",
            unsafe_allow_html=True,
        )
        if st.button("+ New conversation", use_container_width=True):
            _new_conversation()
            st.rerun()
        st.selectbox(
            "Interface language", ("English", "हिन्दी", "Hinglish"), key="interface_language"
        )
        st.markdown("#### How it works")
        st.caption(
            "1. The current question is rewritten for retrieval.\n\n2. Local sources are retrieved.\n\n3. Demo components return a cited answer or abstain."
        )
        st.markdown("#### System status")
        st.markdown(
            "<span class='status-badge'>LOCAL RETRIEVAL READY</span>", unsafe_allow_html=True
        )
        st.caption("Generation & verification: deterministic demo mode")
        st.markdown("---")
        st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(
        page_title="MedDriftGuard", page_icon="M", layout="wide", initial_sidebar_state="expanded"
    )
    _apply_design_system()
    if "conversation_id" not in st.session_state:
        _new_conversation()
    _sidebar()
    left, right = st.columns((4, 1))
    with left:
        st.caption("MedDriftGuard workspace")
    with right:
        st.markdown("<span class='demo-badge'>DEMO MODE</span>", unsafe_allow_html=True)
    selected_prompt = _welcome_screen() if not st.session_state.messages else None
    if st.session_state.messages:
        st.markdown("## Conversation")
        for item in st.session_state.messages:
            with st.chat_message(item["role"]):
                (
                    _render_assistant(item)
                    if item["role"] == "assistant"
                    else st.markdown(item["content"])
                )
    prompt = selected_prompt or st.chat_input("Ask a general health-information question…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking local evidence…"):
                try:
                    reply = send_chat_message(
                        API_BASE_URL, prompt, st.session_state.conversation_id
                    )
                except ChatApiError as error:
                    st.error(f"{error} Start the FastAPI service, then try again.")
                    return
            assistant_item = {
                "role": "assistant",
                "content": reply["answer"],
                "sources": reply.get("sources", []),
                "abstained": reply.get("abstained", False),
            }
            _render_assistant(assistant_item)
            st.caption(reply.get("mode_note", "DEMO/MOCK MODE"))
        st.session_state.messages.append(assistant_item)
    st.markdown(
        "<div class='footer-note'>"
        + DISCLAIMER
        + " For urgent symptoms or an emergency, contact local emergency services or a clinician. MedDriftGuard is not a diagnostic service.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
