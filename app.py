"""
app.py — Streamlit front-end for Teleoptimal.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README / chat notes)
"""

import hmac
import uuid

import streamlit as st
from anthropic import Anthropic

import rag_core as rc

st.set_page_config(page_title="Teleoptimal", page_icon="📚", layout="centered")


# ── shared, per-process resources ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando índice de PDFs… (la primera vez puede tardar)")
def get_store() -> rc.DocumentStore:
    return rc.DocumentStore()


@st.cache_resource
def get_client() -> Anthropic:
    return Anthropic(api_key=rc.ANTHROPIC_API_KEY)


@st.cache_resource
def get_global_limiter() -> rc.RateLimiter:
    return rc.RateLimiter(rc.GLOBAL_DAILY_MESSAGES)


# ── gates ──────────────────────────────────────────────────────────────────────

def require_password() -> None:
    pw = rc.cfg("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    st.title("📚 Teleoptimal")
    entered = st.text_input("Contraseña", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not rc.cfg(k)]
if missing:
    st.error(f"Faltan claves: {', '.join(missing)}. Agregalas en .env (local) o en Secrets (Streamlit Cloud).")
    st.stop()

require_password()

# ── session state ──────────────────────────────────────────────────────────────

ss = st.session_state
ss.setdefault("sid", uuid.uuid4().hex)
ss.setdefault("messages", [])  # what the chat UI shows
ss.setdefault("conv", rc.Conversation())  # what Claude remembers
ss.setdefault("limiter", rc.RateLimiter(rc.MAX_DAILY_MESSAGES))

store = get_store()
client = get_client()
global_limiter = get_global_limiter()

# model picker (only if MODEL_CHOICES lists more than one)
model = rc.MODEL_CHOICES[0]
if len(rc.MODEL_CHOICES) > 1:
    model = st.sidebar.selectbox("Modelo", rc.MODEL_CHOICES)

# ── chat ───────────────────────────────────────────────────────────────────────

st.title("📚 Teleoptimal")
st.caption("Preguntas sobre los textos cargados · respuestas con cita del archivo fuente")


def render(i: int, m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                with st.expander("Textos consultados"):
                    st.markdown("\n".join(f"- {s}" for s in m["sources"]))
            if m.get("pdf"):
                st.download_button("⬇️ Descargar respuesta en PDF", m["pdf"],
                                   file_name=m["filename"], mime="application/pdf", key=f"dl_{i}")


for i, m in enumerate(ss.messages):
    render(i, m)

prompt = st.chat_input("Escribí tu pregunta…")
if prompt:
    ss.messages.append({"role": "user", "content": prompt})
    render(len(ss.messages) - 1, ss.messages[-1])

    if not ss.limiter.can_send():
        reply = {"role": "assistant", "content": "❌ Límite diario alcanzado. Volvé a intentar mañana."}
    elif not global_limiter.can_send():
        reply = {"role": "assistant", "content": "❌ La app alcanzó su límite diario de uso. Volvé a intentar mañana."}
    else:
        try:
            with st.spinner("Buscando en los textos y redactando…"):
                result = rc.answer(client, model, store, ss.conv, prompt)
            ss.limiter.record()
            global_limiter.record()
            reply = {"role": "assistant", **result}
        except Exception as exc:
            rc.logger.error("Error answering: %s", exc, exc_info=True)
            reply = {"role": "assistant", "content": f"Ocurrió un error: {exc}"}

    ss.messages.append(reply)
    render(len(ss.messages) - 1, reply)

# ── sidebar (drawn last so the quota is up to date) ────────────────────────────

with st.sidebar:
    st.metric("Mensajes restantes hoy", ss.limiter.remaining())
    if st.button("🧹 Limpiar conversación"):
        ss.messages, ss.conv = [], rc.Conversation()
        st.rerun()

    st.subheader("PDFs cargados")
    lines = store.status_lines()
    st.markdown("\n\n".join(lines) if lines else "_No hay PDFs cargados._")

    admin_pw = rc.cfg("ADMIN_PASSWORD")
    if admin_pw:
        with st.expander("Admin"):
            entered = st.text_input("Clave de admin", type="password", key="admin_pw")
            if entered and hmac.compare_digest(entered, admin_pw):
                if st.button("Recargar PDFs"):
                    with st.spinner("Recargando…"):
                        st.info(store.reload(force=False))
                if st.button("Reconstruir índice (gasta embeddings de OpenAI)"):
                    with st.spinner("Re-indexando…"):
                        st.info(store.reload(force=True))
