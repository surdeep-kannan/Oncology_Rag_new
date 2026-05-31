import streamlit as st
import time

st.set_page_config(
    page_title="Medical RAG Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Beautiful Sidebar Configuration Panel ---
st.sidebar.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=70)
st.sidebar.title("Clinical RAG Panel")
st.sidebar.markdown("---")

# Active Toggle for RL
use_rl = st.sidebar.toggle(
    "Use RL-Optimized Settings", 
    value=True,
    help="Toggle between mathematically optimized reinforcement learning parameters and baseline unoptimized configurations."
)

st.sidebar.markdown("---")

if use_rl:
    st.sidebar.success("🟢 RL-Optimized Mode Active")
else:
    st.sidebar.warning("🔴 Baseline Mode Active")

# --- UI Styling & Theme ---
st.title("🩺 Medical RAG Explorer")
st.markdown("##### *Empirical Reinforcement Learning for Medical Information Retrieval & Synthesis*")

ANSWER_CARD = """
<div style="
    background: linear-gradient(135deg, #1a1f2e, #0f3460);
    border-left: 5px solid #00d4aa;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,212,170,0.15);
">
<p style="color:#00d4aa; font-size:11px; font-weight:700; letter-spacing:2px; margin:0 0 10px 0; text-transform:uppercase;">🤖 AI Answer ({mode})</p>
<p style="color:#f0f4f8; font-size:16px; line-height:1.75; margin:0;">{answer}</p>
</div>
"""

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            mode_lbl = "RL-Optimized" if msg.get("is_rl", True) else "Baseline Mode"
            st.markdown(ANSWER_CARD.format(mode=mode_lbl, answer=msg["content"]), unsafe_allow_html=True)
            if "sources" in msg:
                with st.expander(f"View Retrieved Sources ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {src.get('heading', 'No Heading')}**")
                        st.caption(f"📄 {src.get('source_file')} | Page: {src.get('page_start')}-{src.get('page_end')}")
                        st.write(src.get('content'))
                        st.divider()
        else:
            st.markdown(msg["content"])

# --- Chat Input ---
query = st.chat_input("Ask a clinical question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    casual = ["hi", "hello", "hey", "how are you", "how are you?", "hi there", "hello there"]
    if query.strip().lower() in casual:
        reply = "Hello! Ask me a specific medical question and I'll search the clinical knowledge base for you!"
        with st.chat_message("assistant"):
            st.markdown(ANSWER_CARD.format(mode="System", answer=reply), unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply, "is_rl": True})
    else:
        with st.chat_message("assistant"):
            status = st.empty()
            try:
                from medrag.hm_rag import HMRAGPipeline

                @st.cache_resource
                def get_pipeline():
                    return HMRAGPipeline()

                pipeline = get_pipeline()
                
                # Dynamic mode execution
                res = pipeline.run(
                    query, 
                    progress_callback=lambda m: status.markdown(f"⏳ *{m}*"),
                    use_rl=use_rl
                )
                status.empty()

                mode_str = "RL-Optimized" if use_rl else "Baseline Mode"
                st.markdown(ANSWER_CARD.format(mode=mode_str, answer=res["answer"]), unsafe_allow_html=True)

                with st.expander(f"View Retrieved Sources ({len(res['context'])})"):
                    for i, src in enumerate(res['context'], 1):
                        st.markdown(f"**{i}. {src.get('heading', 'No Heading')}**")
                        st.caption(f"📄 {src.get('source_file')} | Page: {src.get('page_start')}-{src.get('page_end')}")
                        st.write(src.get('content'))
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["context"],
                    "is_rl": use_rl
                })
            except Exception as e:
                status.empty()
                st.error(f"Error: {e}")
