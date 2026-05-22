"""
🚢 Global Shipping Intelligence — Streamlit RAG App
Run with: streamlit run app.py
"""

import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PERSIST_DIR = "./shipping_db"
DB_TIMESTAMP_FILE = "./shipping_db/last_updated.txt"

SOURCES = [
    # ── UNCTAD ───────────────────────────────────────────────────────────
    {"url": "https://unctad.org/publication/review-maritime-transport-2024",
     "label": "UNCTAD - Maritime Transport Review 2024", "use_jina": False},
    {"url": "https://unctad.org/topic/transport-and-trade-logistics",
     "label": "UNCTAD - Trade Logistics Overview",       "use_jina": False},
    {"url": "https://unctadstat.unctad.org/insights/theme/111",
     "label": "UNCTAD - Liner Shipping Connectivity",    "use_jina": True},
    # ── IMO ──────────────────────────────────────────────────────────────
    {"url": "https://www.imo.org/en/MediaCentre/Pages/Home.aspx",
     "label": "IMO - Media Centre",                      "use_jina": True},
    {"url": "https://www.imo.org/en/ourwork/environment/pages/ghg-emissions.aspx",
     "label": "IMO - GHG Emissions Regulations",         "use_jina": True},
    # ── World Bank ───────────────────────────────────────────────────────
    {"url": "https://www.worldbank.org/en/topic/transport/brief/logistics-performance-index",
     "label": "World Bank - LPI",                        "use_jina": True},
    # ── US Government ────────────────────────────────────────────────────
    {"url": "https://www.fmc.gov/resources/",
     "label": "FMC - Ocean Carrier Resources",           "use_jina": False},
    {"url": "https://www.marad.dot.gov/wp-content/uploads/pdf/MARAD_2022_FactCard_Final.pdf",
     "label": "MARAD - US Maritime Fact Card",           "use_jina": False},
    # ── European Sources ─────────────────────────────────────────────────
    {"url": "https://www.emsa.europa.eu/we-do.html",
     "label": "EMSA - European Maritime Safety",         "use_jina": False},
    {"url": "https://ec.europa.eu/eurostat/statistics-explained/index.php/Maritime_transport_statistics",
     "label": "Eurostat - Maritime Transport Stats",     "use_jina": True},
    # ── Ports ────────────────────────────────────────────────────────────
    {"url": "https://www.portofrotterdam.com/en/the-port/facts-figures",
     "label": "Port of Rotterdam - Facts & Figures",     "use_jina": True},
    {"url": "https://www.portoflosangeles.org/business/statistics",
     "label": "Port of Los Angeles - Statistics",        "use_jina": True},
    {"url": "https://www.mpa.gov.sg/maritime-singapore/what-maritime-singapore-offers/hub-port",
     "label": "MPA Singapore - Hub Port",                "use_jina": True},
    # ── Competitors ──────────────────────────────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/Mediterranean_Shipping_Company",
     "label": "Wikipedia - MSC",                         "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Hapag-Lloyd",
     "label": "Wikipedia - Hapag-Lloyd",                 "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/CMA_CGM",
     "label": "Wikipedia - CMA CGM",                     "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Evergreen_Marine_Corporation",
     "label": "Wikipedia - Evergreen Marine",            "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/COSCO_Shipping",
     "label": "Wikipedia - COSCO Shipping",              "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Yang_Ming_Marine_Transport",
     "label": "Wikipedia - Yang Ming",                   "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Ocean_Network_Express",
     "label": "Wikipedia - ONE (Ocean Network Express)", "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Maersk",
     "label": "Wikipedia - Maersk",                      "use_jina": False},
    # ── Market Data ──────────────────────────────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/List_of_largest_container_shipping_companies",
     "label": "Wikipedia - Largest Container Carriers",  "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/List_of_busiest_container_ports",
     "label": "Wikipedia - Busiest Container Ports",     "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/Logistics_performance_index",
     "label": "Wikipedia - Logistics Performance Index", "use_jina": False},
    # ── Alliances ────────────────────────────────────────────────────────
    {"url": "https://en.wikipedia.org/wiki/Ocean_Alliance",
     "label": "Wikipedia - Ocean Alliance",              "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/THE_Alliance",
     "label": "Wikipedia - THE Alliance",                "use_jina": False},
    {"url": "https://en.wikipedia.org/wiki/2M_Alliance",
     "label": "Wikipedia - 2M Alliance",                 "use_jina": False},
    # ── News ─────────────────────────────────────────────────────────────
    {"url": "https://safety4sea.com/unctad-review-of-maritime-transport-2024/",
     "label": "Safety4Sea - UNCTAD 2024 Report",         "use_jina": False},
]

SUGGESTED_QUESTIONS = [
    "What is the global market share of top shipping carriers?",
    "How does MSC compare to Maersk in fleet size?",
    "What are the busiest container ports in the world?",
    "What are the latest IMO environmental regulations?",
    "Which countries rank highest in the Logistics Performance Index?",
    "What shipping alliances exist and who are their members?",
    "What happened to freight rates in 2024?",
    "How did the Red Sea crisis affect global shipping?",
]


# ─────────────────────────────────────────────
# SCRAPER FUNCTIONS
# ─────────────────────────────────────────────

def scrape_with_bs4(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["nav", "footer", "header", "script", "style", "aside", "form", "iframe"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def scrape_with_jina(url):
    r = requests.get(f"https://r.jina.ai/{url}",
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"},
                     timeout=20)
    r.raise_for_status()
    return r.text

def scrape_page(url, label, use_jina=False):
    try:
        text = scrape_with_jina(url) if use_jina else scrape_with_bs4(url)
    except Exception as e:
        if not use_jina:
            try:
                text = scrape_with_jina(url)
            except:
                return None, f"✗ Failed"
        else:
            return None, f"✗ Failed: {str(e)[:60]}"

    if len(text.strip()) < 200:
        return None, "⚠ Too short"

    return Document(page_content=text[:20000], metadata={"source": url, "label": label}), f"✓ {len(text):,} chars"


# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────

def get_last_updated():
    if os.path.exists(DB_TIMESTAMP_FILE):
        with open(DB_TIMESTAMP_FILE) as f:
            try:
                return datetime.fromisoformat(f.read().strip())
            except:
                return None
    return None

def mark_db_fresh():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(DB_TIMESTAMP_FILE, "w") as f:
        f.write(datetime.now().isoformat())

def is_db_stale(days=7):
    last = get_last_updated()
    if not last:
        return True
    return datetime.now() - last > timedelta(days=days)

def db_exists():
    return (
        os.path.exists(PERSIST_DIR) and
        os.path.isdir(PERSIST_DIR) and
        any(f.endswith(".faiss") for f in os.listdir(PERSIST_DIR))
    )


# ─────────────────────────────────────────────
# VECTORSTORE
# ─────────────────────────────────────────────

def build_vectorstore(docs, progress_bar=None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if progress_bar:
        progress_bar.progress(0.7, text="🧠 Embedding chunks into vector DB...")
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(PERSIST_DIR)
    mark_db_fresh()
    return vs, len(chunks)

def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)


# ─────────────────────────────────────────────
# QA CHAIN
# ─────────────────────────────────────────────

def build_qa_chain(vectorstore):
    prompt_template = """You are a global shipping industry analyst. Use the context below to answer the question as specifically and factually as possible. Include numbers, statistics, company names, and dates when available. Do NOT say "I don't have information" if relevant data exists in the context.

Context:
{context}

Question: {question}

Answer with specific facts and figures from the context:"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.session_state.api_key)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )}
    )


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🚢 Shipping Intelligence RAG",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #00356B 0%, #0066CC 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #cce0ff; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .source-tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
    }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #0066CC;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stChatMessage { border-radius: 12px; }
    .db-status-ok  { color: #28a745; font-weight: 600; }
    .db-status-old { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚢 Global Shipping Intelligence</h1>
    <p>RAG-powered Q&A across government sources, port authorities & competitor data</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Maersk_Logo.svg/320px-Maersk_Logo.svg.png", width=140)
    st.markdown("---")

    # DB Status
    st.subheader("📦 Knowledge Base")
    last_updated = get_last_updated()
    if db_exists() and last_updated:
        age = datetime.now() - last_updated
        age_str = f"{age.days}d {age.seconds//3600}h ago" if age.days > 0 else f"{age.seconds//3600}h ago"
        freshness = "db-status-ok" if not is_db_stale() else "db-status-old"
        st.markdown(f"**Status:** <span class='{freshness}'>{'✅ Fresh' if not is_db_stale() else '⚠️ Stale'}</span>", unsafe_allow_html=True)
        st.markdown(f"**Last scraped:** {age_str}")
        st.markdown(f"**Sources:** {len(SOURCES)}")
    elif db_exists():
        st.markdown("**Status:** ✅ Ready")
        st.markdown(f"**Sources:** {len(SOURCES)}")
    else:
        st.markdown("**Status:** ⚪ Not built yet")

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("🔄 Build DB", use_container_width=True,
                               help="Scrape all sources and build vector DB")
    with col2:
        rebuild_btn = st.button("🗑️ Rebuild", use_container_width=True,
                                 help="Delete existing DB and rebuild from scratch")

    st.markdown("---")

    # Sources list
    with st.expander("📚 View all sources", expanded=False):
        categories = {}
        for s in SOURCES:
            cat = s["label"].split(" - ")[0]
            categories.setdefault(cat, []).append(s["label"])
        for cat, labels in categories.items():
            st.markdown(f"**{cat}**")
            for l in labels:
                st.markdown(f"  • {l.split(' - ', 1)[-1]}")

    st.markdown("---")
    st.caption("⚠️ Data is scraped — not real-time.\nRebuild weekly for freshness.")


# ── Init session state ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── DB Build Logic ─────────────────────────────────────────────────────────────
def force_delete_dir(path):
    """Windows-safe directory deletion — unlocks read-only files before removing."""
    import shutil, stat

    def remove_readonly(func, fpath, _):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    try:
        shutil.rmtree(path, onexc=remove_readonly)   # Python 3.12+
    except TypeError:
        shutil.rmtree(path, onerror=remove_readonly) # Python < 3.12


def run_build(force_rebuild=False):
    if not st.session_state.api_key:
        st.error("OpenAI API key not found in .env file.")
        return

    if force_rebuild and os.path.exists(PERSIST_DIR):
        # Release FAISS handles BEFORE deleting — prevents WinError 5
        st.session_state.qa_chain = None
        st.session_state.db_loaded = False
        import gc; gc.collect()   # force garbage collection to release file locks
        try:
            force_delete_dir(PERSIST_DIR)
        except Exception as e:
            st.error(f"❌ Could not delete old DB: {e}\n\nTry manually deleting the `shipping_db` folder and re-running.")
            return

    with st.status("🔄 Building knowledge base...", expanded=True) as status:
        st.write(f"📡 Scraping {len(SOURCES)} sources...")
        progress = st.progress(0, text="Starting scrape...")

        docs = []
        scrape_log = []
        for i, item in enumerate(SOURCES):
            pct = int((i / len(SOURCES)) * 60)
            progress.progress(pct, text=f"Scraping {i+1}/{len(SOURCES)}: {item['label'][:40]}...")
            doc, msg = scrape_page(item["url"], item["label"], item.get("use_jina", False))
            scrape_log.append(f"{'✓' if doc else '✗'} {item['label']}: {msg}")
            if doc:
                docs.append(doc)
            time.sleep(0.8)

        st.write(f"✅ Scraped {len(docs)}/{len(SOURCES)} pages successfully")

        progress.progress(0.65, text="✂️ Chunking documents...")
        st.write("✂️ Chunking and embedding...")

        vs, n_chunks = build_vectorstore(docs, progress)

        progress.progress(1.0, text="Done!")
        st.write(f"🧠 Created {n_chunks} chunks → vector DB saved")

        st.session_state.qa_chain = build_qa_chain(vs)
        status.update(label="✅ Knowledge base ready!", state="complete")

    with st.expander("📋 Scrape log", expanded=False):
        for line in scrape_log:
            st.text(line)

if build_btn:
    if db_exists():
        vs = load_vectorstore()
        st.session_state.qa_chain = build_qa_chain(vs)
        st.session_state.db_loaded = True
    else:
        run_build(force_rebuild=False)

if rebuild_btn:
    run_build(force_rebuild=True)

# Auto-load existing DB on startup (only once)
if db_exists() and st.session_state.qa_chain is None and st.session_state.api_key and not st.session_state.db_loaded:
    try:
        vs = load_vectorstore()
        st.session_state.qa_chain = build_qa_chain(vs)
        st.session_state.db_loaded = True
        st.rerun()  # rerun once to refresh sidebar status
    except Exception as e:
        st.warning(f"Could not auto-load DB: {e}")
        st.session_state.db_loaded = True  # prevent retry loop


# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.api_key:
    st.error("⚠️ OpenAI API key not found. Please add `OPENAI_API_KEY=sk-...` to your `.env` file and restart the app.")
elif not db_exists():
    st.info("👈 Click **Build DB** in the sidebar to scrape sources and build the knowledge base (~3-4 mins).")
else:
    # Suggested questions (only shown when chat is empty)
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🚢"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.markdown(" ".join([f'<span class="source-tag">📌 {s}</span>' for s in msg["sources"]]),
                            unsafe_allow_html=True)

    # Determine what question to answer — typed input OR suggested button
    prompt = None
    if typed := st.chat_input("Ask anything about global shipping, competitors, ports, regulations..."):
        prompt = typed
    elif st.session_state.get("pending_question"):
        prompt = st.session_state.pop("pending_question")

    # Process the question
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🚢"):
            if st.session_state.qa_chain is None:
                st.error("Knowledge base not loaded. Click Build DB in the sidebar.")
            else:
                with st.spinner("Searching knowledge base..."):
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    sources = list({doc.metadata["label"] for doc in result["source_documents"]})

                st.markdown(answer)
                st.markdown(" ".join([f'<span class="source-tag">📌 {s}</span>' for s in sources]),
                            unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear"):
            st.session_state.messages = []
            st.rerun()