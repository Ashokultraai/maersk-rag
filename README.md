# 🚢 Global Shipping Intelligence — RAG System

A **Retrieval-Augmented Generation (RAG)** application that scrapes 27+ global shipping, government, and competitor websites, stores the data in a local vector database (FAISS), and lets you ask natural language questions about the global shipping industry — all powered by GPT-4o.

Built with **Python + Streamlit**, no paid scraping tools required.

---

## 📸 What It Looks Like

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar              │  Main Chat Area                  │
│  ─────────────────    │  ───────────────────────────     │
│  📦 Knowledge Base    │  🚢 Global Shipping Intelligence │
│  Status: ✅ Fresh     │                                  │
│  Last scraped: 0h ago │  💡 Try asking:                  │
│  Sources: 27          │  [What is MSC fleet size?]       │
│                       │  [Busiest container ports?]      │
│  [🔄 Build DB]        │                                  │
│  [🗑️ Rebuild]         │  > Ask anything...               │
│                       │                                  │
│  > View all sources   │                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
27+ Websites (Gov, Ports, Competitors)
            ↓
   Web Scraping Layer
   ┌─────────────────────────────┐
   │  BeautifulSoup  │   Jina   │
   │  (direct HTML)  │ (fallback│
   │                 │  bypass) │
   └─────────────────────────────┘
            ↓
   Text Chunking (LangChain)
   1000 chars / 150 overlap
            ↓
   OpenAI Embeddings (text-embedding-ada-002)
            ↓
   FAISS Vector Database (local)
            ↓
   MMR Retrieval (k=8, fetch_k=20)
            ↓
   GPT-4o — Answer Generation
            ↓
   Streamlit Chat UI
```

---

## 📦 Data Sources (27 Sources)

### 🌍 International Organizations
| Source | Data |
|--------|------|
| UNCTAD | Maritime Transport Review 2024, Trade Logistics, Liner Shipping Connectivity |
| IMO | Media Centre, GHG Emissions Regulations |
| World Bank | Logistics Performance Index (LPI) |

### 🏛️ Government Sources
| Source | Data |
|--------|------|
| FMC (USA) | Ocean Carrier Resources |
| EMSA (EU) | European Maritime Safety |
| Eurostat | Maritime Transport Statistics |

### 🚢 Port Authorities
| Source | Data |
|--------|------|
| Port of Rotterdam | Facts & Figures |
| Port of Los Angeles | Statistics |
| MPA Singapore | Hub Port Info |

### 🏢 Competitors (via Wikipedia)
| Company |
|---------|
| MSC (Mediterranean Shipping Company) |
| Hapag-Lloyd |
| CMA CGM |
| Evergreen Marine |
| COSCO Shipping |
| Yang Ming |
| ONE (Ocean Network Express) |
| Maersk |

### 📊 Market Data (via Wikipedia)
- List of Largest Container Shipping Companies
- List of Busiest Container Ports
- Logistics Performance Index
- Ocean Alliance, THE Alliance, 2M Alliance

### 📰 News
- Safety4Sea — UNCTAD 2024 Report Summary

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Step 1: Clone the repo
```bash
git clone https://github.com/yourusername/shipping-rag.git
cd shipping-rag
```

### Step 2: Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up your `.env` file
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### Step 5: Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 💬 Usage

### First Run
1. Click **🔄 Build DB** in the sidebar
2. Wait ~3-4 minutes while it scrapes all 27 sources
3. Once complete, start asking questions!

### Asking Questions
- Click any suggested question button, or
- Type your own question in the chat box

### Example Questions
```
- What is the global market share of top shipping carriers?
- How does MSC compare to Maersk in fleet size?
- What are the busiest container ports in the world?
- What are the latest IMO environmental regulations?
- Which countries rank highest in the Logistics Performance Index?
- What shipping alliances exist and who are their members?
- What happened to freight rates in 2024?
- How did the Red Sea crisis affect global shipping?
```

### Keeping Data Fresh
- Click **🗑️ Rebuild** weekly to re-scrape all sources
- The sidebar shows when data was last scraped and whether it's fresh or stale

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **UI** | Streamlit |
| **Scraping** | BeautifulSoup4 + Requests |
| **Scraping fallback** | Jina Reader (free, no API key) |
| **Text splitting** | LangChain Text Splitters |
| **Embeddings** | OpenAI `text-embedding-ada-002` |
| **Vector DB** | FAISS (local, no server needed) |
| **Retrieval** | MMR (Maximal Marginal Relevance) |
| **LLM** | GPT-4o via OpenAI API |
| **Chain** | LangChain LCEL (modern pipeline) |

---

## 📁 Project Structure

```
shipping-rag/
├── app.py              ← Main Streamlit app
├── requirements.txt    ← Python dependencies
├── .env                ← API keys (never commit this!)
├── .gitignore          ← Excludes .env and shipping_db/
├── README.md           ← This file
└── shipping_db/        ← Auto-created: FAISS vector store
    ├── index.faiss
    ├── index.pkl
    └── last_updated.txt
```

---

## ⚠️ Important Notes

### Data Freshness
This app uses **web scraping**, not live APIs. Data is a snapshot from the last rebuild:
- Wikipedia pages update frequently but may lag real changes by weeks
- Government/UNCTAD reports are published annually
- **Rebuild weekly** for the most accurate responses

### What It's Good For
✅ Company background & history (fleet sizes, founding, HQ)  
✅ Industry regulations (IMO, emissions, environmental rules)  
✅ Port statistics & rankings  
✅ Shipping alliance memberships  
✅ General market structure & trends  

### What It's NOT Good For
❌ Real-time freight rates  
❌ Live vessel tracking  
❌ Today's news  
❌ Stock prices  

For live data, use [Maersk Developer API](https://developer.maersk.com) or [MarineTraffic API](https://www.marinetraffic.com/en/ais-api-services).

---

## 🔒 Security

- **Never commit your `.env` file** — it contains your OpenAI API key
- The `.gitignore` already excludes `.env` and `shipping_db/`
- The `shipping_db/` folder can be large — exclude it from git and rebuild locally

---

## 🤝 Contributing

Pull requests welcome! Ideas for improvement:
- Add more shipping news sources (Splash247, Lloyd's List)
- Add PDF ingestion for annual reports
- Add Tavily web search for real-time results
- Deploy to Streamlit Cloud

---

## 👤 Author

Built as a shipping intelligence tool for the maritime industry.  
Powered by OpenAI GPT-4o + LangChain + FAISS + Streamlit.
