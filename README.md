```markdown
# Creation of Intelligent Bug Diagnosis Platform with Fix Recommendation Assistance

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

## Overview
**Creation of Intelligent Bug Diagnosis Platform with Fix Recommendation Assistance** is an enterprise-grade, AI-driven defect diagnosis, root cause analysis, and automated remediation platform. Engineered to drastically minimize Mean Time to Resolution (MTTR), the system ingests complex runtime logs, vectorizes error signatures via ChromaDB for semantic retrieval, and executes a rigorous 4-stage multi-agent diagnostic pipeline to isolate anomalies and generate production-ready code patches.

---

## System Architecture & Multi-Agent Data Flow

```

[Raw Log Ingestion Engine]
│
▼ (Asynchronous Chunking & Noise Filtering)
[Stage 1: Log Analysis Agent] ──────► Extract Levels, Call Sites & Anomalies
│
▼
[Stage 2: Triage & Classifier] ─────► Determine Severity & Exception Class
│
▼
[Stage 3: Root Cause Diagnostics] ──► RAG Correlation via ChromaDB Vector Memory
│
▼
[Stage 4: Fix Recommendation Advisor] ─► Generate Executable Code Patches & Guardrails
│
▼
[Interactive Dashboard / REST API Output]

```

---

## Core Multi-Agent Pipeline
The platform leverages a sequential multi-agent orchestration architecture to process software failures with absolute precision:

* **Stage 1: Log Analysis Agent**
  * Parses raw, unstructured log lines at scale using high-performance regex engines.
  * Extracts structural log levels (`ERROR`, `CRITICAL`, `WARN`), isolates execution call sites, and filters out noise.
* **Stage 2: Triage & Classification Agent**
  * Evaluates extracted error signatures against known system exception catalogues.
  * Assigns deterministic severity weights (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) and categorizes core exception classes.
* **Stage 3: Root Cause Diagnostics Agent**
  * Correlates live runtime traces with historical RAG context stored in persistent vector memory.
  * Pinpoints underlying failure vectors and evaluates compound systemic risk footprints.
* **Stage 4: Fix Recommendation Advisor Agent**
  * Synthesizes cross-stage diagnostic telemetry to produce executable code patches (e.g., connection pool adjustments, token interceptors).
  * Outlines step-by-step remediation protocols and defensive architectural guardrails.

---

## Key Capabilities & Features
* **High-Speed Log Ingestion Engine**: Multi-threaded chunking and noise-filtering pipeline capable of processing large enterprise logs (6MB+) in under 30 seconds.
* **Persistent Vector RAG (ChromaDB)**: Advanced vector indexing for historical bug records, enabling low-latency similarity matching and intelligent duplicate detection.
* **Automated Integration Test Suite**: Comprehensive testing harness built to validate multi-agent orchestration, vector state synchronization, and classification accuracy.
* **Benchmark Knowledge Base Seeding**: Pre-configured seeding mechanisms for standard exception traces and verified mitigation patterns.
* **Statistical Analytics Engine**: Real-time telemetry tracking error distributions, system risk indices, and mean triage latency metrics.
* **Secure Session Handling**: Built-in user authentication and secure token-based workspace session management.

---

## Technology Stack
* **Backend Framework**: FastAPI (Python 3.10+), Uvicorn ASGI Server
* **Vector Store & Retrieval**: ChromaDB (Persistent Client, Cosine Similarity Indexing)
* **Concurrency & Processing**: Asynchronous ThreadPoolExecutors, Pre-compiled Regular Expressions
* **Frontend Dashboard**: Responsive Single-Page Interface (HTML5, Modern Dark Theme CSS3, Vanilla JavaScript)

---

## Installing Requirements

Follow these steps to set up your local development environment and install all necessary python dependencies:

1. **Clone the repository and navigate to the root directory**:
   ```bash
   git clone <repository-url>
   cd intelligent-bug-diagnosis-platform

```

2. **Create and activate a virtual environment (Recommended)**:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install the required packages**:
```bash
pip install --upgrade pip
pip install fastapi uvicorn chromadb pydantic

```



---

## Application Server Starting

Once dependencies are successfully installed, follow these instructions to launch the application server:

1. **Start the backend server using Uvicorn**:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000

```


2. **Verify server status**:
* Successful startup will display log messages indicating Uvicorn is running on `http://127.0.0.1:8000`.


3. **Access the Web Dashboard**:
* Open your web browser and navigate to `http://127.0.0.1:8000` to interact with the diagnostic platform dashboard.



---

## API Reference Summary

* `POST /api/v1/register` — Register a new platform operator account.
* `POST /api/v1/signin` — Authenticate and initialize an active user session.
* `POST /api/v1/analyze-bug` — Trigger the 4-stage multi-agent diagnostic pipeline for target stack traces.
* `POST /api/v1/ingest-file` — Execute high-speed parsing and vector indexing for uploaded trace files (`.log`, `.txt`, `.json`, `.csv`).
* `POST /api/v1/deduplicate` — Run trace similarity queries against vector memory databases.
* `GET /api/v1/analytics` — Fetch vector database health metrics and bug categorization distributions.
* `POST /api/v1/seed-kb` — Seed the vector knowledge base with baseline benchmark records.
* `POST /api/v1/run-tests` — Execute the automated multi-agent integration test runner.
* `GET /api/v1/statistical-analysis` — Retrieve aggregate system risk scores and performance telemetry.

---

## Future Scope & Roadmap

* **Advanced LLM Orchestration**: Transitioning agent modules to support dynamic plug-and-play LLM providers (OpenAI, Anthropic, local Ollama models).
* **CI/CD Pipeline Integrations**: Direct GitHub Actions and GitLab CI integrations to automatically analyze continuous integration build failures and open pull requests with fixes.
* **Distributed Vector Sharding**: Scaling ChromaDB cluster configurations to handle multi-terabyte enterprise log archives seamlessly.

---
 ## Live Deployment Link:
https://ai-smart-bug-analyzer-and-fix-advisor-1.onrender.com/
---
