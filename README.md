---
name: fact-checker-architecture
overview: Design of a high-throughput multilingual automated fact-checking system for Indian vernacular content, including architecture, pipeline, and detailed team responsibilities.
todos:
  - id: backend-core-apis
    content: Implement core FastAPI backend, DB schema, and message-driven workers for ingestion and fact-checking.
    status: pending
  - id: ml-pipeline-core
    content: Develop multilingual preprocessing, claim extraction, retrieval, and verification ML modules.
    status: pending
  - id: frontend-dashboard
    content: Build React-based dashboard for live monitoring, claim review, and fact database management.
    status: pending
isProject: false
---

# Automated Vernacular Fact-Checking System – Architecture & Plan

## 1. High-level system architecture & pipeline stages

```mermaid
flowchart LR
  subgraph ingestLayer [Ingestion Layer]
    socialAPIs[SocialFeedIngestor]
    rssIngest[NewsRSSIngestor]
    webScraper[WebScraper]
    queueIn[kafka_topic_raw_posts]
  end

  subgraph preprocessingLayer [Preprocessing & Normalization]
    langDetect[LangDetector]
    normalizeText[Normalizer & Cleaner]
    fluffFilter[FluffIrrelevantFilter]
    claimExtractor[ClaimExtractor]
    queueClaims[kafka_topic_claims]
  end

  subgraph factCheckLayer [Fact Retrieval & Verification]
    retriever[HybridRetriever]
    factDB[(FactDB + VectorStore)]
    verifier[ClaimVerifier]
    scorer[ConfidenceScorer]
    queueResults[kafka_topic_results]
  end

  subgraph apiLayer [API & UI]
    backendAPI[BackendAPI(FastAPI)]
    reviewerUI[ReviewerDashboard]
  end

  socialAPIs --> queueIn
  rssIngest --> queueIn
  webScraper --> queueIn

  queueIn --> langDetect --> normalizeText --> fluffFilter --> claimExtractor --> queueClaims
  queueClaims --> retriever
  retriever --> factDB
  factDB --> retriever
  retriever --> verifier --> scorer --> queueResults

  queueResults --> backendAPI --> reviewerUI
```



### 1.1 Stages

- **Ingestion layer**
  - Connectors to Twitter/X, Facebook, Instagram, YouTube comments, WhatsApp-like proxies (if any), and news sites (RSS, sitemaps, HTML scrapers).
  - Write normalized `RawPost` records into `kafka_topic_raw_posts` (or similar message queue) with metadata (source, language hint, timestamp, geo, engagement, author).
- **Preprocessing & claim extraction layer**
  - **Language detection & transliteration**: detect language + script (e.g., Hindi in Devanagari vs Latin), transliterate to canonical script for some models.
  - **Normalization & cleaning**: remove URLs, emojis (optional), HTML, excessive punctuation; normalize Unicode; standardize hashtags/usernames.
  - **Fluff & irrelevant text filtering**: remove greetings, sign-offs, discourse markers, and non-claim chatter; score sentences for claim-likeness.
  - **Claim segmentation & extraction**: split text into sentences/clauses; identify and extract check-worthy factual claims.
  - **Output**: `Claim` records (claim text, language, source_post_id, entities, time, geo, topic) to `kafka_topic_claims`.
- **Fact retrieval & verification layer**
  - **Hybrid retrieval** from fact DB: BM25 + dense embeddings (vector search) + optional keyword filters.
  - **Context construction**: build a context bundle of top-k candidate facts + evidence, normalized entities, and metadata (time, place).
  - **Verification**: ML or rule-based labeler decides {Supported, Refuted, NotEnoughEvidence, NeedsReview} and assigns confidence.
  - **Misinformation detection**: flag posts/claims that are refuted or unknown but potentially dangerous (e.g., health/election-related, viral).
  - **Output**: `FactCheckResult` records to `kafka_topic_results` and persisted in DB.
- **API & presentation layer**
  - **Backend API (FastAPI)** to:
    - Query fact-check results by post URL, ID, or text.
    - Provide search UI for fact-checkers.
    - Expose admin tools to manage fact DB.
  - **Frontend dashboard** (React) for:
    - Monitoring live stream of claims and statuses.
    - Manual review/override of low-confidence cases.
    - Annotation interface for building the verified fact database.
- **Storage & infrastructure**
  - **Operational store**: PostgreSQL (claims, posts, results, users, annotations, jobs).
  - **Search**: Elasticsearch/OpenSearch for keyword + filters.
  - **Vector store**: FAISS / Milvus for dense embeddings of claims/facts.
  - **Object storage**: MinIO/S3 for raw HTML, screenshots, training data dumps.
  - **Orchestration/Deployment**: Docker + Kubernetes (or docker-compose for student scope), Prometheus/Grafana for metrics.

---

## 2. Methods for high-throughput processing (thousands of posts/min)

- **Message queues for decoupling**
  - **Kafka or Redpanda** topics per stage: `raw_posts`, `normalized_posts`, `claims`, `results`.
  - Consumers scaled horizontally; partitions align with throughput needs.
- **Horizontal scaling of stateless workers**
  - Microservices or modular workers:
    - `ingest-service`, `preprocess-service`, `claim-extractor-service`, `retriever-service`, `verifier-service`.
  - Each runs multiple replicas behind a queue; scale out by running more replicas.
- **Batch & micro-batch processing**
  - Process messages in small batches (e.g., 16–128 posts) to exploit vectorization in NLP models (PyTorch batch inference).
  - Group by language to reduce model switching overhead.
- **Asynchronous I/O and concurrency**
  - Use **async FastAPI** for APIs.
  - Workers use async HTTP clients for external APIs, non-blocking DB drivers where possible.
- **Caching**
  - Redis cache for:
    - Popular posts and their results.
    - Recently computed embeddings of common text fragments (e.g., viral messages forwarded multiple times).
    - Frequently used fact entries and retrieval results.
- **Backpressure & prioritization**
  - Kafka consumer groups with max poll interval + lag monitoring.
  - Prioritize high-impact sources (e.g., verified political accounts, highly shared URLs) using priority queues or separate topics.
- **Observability**
  - Metrics: per-stage throughput, latency histograms, error rates, queue lag, GPU utilization.
  - Logs: structured JSON logs for each worker; correlation IDs for tracing per post/claim.

---

## 3. Techniques to maintain context accuracy in the fact database

- **Fact schema design**
  - `Fact` entity fields:
    - `id`, `canonical_claim_text`, `language`, `translations`.
    - `entities` (linked to entity store), `time_validity` (from/to), `geo_scope` (country/state/city).
    - `source` (fact-checking org, article URL, evidence URLs), `verdict` (True/False/Misleading/etc.).
    - `topic_tags` (health, politics, crime, finance, religion, etc.).
- **Entity-centric modeling**
  - Separate `Entity` table (persons, orgs, places, events) with aliases across languages and scripts.
  - Use NER + entity linker to map claims to entities; retrieval is then entity-aware.
- **Time & geography-aware retrieval**
  - Include time and geo filters in retrieval queries.
  - Example: claim about “fuel price now” should prioritize recent facts from the same country/state.
- **Hybrid retrieval for context accuracy**
  - **BM25 (Elasticsearch)** on normalized text, plus filters on language, topic, time.
  - **Dense retrieval (FAISS/Milvus)** using multilingual encoders (e.g., XLM-R, IndicBERT) for semantic similarity across languages.
  - **Re-ranking** with cross-encoder models that take `[claim, fact]` pairs and output relevance scores.
- **Normalization & canonicalization pipeline for new facts**
  - When ingesting new verified facts:
    - Normalize text (same preprocessing as incoming claims).
    - Extract entities, time, location, topics.
    - Generate embeddings for canonical claim and translations.
    - Store full provenance (links to sources, authors, publication date).
- **Human-in-the-loop review**
  - UI to approve/edit canonical claims, correct entities, and mark duplicates.
  - Periodic deduplication job to merge semantically identical facts across languages.
- **Versioning & auditability**
  - Maintain version history for each fact.
  - Store which model version produced which verdict for reproducibility.

---

## 4. NLP methods to remove conversational fluff and irrelevant text

- **Rule-based preprocessing**
  - Regex and heuristics to strip:
    - Greetings ("good morning", "namaste", "salaam", equivalents in Indian languages).
    - Courtesy phrases ("please share", "forward this", "subscribe", "like/comment/share").
    - User mentions, hashtags (optionally keep content words from hashtags), emojis, excessive punctuation.
- **Sentence segmentation and classification**
  - Split text into sentences/clauses using language-specific sentence boundary detection (Indic NLP library, spaCy models, or rule-based for resource-poor languages).
  - Train a **binary classifier** (claim vs non-claim) on sentences:
    - Models: small transformers (e.g., `xlm-roberta-base`, `indic-bert`) fine-tuned for claim detection.
    - Negative examples: jokes, opinions, questions without factual assertions, chit-chat.
- **Claim-worthiness scoring**
  - Multi-class classifier to rank sentences by check-worthiness (e.g., elections, public health, communal issues).
  - Use thresholds to drop low-scoring sentences.
- **Dependency & POS-based heuristics**
  - Use syntactic patterns to ensure factual assertions (presence of verbs, entities, numbers, dates).
  - Filter out isolated nouns or fragments (e.g., just a URL or hashtag cloud).
- **Summarization for long posts**
  - For long rants or threads, use extractive summarization based on claim-worthiness scores to reduce to a few key factual sentences before retrieval.
- **Template & keyword filters**
  - Maintain lists of generic phrases (e.g., "like and share", "I think", "in my opinion") in multiple languages to downweight/remove those segments.

---

## 5. Strategies for multilingual processing for Indian languages

- **Language & script detection**
  - Use fast language ID (fastText langid models) trained/fine-tuned for Indian languages.
  - Detect script (Devanagari, Latin, Bengali, Gurmukhi, etc.) via Unicode ranges.
- **Transliteration & normalization**
  - For code-mixed and Romanized Indian languages, use transliteration libraries or sequence models to map to native scripts (e.g., Hindi Latin → Devanagari).
  - Normalize spacing, digits (Devanagari numerals → ASCII), dates, and currency formats.
- **Shared multilingual encoders**
  - Use models like **IndicBERT, MuRIL, XLM-R** so that semantically similar claims in different languages are mapped to nearby vector space locations.
  - Fine-tune for retrieval and classification tasks (claim detection, stance classification) with multilingual datasets.
- **Language-specific components where needed**
  - Per-language stopword lists, punctuation rules, and slang dictionaries.
  - Optional per-language light stemmers or lemmatizers.
- **Code-mixing handling**
  - Identify mixed-language stretches at the token level; route to models fine-tuned on code-mixed data.
  - Use token-level language tags to help models disambiguate.
- **Multilingual fact DB**
  - Store canonical fact in primary language plus translations.
  - For retrieval, search across all languages using embeddings; but apply language preferences (e.g., prefer same-language evidence first, then cross-lingual).
- **Evaluation per language**
  - Track precision/recall by language and key dialect groups to avoid underperformance on low-resource languages.

---

## 6. Recommended tools, frameworks, and ML models

- **Programming & frameworks**
  - **Python 3.11+** as main language.
  - **FastAPI** for backend APIs.
  - **Kafka/Redpanda** for streaming; alternatively **RabbitMQ** for smaller-scale student deployments.
  - **Celery + Redis** or **Kafka consumer workers** for background processing.
  - **PostgreSQL** for relational data.
  - **Elasticsearch/OpenSearch** for full-text + BM25 retrieval.
  - **FAISS or Milvus** for vector similarity search.
- **NLP & ML**
  - **PyTorch** / **HuggingFace Transformers** for model training/inference.
  - **spaCy** for tokenization, NER (where models exist), custom pipelines; integrate with Indic NLP tools.
  - **Indic NLP Library** and similar projects for tokenization, transliteration, and script handling for Indian languages.
- **Candidate models** (for student-friendly setup)
  - Language-agnostic sentence embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
  - Indic-focused: `ai4bharat/indic-bert`, `google/muril-base-cased`, `xlm-roberta-base` fine-tuned.
  - Claim detection / stance classification: fine-tune above encoders on claim datasets (e.g., FakeNewsNet, ClaimBuster-like sources, Indian fact-check corpora where available).
- **Frontend**
  - **React + TypeScript** with **Vite** or **Next.js**.
  - UI: **Tailwind CSS** or **Material UI**.
  - Charts/metrics: **Recharts**, **Chart.js**, or **ECharts**.
- **DevOps & tooling**
  - **Docker** for containerization; **docker-compose** for local.
  - **GitHub Actions** or **GitLab CI** for CI/CD.
  - **Poetry** or **pip + requirements.txt** for Python deps.

---

## 7. Example project folder structure

- **Root**
  - `backend/`
    - `app/`
      - `main.py` (FastAPI entrypoint)
      - `api/`
        - `routes_posts.py`
        - `routes_claims.py`
        - `routes_facts.py`
        - `routes_admin.py`
      - `core/`
        - `config.py`
        - `logging.py`
      - `models/`
        - `post.py`
        - `claim.py`
        - `fact.py`
        - `fact_check_result.py`
        - `user.py`
      - `db/`
        - `session.py`
        - `schemas.sql` or Alembic migrations
      - `services/`
        - `ingest_service.py`
        - `preprocess_service.py`
        - `claim_extractor_service.py`
        - `retrieval_service.py`
        - `verification_service.py`
        - `cache_service.py`
      - `workers/`
        - `kafka_consumer_claims.py`
        - `kafka_consumer_results.py`
      - `tests/`
    - `Dockerfile`
    - `requirements.txt`
  - `ml/`
    - `notebooks/`
    - `data/` (small samples; big data in external storage)
    - `models/`
      - `claim_detector/`
      - `stance_classifier/`
      - `retriever_encoder/`
    - `training/`
      - `train_claim_detector.py`
      - `train_stance_classifier.py`
    - `inference/`
      - `claim_detector.py`
      - `fluff_filter.py`
      - `embedder.py`
      - `retrieval_pipeline.py`
  - `frontend/`
    - `src/`
      - `App.tsx`
      - `components/`
        - `LiveFeed.tsx`
        - `ClaimCard.tsx`
        - `FactCheckDetail.tsx`
        - `FiltersPanel.tsx`
        - `AnnotationForm.tsx`
      - `pages/`
        - `Dashboard.tsx`
        - `ClaimReview.tsx`
        - `FactDatabase.tsx`
      - `services/`
        - `api.ts` (REST client)
    - `public/`
    - `package.json`
  - `infra/`
    - `docker-compose.yml`
    - `k8s/` (optional manifests)
  - `docs/`
    - `architecture.md`
    - `api-spec.md`
    - `ml-design.md`
  - `README.md`

---

## 8. Team structure and work division

### 8.1 Member A – Backend Engineer

- **Responsibilities**
  - Design and implement the backend services, REST APIs, and message-driven workers.
  - Integrate with Kafka/Redis, PostgreSQL, Elasticsearch, FAISS/Milvus (through ML engineer’s modules).
  - Implement authentication/authorization for internal dashboard.
  - Expose endpoints for ingestion, fact-check results, fact DB management, and annotations.
  - Implement monitoring, logging, and basic rate limiting.
- **Key technologies**
  - Python, FastAPI, Pydantic.
  - Kafka (or Redis Streams/RabbitMQ for simpler setup).
  - PostgreSQL, SQLAlchemy/SQLModel.
  - Elasticsearch/OpenSearch client.
  - Redis for caching.
  - Docker, docker-compose.
- **Step-by-step tasks**
  1. **Core backend scaffold**
    - Initialize `backend/` FastAPI project with environment-based config.
    - Set up DB connection (PostgreSQL) and ORM models for posts, claims, facts, results, users.
  2. **API design & implementation**
    - Implement `GET /health`, `GET /stats` for monitoring.
    - Implement APIs:
      - `POST /ingest/post` (for testing/manual input).
      - `GET /posts/{id}` and `GET /claims/{id}`.
      - `GET /fact-check` with query by text or URL.
      - Admin endpoints for CRUD on facts and users.
  3. **Messaging and workers**
    - Implement Kafka consumer/producer wrappers.
    - Build workers to:
      - Read from `raw_posts` and call ML preprocessing/claim extraction (via ML modules).
      - Read claims from `claims` and call retrieval/verification ML modules.
      - Save results to DB and publish to `results` topic.
  4. **Caching and performance**
    - Integrate Redis caching for hot results and metadata.
    - Add simple request-level caching for the most common queries.
  5. **Security & auth**
    - Implement JWT-based auth for internal dashboard APIs.
  6. **Monitoring & logging**
    - Add structured logging middleware.
    - Add Prometheus metrics endpoints if time allows.
- **Files/modules Member A will create**
  - `backend/app/main.py`
  - `backend/app/core/config.py`, `logging.py`
  - `backend/app/models/{post,claim,fact,fact_check_result,user}.py`
  - `backend/app/api/{routes_posts,routes_claims,routes_facts,routes_admin}.py`
  - `backend/app/services/{ingest_service,preprocess_service,claim_extractor_service,retrieval_service,verification_service,cache_service}.py`
  - `backend/app/workers/{kafka_consumer_claims,kafka_consumer_results}.py`
  - `backend/requirements.txt`
- **Expected outputs**
  - Running backend service exposing REST APIs and Kafka workers.
  - Database schema for posts/claims/facts/results.
  - Internal documentation in `docs/api-spec.md` describing endpoints.
- **Integration with others**
  - Consumes ML modules from Member C as Python packages/modules (`ml/inference/`*).
  - Provides HTTP APIs for Member B’s frontend and annotation UI.
  - Exposes admin endpoints to let Member C update fact DB and trigger re-indexing.

---

### 8.2 Member B – Frontend Engineer

- **Responsibilities**
  - Build the web dashboard for monitoring streams, reviewing claims, and managing facts.
  - Implement UI components for browsing posts, viewing fact-check results, and annotating new facts.
  - Provide clear UX for multilingual content and filtering.
  - Integrate with backend APIs for live updates (polling or websockets if added later).
- **Key technologies**
  - React + TypeScript.
  - Tailwind CSS or Material UI.
  - Axios or Fetch for API calls.
  - Vite/Next.js for build/dev environment.
- **Step-by-step tasks**
  1. **Project setup**
    - Initialize React + TypeScript project in `frontend/`.
    - Configure routing (e.g., React Router).
  2. **Design system & layout**
    - Create core layout: navbar, sidebar, main content area.
    - Define color palette and typography suitable for data-heavy dashboards.
  3. **Dashboard views**
    - `Dashboard` page:
      - Live table/stream of latest claims with status, language, source.
      - Filters by language, verdict, confidence, topic.
    - `ClaimReview` page:
      - Detail view showing original post text, extracted claims, evidence facts, model verdict, and confidence.
      - Controls to approve/override, assign labels for training data.
    - `FactDatabase` page:
      - Search and browse verified facts.
      - Form to add/edit facts (including language, entities, time/geo scope).
  4. **API integration**
    - Implement `frontend/src/services/api.ts` to talk to backend endpoints (Member A).
    - Map responses to TypeScript types.
  5. **User auth UI**
    - Basic login form, token storage, and logout.
  6. **UX enhancements & validation**
    - Loading states, error handling, pagination.
    - Indicate model confidence visually (e.g., colored badges).
- **Files/modules Member B will create**
  - `frontend/src/App.tsx`
  - `frontend/src/pages/{Dashboard,ClaimReview,FactDatabase}.tsx`
  - `frontend/src/components/{LiveFeed,ClaimCard,FactCheckDetail,FiltersPanel,AnnotationForm,Navbar,Sidebar}.tsx`
  - `frontend/src/services/api.ts`
  - `frontend/package.json`, config files.
- **Expected outputs**
  - Working single-page dashboard that interacts with backend APIs.
  - Annotation interface for fact-checkers to label claims and add facts.
  - Basic styling and responsive layout.
- **Integration with others**
  - Uses backend REST APIs from Member A for data.
  - Sends annotation data (labeled claims, new facts) that Member C can use to improve models.
  - Displays language, confidence, and model outputs provided by Member C’s modules via backend.

---

### 8.3 Member C – ML/Data Engineer

- **Responsibilities**
  - Design and implement all NLP/ML components: preprocessing, language detection, fluff removal, claim extraction, retrieval, and verification.
  - Build and maintain the fact database’s indexing (text + vector) pipelines.
  - Prepare datasets, train/fine-tune models, and ship inference modules consumable by backend.
- **Key technologies**
  - Python, Jupyter notebooks.
  - PyTorch, HuggingFace Transformers.
  - spaCy, Indic NLP Library.
  - FAISS or Milvus for vector search.
  - Pandas, scikit-learn for data prep and basic baselines.
- **Step-by-step tasks**
  1. **Data collection & preprocessing**
    - Collect sample datasets from public Indian fact-checking sites and open datasets.
    - Build scripts to parse, clean, and normalize text (multi-language support).
  2. **Language/script detection module**
    - Train/finetune or adopt existing lang ID model for Indic languages.
    - Implement `detect_language(text)` and `detect_script(text)` functions.
  3. **Fluff filtering & claim detection**
    - Implement heuristic cleaner (remove URLs, emojis, boilerplate phrases in multiple languages).
    - Create labeled dataset for claim vs non-claim at sentence level.
    - Fine-tune a small multilingual transformer as claim detector; export inference code.
  4. **Claim extraction pipeline**
    - Combine sentence splitter + claim detector to output structured `Claim` objects.
  5. **Retrieval pipeline**
    - Choose embedding model (e.g., XLM-R or IndicBERT-based sentence transformer).
    - Build scripts to index verified facts into FAISS/Milvus + Elasticsearch.
    - Implement hybrid retrieval: BM25 + vector similarity + re-ranking.
  6. **Verification / stance classification**
    - Train model to classify [claim, fact] pairs as Supported / Refuted / NotEnoughEvidence.
    - Implement scoring and threshold tuning.
  7. **Packaging & integration**
    - Wrap all inference code into lightweight modules callable from backend:
      - `ml/inference/claim_detector.py`
      - `ml/inference/fluff_filter.py`
      - `ml/inference/embedder.py`
      - `ml/inference/retrieval_pipeline.py`
      - `ml/inference/verifier.py`
    - Provide clear function signatures and configuration options.
  8. **Evaluation & monitoring**
    - Define evaluation metrics per language (precision/recall/F1 for claim detection, retrieval quality, stance classification).
    - Provide evaluation scripts and reports.
- **Files/modules Member C will create**
  - `ml/notebooks/*.ipynb` for experiments.
  - `ml/data_preparation/*.py` for scraping and preprocessing.
  - `ml/training/{train_claim_detector.py,train_stance_classifier.py}`.
  - `ml/inference/{claim_detector.py,fluff_filter.py,embedder.py,retrieval_pipeline.py,verifier.py}`.
  - `ml/models/` storing checkpoints (small for repo, larger in external storage).
  - `docs/ml-design.md` documenting model choices and pipelines.
- **Expected outputs**
  - Trained or fine-tuned models (or configured open models) for:
    - Language detection (or integrated third-party model).
    - Claim detection and claim-worthiness.
    - Retrieval embeddings and hybrid retrieval.
    - Stance classification / verification.
  - Reusable Python modules for inference, ready to be imported by backend services.
- **Integration with others**
  - Provides Python inference APIs and config files to Member A for integration in workers.
  - Supplies evaluation metrics and guidelines so Member B can surface model performance in the UI.
  - Consumes annotation data (labels, new facts) from frontend/ backend to retrain/refine models.

---

## 9. Suggested initial milestones (for the team)

- **Milestone 1 – Skeleton system (Week 1–2)**
  - Backend: health APIs, DB schema, minimal ingest + result endpoints.
  - Frontend: static dashboard mock with dummy data.
  - ML: basic preprocessing, off-the-shelf multilingual embeddings, trivial keyword-based retrieval.
- **Milestone 2 – End-to-end MVP (Week 3–4)**
  - Claim extraction and simple verification in one or two languages (e.g., Hindi + English).
  - Kafka-based pipeline wired up with at least one ingestion source.
  - Dashboard shows live claims and verdicts.
- **Milestone 3 – Multilingual & optimization (Week 5–6)**
  - Add more Indian languages and better claim detection.
  - Implement fluff removal, hybrid retrieval, and time/geo-aware filters.
  - Improve UI for annotation and incorporate feedback loop into ML training.

