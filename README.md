# Vernacular Fact-Checker

## 1. Project Overview

### Purpose

Vernacular Fact-Checker is a full-stack multilingual fact-checking system designed to process short-form social media style text, extract factual claims, retrieve relevant evidence from a verified knowledge base, and return a verification verdict with confidence and supporting sources.

### Problem It Solves

The project targets misinformation and unverifiable viral content that appears in English, Hindi, and mixed-language Indian social content. In practice, these posts often include conversational fluff such as emojis, hashtags, forwarding requests, and informal phrasing. The system removes that noise, identifies claim-like text, and performs cross-lingual fact verification without requiring manual translation.

### Current Scope

The repository currently implements:

- A FastAPI backend for ingestion, claim extraction, and claim verification.
- An ML layer for text cleaning, claim extraction, embedding-based retrieval, and multilingual NLI-based verification.
- A React + Vite frontend for entering a claim and viewing the returned verdict, confidence, and sources.
- A SQLModel-based persistence layer for posts, extracted claims, and stored verdicts.
- A bilingual fact base and test samples for English and Hindi verification.

The repository also includes dependencies for Redis and Kafka, but those are not currently wired into the active request flow.

## 2. Requirements

### 2.1 System Requirements

| Area | Requirement | Notes |
| --- | --- | --- |
| Operating system | Windows, Linux, or macOS | Current workspace usage is on Windows. |
| Python | Python 3.12 recommended | Current configured environment is Python 3.12.10. |
| Node.js | Modern Node.js runtime recommended, preferably Node 20+ | The frontend does not pin an engine version, but Vite 7 and current TypeScript tooling expect a modern Node runtime. |
| Package managers | `pip`, `npm` | Used for backend/ML and frontend respectively. |
| Database | SQLite for local development, PostgreSQL-compatible setup possible | The backend uses `SQLModel` and reads `DATABASE_URL` from environment. |
| ML runtime | CPU supported, CUDA optional | The ML config uses GPU automatically when `torch.cuda.is_available()` is true. |
| Internet access | Required for first model download unless models are already cached locally | Hugging Face models are loaded lazily on first use. |

### 2.2 Backend Dependencies

Defined in [backend/requirements.txt](project-integration/vernacular-fact-checker/backend/requirements.txt):

| Package | Role |
| --- | --- |
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Validation |
| `pydantic-settings` | Environment-based settings |
| `python-dotenv` | `.env` support |
| `sqlalchemy` | ORM foundation used under SQLModel |
| `sqlmodel` | Models and session management |
| `psycopg2-binary` | PostgreSQL driver |
| `redis` | Optional future caching/integration |
| `kafka-python` | Optional future messaging/integration |

### 2.3 ML Dependencies

Defined in [ml/requirements-ml.txt](project-integration/vernacular-fact-checker/ml/requirements-ml.txt):

| Package | Role |
| --- | --- |
| `numpy` | Numeric operations |
| `pandas` | Dataset loading and preprocessing |
| `scikit-learn` | Claim detector training and cosine similarity utilities |
| `joblib` | Persisting claim detector artifacts |
| `torch` | Model execution and fine-tuning |
| `transformers` | Multilingual verifier model loading and inference |
| `sentence-transformers` | LaBSE embeddings |
| `huggingface-hub` | Model download and cache management |
| `sentencepiece` | Tokenization support |
| `protobuf` | Model/runtime dependency |
| `py3langid` | Language identification |
| `indic-nlp-library` | Indic-language text processing support |
| `regex` | Emoji and Unicode-aware cleanup |
| `unidecode` | ASCII normalization for Latin-only text |
| `pytest` | Testing |

### 2.4 Frontend Dependencies

Defined in [frontend/factcheck-frontend/package.json](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/package.json):

| Package | Role |
| --- | --- |
| `react` | UI library |
| `react-dom` | DOM rendering |
| `vite` | Dev server and build tool |
| `typescript` | Static typing |
| `@vitejs/plugin-react` | React support for Vite |
| `eslint` and related plugins | Linting |
| `tailwindcss`, `postcss`, `autoprefixer` | Styling toolchain available in the project |

## 3. Architecture

### 3.1 High-Level Architecture

The implemented system has four main layers:

1. Frontend UI
2. FastAPI backend
3. ML inference and training modules
4. Database and fact knowledge base

### 3.2 High-Level System Diagram Explanation

```text
User
  -> React frontend (Dashboard, SearchBar, ClaimCard)
  -> FastAPI backend routes
  -> Backend services
  -> ML inference modules
  -> Verified facts KB + SQL database
  -> Verification result returned to frontend
```

### 3.3 Layer Interaction

| Layer | Responsibility | Interacts With |
| --- | --- | --- |
| Frontend | Accepts user query, sends verify request, renders result | FastAPI backend |
| Backend API | Exposes routes, validates payloads, persists data, invokes ML services | Frontend, SQL database, ML layer |
| ML layer | Cleans text, extracts claims, retrieves evidence, verifies claims | Backend, fact KB, local model artifacts |
| Database | Stores posts, claims, and verdicts | Backend |
| Verified fact base | Provides known facts for retrieval and verification | ML retrieval pipeline |

### 3.4 Data Flow

#### Ingestion to Display Flow

1. A post can be created through the backend ingestion route.
2. Claims are extracted from the stored post text.
3. A claim is passed to the verification logic.
4. The ML pipeline cleans the text and retrieves top candidate facts using multilingual embeddings.
5. The verifier compares the claim against retrieved facts using multilingual NLI.
6. The backend returns a verdict, confidence score, and source list.
7. The frontend displays the claim card with verdict and evidence.

#### Direct Verify Flow

1. User enters a claim in the frontend search bar.
2. Frontend sends `POST /verify` with raw text.
3. Backend calls `verify_claim_logic`.
4. ML pipeline runs fluff filtering, retrieval, and NLI verification.
5. JSON response is returned and rendered in the dashboard.

## 4. Backend Details

### 4.1 Frameworks and Core Backend Stack

The backend uses:

- FastAPI for route handling and OpenAPI generation.
- SQLModel for schema definition and DB interaction.
- SQLAlchemy engine through SQLModel.
- Pydantic Settings for environment-driven configuration.
- Uvicorn for application serving.

Core backend entrypoint: [backend/app/main.py](project-integration/vernacular-fact-checker/backend/app/main.py)

### 4.2 Backend Configuration

Configuration is defined in [backend/app/core/config.py](project-integration/vernacular-fact-checker/backend/app/core/config.py).

| Setting | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | SQLModel database connection string |
| `REDIS_URL` | No | Reserved for future cache integration |
| `KAFKA_BOOTSTRAP_SERVERS` | No | Reserved for future stream integration |

### 4.3 Database Models

#### Post

Defined in [backend/app/models/post.py](project-integration/vernacular-fact-checker/backend/app/models/post.py)

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Primary key |
| `source` | str | Source platform or origin label |
| `text` | str | Raw post content |
| `language` | str or null | Language tag |
| `author` | str or null | Optional author |
| `url` | str or null | Optional source URL |
| `created_at` | datetime | Timestamp |

#### Claim

Defined in [backend/app/models/claim.py](project-integration/vernacular-fact-checker/backend/app/models/claim.py)

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Primary key |
| `post_id` | int | Source post identifier |
| `claim_text` | str | Extracted claim text |
| `language` | str or null | Language inherited from post |
| `created_at` | datetime | Timestamp |

#### Verdict

Defined in [backend/app/models/verdict.py](project-integration/vernacular-fact-checker/backend/app/models/verdict.py)

| Field | Type | Description |
| --- | --- | --- |
| `id` | int | Primary key |
| `claim_id` | int | Linked claim |
| `verdict` | str | Verification result |
| `confidence` | float | Confidence score |
| `evidence` | str | JSON-serialized source evidence |
| `created_at` | datetime | Timestamp |

### 4.4 API Routes

#### Core Utility Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Root status message |
| `GET` | `/health` | Health check |

#### Post and Claim Routes

| Method | Path | File | Purpose |
| --- | --- | --- | --- |
| `POST` | `/ingest/post` | [backend/app/api/routes_posts.py](project-integration/vernacular-fact-checker/backend/app/api/routes_posts.py) | Persist a post to the database |
| `POST` | `/extract/claims/{post_id}` | [backend/app/api/routes_claims.py](project-integration/vernacular-fact-checker/backend/app/api/routes_claims.py) | Split a stored post into claim records |

#### Verification Routes

| Method | Path | File | Purpose |
| --- | --- | --- | --- |
| `POST` | `/verify` | [backend/app/api/routes_verification.py](project-integration/vernacular-fact-checker/backend/app/api/routes_verification.py) | Verify raw text without DB writes |
| `POST` | `/verify/claim/{claim_id}` | [backend/app/api/routes_verification.py](project-integration/vernacular-fact-checker/backend/app/api/routes_verification.py) | Verify a stored claim and persist the verdict |

Note: there is no generic `/ingest` route in the current implementation. The active route is `/ingest/post`.

### 4.5 Backend Services and Responsibilities

#### Verification Service

File: [backend/app/services/verification_service.py](project-integration/vernacular-fact-checker/backend/app/services/verification_service.py)

Responsibilities:

- Lazily loads ML functions with `lru_cache`.
- Cleans and normalizes claim text.
- Retrieves candidate facts from the fact knowledge base.
- Calls the ML verifier.
- Normalizes source payloads for backend responses.
- Raises structured service-level errors when ML loading or inference fails.

#### Ingest Service

File: [backend/app/services/ingest_service.py](project-integration/vernacular-fact-checker/backend/app/services/ingest_service.py)

Responsibilities:

- Cleans post text.
- Calls the ML claim detector.
- Stores extracted claims in the SQL database.

Current status:

- This service exists and is more ML-aware than the simple route implementation.
- The current `/extract/claims/{post_id}` route still uses a direct sentence split on periods instead of calling this service.

#### Similarity Service

File: [backend/app/services/similarity_service.py](project-integration/vernacular-fact-checker/backend/app/services/similarity_service.py)

Responsibilities:

- Generates embeddings for a single text.
- Computes cosine similarity between claim and evidence text.

Current status:

- The service is available as a utility module.
- It is not currently exposed through an API route.

## 5. ML Details

### 5.1 ML Pipeline Overview

The ML layer is divided into:

- `ml/inference` for runtime prediction.
- `ml/pipeline` for supporting pipeline utilities.
- `ml/training` for model training scripts.
- `ml/tests` for pipeline tests.
- `ml/data` for the verified fact base, sample posts, and test fixtures.

### 5.2 Runtime Verification Pipeline

The active inference path is centered around:

- [ml/inference/fluff_filter.py](project-integration/vernacular-fact-checker/ml/inference/fluff_filter.py)
- [ml/inference/claim_detector.py](project-integration/vernacular-fact-checker/ml/inference/claim_detector.py)
- [ml/inference/embedder.py](project-integration/vernacular-fact-checker/ml/inference/embedder.py)
- [ml/inference/retrieval_pipeline.py](project-integration/vernacular-fact-checker/ml/inference/retrieval_pipeline.py)
- [ml/inference/verifier.py](project-integration/vernacular-fact-checker/ml/inference/verifier.py)
- [ml/inference/pipeline.py](project-integration/vernacular-fact-checker/ml/inference/pipeline.py)

#### Step 1: Fluff Filtering and Normalization

`clean_text` removes:

- URLs
- social mentions
- hashtags
- emojis
- repeated punctuation
- viral-forwarding patterns such as `please share`, `like and share`, and Hindi equivalents

This is implemented in [ml/inference/fluff_filter.py](project-integration/vernacular-fact-checker/ml/inference/fluff_filter.py).

Important implementation note:

- Fluff is removed case-insensitively without lowercasing the full string, which preserves model-sensitive casing behavior for the verifier.

#### Step 2: Claim Detection

`extract_claims` works in two modes:

- Preferred mode: load a trained TF-IDF + Logistic Regression classifier from disk if artifacts exist.
- Fallback mode: use heuristic sentence filtering based on length, numbers, and verb hints in English and Hindi.

Implemented in [ml/inference/claim_detector.py](project-integration/vernacular-fact-checker/ml/inference/claim_detector.py).

#### Step 3: Cross-Lingual Retrieval

The retrieval pipeline:

- Loads fact records from [ml/data/verified_facts.jsonl](project-integration/vernacular-fact-checker/ml/data/verified_facts.jsonl).
- Embeds fact statements with LaBSE.
- Caches fact embeddings to disk in `ml/cache/retrieval`.
- Uses cosine similarity to rank top-k matching facts.
- Applies language-aware similarity thresholds and fallbacks.

Implemented in [ml/inference/retrieval_pipeline.py](project-integration/vernacular-fact-checker/ml/inference/retrieval_pipeline.py).

Current fact base size:

- 392 fact records in `verified_facts.jsonl`

#### Step 4: Verification via Multilingual NLI

The verifier:

- Builds premise-hypothesis pairs from retrieved facts and the cleaned claim.
- Tokenizes both premise and hypothesis in lowercase.
- Runs multilingual NLI classification.
- Chooses `Supported`, `Refuted`, or `NotEnoughEvidence` using both NLI probabilities and retrieval thresholds.

Implemented in [ml/inference/verifier.py](project-integration/vernacular-fact-checker/ml/inference/verifier.py).

### 5.3 Models Used

Configured in [ml/config.py](project-integration/vernacular-fact-checker/ml/config.py).

| Component | Model | Purpose |
| --- | --- | --- |
| Embeddings | `sentence-transformers/LaBSE` | Cross-lingual sentence embeddings for retrieval |
| Verifier | `joeddav/xlm-roberta-large-xnli` | Multilingual natural language inference |
| Claim detector | Local TF-IDF + Logistic Regression artifacts when trained | Sentence-level claim classification |

### 5.4 ML Thresholds and Runtime Configuration

Current verification-related thresholds:

| Setting | Value | Purpose |
| --- | --- | --- |
| `TOP_K_FACTS` | `5` | Number of retrieved facts |
| `MIN_SIMILARITY` | `0.40` | Base English retrieval threshold |
| `MIN_SIMILARITY_HI` | `0.35` | Hindi retrieval threshold |
| `MIN_SIMILARITY_FALLBACK` | `0.20` | Low-confidence fallback threshold |
| `NLI_DECISION_THRESHOLD` | `0.45` | Strong entailment/contradiction threshold |
| `NLI_WEAK_SIGNAL_THRESHOLD` | `0.38` | Weak but acceptable NLI threshold |
| `RETRIEVAL_SUPPORT_THRESHOLD` | `0.50` | Retrieval support required for verdict |
| `RETRIEVAL_STRONG_THRESHOLD` | `0.60` | Strong retrieval threshold for weak NLI signals |

### 5.5 Training Datasets and Methodology

#### Claim Detector Training

Implemented in [ml/training/train_claim_detector.py](project-integration/vernacular-fact-checker/ml/training/train_claim_detector.py).

Methodology:

- Input can be CSV or JSONL.
- Supports flexible text columns such as `text`, `claim_text`, or `sentence`.
- Supports label columns `label` or `is_claim`.
- Uses `TfidfVectorizer` with unigram and bigram features.
- Trains a `LogisticRegression` classifier with balanced class weights.
- Saves model and vectorizer using `joblib`.

Auto-generation option:

- The script can generate positive claim examples from the verified fact base.
- It creates negative non-claim examples from social or conversational templates.

#### Verifier Training

Implemented in [ml/training/train_verifier.py](project-integration/vernacular-fact-checker/ml/training/train_verifier.py).

Methodology:

- Accepts labelled premise-hypothesis data.
- Maps labels into three NLI classes: contradiction, neutral, and entailment.
- Fine-tunes `joeddav/xlm-roberta-large-xnli` for three-way classification.
- Saves tokenizer and model artifacts to the configured verifier model directory.

Auto-generation option:

- Builds entailment pairs from same-topic bilingual facts using `topic_id`.
- Builds neutral pairs from random cross-topic facts.
- Builds contradiction proxies by negating unrelated claims.

### 5.6 Cross-Lingual Support

Cross-lingual support is a core feature of the system.

How it works:

- LaBSE maps semantically equivalent English and Hindi claims into nearby embedding vectors.
- The verified fact base contains bilingual records sharing a common `topic_id`.
- A Hindi user claim can retrieve English facts and vice versa.
- The NLI verifier is multilingual and can score claim-fact alignment without a translation stage.

Example:

- English fact: `The Earth is the third planet from the Sun.`
- Hindi fact: `पृथ्वी सूर्य से तीसरा ग्रह है।`
- Both can support the same claim because retrieval is embedding-based rather than exact-string-only.

### 5.7 ML Tests and Evaluation Assets

Repository testing assets include:

- [ml/tests/test_pipeline.py](project-integration/vernacular-fact-checker/ml/tests/test_pipeline.py)
- [ml/tests/test_retrieval.py](project-integration/vernacular-fact-checker/ml/tests/test_retrieval.py)
- [ml/tests/test_text_cleaning.py](project-integration/vernacular-fact-checker/ml/tests/test_text_cleaning.py)
- [ml/tests/test_language_id.py](project-integration/vernacular-fact-checker/ml/tests/test_language_id.py)
- [ml/data/_smoke_test.py](project-integration/vernacular-fact-checker/ml/data/_smoke_test.py)
- [ml/data/test_all_samples.py](project-integration/vernacular-fact-checker/ml/data/test_all_samples.py)
- [ml/data/verify_test_samples.jsonl](project-integration/vernacular-fact-checker/ml/data/verify_test_samples.jsonl)

Current sample test asset size:

- 25 bilingual verification samples

## 6. Frontend Details

### 6.1 Frontend Stack

The frontend uses:

- React
- TypeScript
- Vite
- CSS-based styling with Tailwind-related tooling available in the dependency graph

Main files:

- [frontend/factcheck-frontend/src/main.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/main.tsx)
- [frontend/factcheck-frontend/src/pages/Dashboard.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/pages/Dashboard.tsx)
- [frontend/factcheck-frontend/src/components/SearchBar.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/SearchBar.tsx)
- [frontend/factcheck-frontend/src/components/ClaimCard.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/ClaimCard.tsx)
- [frontend/factcheck-frontend/src/components/Header.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/Header.tsx)

### 6.2 Component Responsibilities

| Component | File | Responsibility |
| --- | --- | --- |
| `Dashboard` | [frontend/factcheck-frontend/src/pages/Dashboard.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/pages/Dashboard.tsx) | Holds verification result state, loading state, and error state |
| `SearchBar` | [frontend/factcheck-frontend/src/components/SearchBar.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/SearchBar.tsx) | Accepts user input and triggers verification requests |
| `ClaimCard` | [frontend/factcheck-frontend/src/components/ClaimCard.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/ClaimCard.tsx) | Displays claim text, verdict, confidence, and sources |
| `Header` | [frontend/factcheck-frontend/src/components/Header.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/Header.tsx) | Renders the page header |

### 6.3 Frontend to Backend Interaction

The frontend calls the backend by sending a `POST /verify` request with a JSON payload of the form:

```json
{
  "text": "The Earth is the third planet from the Sun."
}
```

This request is sent from [frontend/factcheck-frontend/src/components/SearchBar.tsx](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/src/components/SearchBar.tsx).

### 6.4 Vite Proxy Configuration

Defined in [frontend/factcheck-frontend/vite.config.ts](project-integration/vernacular-fact-checker/frontend/factcheck-frontend/vite.config.ts).

Current proxy behavior:

- Requests to `/verify` are proxied to `http://127.0.0.1:8000`

This keeps the frontend code simple during local development.

### 6.5 UI Runtime Behavior

Current UI behavior includes:

- Loading state while verification is in progress.
- Error display if backend verification fails.
- A client-side timeout for long verification requests so the UI does not stay indefinitely in a loading state.

## 7. Deployment and Run Instructions

### 7.1 Repository Setup

From the repository root:

```powershell
cd project-integration/vernacular-fact-checker
```

### 7.2 Python Environment Setup

Example local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
pip install -r ml/requirements-ml.txt
```

Note:

- The current workspace has been running with an already existing Python virtual environment.
- Any equivalent activated Python environment works as long as the backend and ML requirements are installed.

### 7.3 Backend Run Instructions

Set the required database environment variable and run Uvicorn from the backend directory.

#### Local SQLite Example

```powershell
cd backend
$env:DATABASE_URL = "sqlite:///./app.db"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Useful backend URLs:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

Important notes:

- `DATABASE_URL` is required.
- On first verification request, model loading can take longer because Hugging Face artifacts may be loaded into cache.
- The backend inserts the repository root into `sys.path` so the `ml` package is importable from the backend app.

### 7.4 Frontend Run Instructions

From the frontend directory:

```powershell
cd frontend/factcheck-frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend URL:

- `http://127.0.0.1:5173`

### 7.5 Connecting Frontend to Backend

The local development setup assumes:

- Frontend on `127.0.0.1:5173`
- Backend on `127.0.0.1:8000`

The Vite proxy forwards `/verify` automatically. If additional backend routes are added to the frontend later, the proxy configuration should be extended accordingly.

## 8. Sample Input and Output

### 8.1 Example Inputs

Examples are drawn from [ml/data/verify_test_samples.jsonl](project-integration/vernacular-fact-checker/ml/data/verify_test_samples.jsonl).

| Language | Example Input | Expected Verdict |
| --- | --- | --- |
| English | `🔥 Like and share! The Earth is the third planet from the Sun. #science` | Supported |
| Hindi | `🌍 पृथ्वी सूर्य से तीसरा ग्रह है। #science` | Supported |
| English misinformation | `5G towers are the real cause of COVID-19. Scientists have confirmed this. Please share this vital information with your family!` | Refuted |
| Hindi misinformation | `खुशखबरी! 500 और 1000 के पुराने नोट जनवरी 2024 से वापस चालू कर दिए गए हैं। RBI ने इसकी पुष्टि की है। 😍😍 ज़्यादा से ज़्यादा शेयर करें!` | Refuted |
| Speculative claim | `The government is planning to launch a new free electricity scheme for all rural households by December 2026.` | NotEnoughEvidence |

### 8.2 Example Verification Response

Representative response shape for `POST /verify`:

```json
{
  "claim": "The Earth is the third planet from the Sun.",
  "verdict": "Supported",
  "confidence": 0.9995,
  "sources": [
    {
      "id": "fact6",
      "claim": "The Earth is the third planet from the Sun.",
      "language": "en",
      "score": 1.0
    },
    {
      "id": "fact7",
      "claim": "पृथ्वी सूर्य से तीसरा ग्रह है।",
      "language": "hi",
      "score": 0.9163
    }
  ]
}
```

### 8.3 Example Route Usage

#### Verify Raw Text

```http
POST /verify
Content-Type: application/json

{
  "text": "The Earth is the third planet from the Sun."
}
```

#### Ingest a Post

```http
POST /ingest/post
Content-Type: application/json

{
  "source": "whatsapp",
  "text": "भारत को 15 अगस्त 1947 को स्वतंत्रता मिली।",
  "language": "hi",
  "author": "sample-user",
  "url": null
}
```

## 9. Additional Notes

### 9.1 Current Optimizations and Caching

The project already includes several practical optimizations:

- Lazy loading of ML components with `lru_cache` to avoid repeated model initialization.
- On-disk caching of fact embeddings and fact fingerprints in `ml/cache/retrieval`.
- Automatic index invalidation if the verified fact file changes.
- Retrieval fallbacks for low-match or cross-lingual edge cases.
- Fluff filtering before claim extraction and retrieval.
- Frontend timeout handling for slow cold-start verification requests.

### 9.2 Known Implementation Characteristics

- The current claim extraction route uses simple period-based splitting, while a more advanced ML-backed ingest service also exists in the codebase.
- The frontend currently proxies only `/verify`; other backend routes are not yet integrated into the UI.
- Redis and Kafka are present as dependencies and configuration hooks, but they are not yet active parts of the running application.

### 9.3 Future Improvements

- Route post ingestion and claim extraction through the ML-based ingest service by default.
- Expose additional backend APIs for claim review, similarity search, and fact management.
- Expand the frontend beyond single-claim verification into a richer analyst dashboard.
- Add stronger persistence and indexing support for larger fact bases.
- Introduce explicit vector-store support if the knowledge base grows beyond the current file-based retrieval flow.
- Add authentication, moderation workflows, and reviewer feedback loops.
- Extend multilingual coverage beyond English and Hindi to additional Indian languages.

### 9.4 Scaling Considerations

For larger-scale deployment:

- Move from local SQLite to PostgreSQL.
- Place the fact retrieval index behind a dedicated vector-search system if dataset size increases significantly.
- Use background workers for ingestion and batch verification.
- Add Redis-based caching for frequent verify requests.
- Preload models or run dedicated model-serving processes to avoid cold-start latency.

## 10. Repository Summary

Vernacular Fact-Checker is currently a working full-stack prototype with:

- A FastAPI backend
- A multilingual ML verification pipeline
- A React frontend
- A bilingual verified fact base
- Local development support for end-to-end claim verification

It is suitable both as a practical demo application and as a foundation for a larger multilingual misinformation analysis platform.