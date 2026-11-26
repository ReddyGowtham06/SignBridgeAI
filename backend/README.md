# SIGNBRIDGE AI — Production Backend Foundation

Welcome to the SIGNBRIDGE AI backend application service. This backend powers user management, translation session history, the Indian Sign Language (ISL) Learning Studio curriculum, and server-side M1 Champion AI sign classification.

---

## 1. System Architecture

The backend follows a clean, modular, layered architecture:

```
backend/
├── app/
│   ├── main.py                  # FastAPI application entrypoint & middleware configuration
│   ├── core/                    # Security, configuration, and request dependencies
│   │   ├── config.py            # Environment-driven settings (pydantic-settings)
│   │   ├── security.py          # Argon2id password hashing and JWT token handling
│   │   └── dependencies.py      # get_db, get_current_user auth validation
│   ├── database/                # SQLAlchemy session, base model, and connection diagnostics
│   │   ├── base.py              # Declarative Base and TimestampMixin
│   │   └── session.py           # Engine & SessionLocal with PostgreSQL / SQLite support
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── user.py              # Users table
│   │   ├── history.py           # Translation history table (isolated per user)
│   │   └── learning.py          # Learning lessons & progress tables
│   ├── schemas/                 # Pydantic validation schemas
│   │   ├── auth.py              # Register, login, refresh tokens
│   │   ├── user.py              # User profile responses
│   │   ├── history.py           # Translation history payloads
│   │   ├── learning.py          # Lessons & progress tracking
│   │   └── inference.py         # 758-D feature vector request & gating response
│   ├── services/                # Business logic layer
│   │   ├── auth_service.py      # Registration, authentication, token refresh
│   │   ├── history_service.py   # Translation history CRUD with user isolation
│   │   ├── learning_service.py  # ISL curriculum management & score tracking
│   │   └── inference_service.py # M1 Champion linear model & Adaptive Motion Gate
│   ├── api/                     # REST API routers
│   │   ├── router.py            # Main API router mounting all modules under /api
│   │   └── endpoints/           # Endpoint controllers
│   └── middleware/              # IP-based rate limiting & error handlers
├── migrations/                  # Alembic database migrations
│   ├── env.py
│   └── versions/
│       └── 0001_initial_schema.py # Initial production schema
├── tests/                       # Comprehensive Pytest test suite (25 tests)
├── .env.example                 # Environment variables template
├── requirements.txt             # Production dependency manifest
└── README.md                    # Setup and operational documentation
```

---

## 2. Database Schema (PostgreSQL)

The primary database is PostgreSQL (using `psycopg` 3).

### `users`
- `id` (VARCHAR(36), PK): UUID4 primary key.
- `name` (VARCHAR(100)): User display name.
- `email` (VARCHAR(255), UNIQUE, INDEX): Login email.
- `hashed_password` (VARCHAR(255)): Argon2id password hash (`$argon2id$...`).
- `status` (VARCHAR(20)): Account status (`active`, `suspended`).
- `created_at` / `updated_at` (TIMESTAMP WITH TIME ZONE): UTC timestamps.

### `translation_history`
- `id` (VARCHAR(36), PK): UUID4 primary key.
- `user_id` (VARCHAR(36), FK -> users.id ON DELETE CASCADE): User ownership.
- `source_language` (VARCHAR(20)): Source language code (default `'ISL'`).
- `translation_direction` (VARCHAR(30)): Translation mode (`sign_to_text`, `text_to_sign`).
- `input_text` (TEXT, nullable): Original text or note.
- `recognized_output` (TEXT): Recognized sign gloss / translation.
- `confidence` (FLOAT, nullable): Top prediction confidence score.
- `recognition_path` (VARCHAR(30), nullable): Recognition routing path (`dynamic_path`, `static_path`).
- `created_at` (TIMESTAMP WITH TIME ZONE): UTC event timestamp.
- *Strict Privacy*: No raw webcam frames, video files, or biometric images are ever stored.

### `learning_lessons`
- `id` (VARCHAR(50), PK): Semantic lesson slug (e.g., `'lesson-01-greetings'`).
- `title` (VARCHAR(150)): Lesson display title.
- `description` (TEXT): Overview and learning outcomes.
- `difficulty` (VARCHAR(20)): Level (`beginner`, `intermediate`, `advanced`).
- `category` (VARCHAR(50)): Curriculum category (`Greetings`, `Foundations`, `Anatomy`, `Emergency`).
- `content_metadata` (JSON): Vocabulary list, recommended gestures, estimated duration.
- `created_at` (TIMESTAMP WITH TIME ZONE): Creation timestamp.

### `learning_progress`
- `id` (VARCHAR(36), PK): Record UUID.
- `user_id` (VARCHAR(36), FK -> users.id ON DELETE CASCADE).
- `lesson_id` (VARCHAR(50), FK -> learning_lessons.id ON DELETE CASCADE).
- `completion_status` (VARCHAR(20)): Progress state (`not_started`, `in_progress`, `completed`).
- `score` (FLOAT): Performance score (0.0 to 100.0).
- `last_activity` (TIMESTAMP WITH TIME ZONE): Activity timestamp.
- `UNIQUE(user_id, lesson_id)` constraint ensures one active record per lesson per user.

---

## 3. API Endpoints

All endpoints are prefixed with `/api`. Interactive documentation is available at `/api/docs` (Swagger UI) and `/api/redoc`.

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | No | Server status & documentation links |
| `GET` | `/api/health` | No | System health, database connection, and AI model status |
| `POST` | `/api/auth/register` | No (Rate Limited) | Register new user account |
| `POST` | `/api/auth/login` | No (Rate Limited) | Authenticate user & return JWT token pair |
| `GET` | `/api/auth/me` | Yes (Bearer) | Get authenticated user profile |
| `POST` | `/api/auth/refresh` | No | Refresh access token using refresh token |
| `POST` | `/api/auth/logout` | Yes (Bearer) | Invalidate client session |
| `GET` | `/api/users/me` | Yes (Bearer) | Retrieve current user profile |
| `GET` | `/api/history` | Yes (Bearer) | List user's translation history (paginated) |
| `POST` | `/api/history` | Yes (Bearer) | Record sign translation result |
| `DELETE` | `/api/history/{id}` | Yes (Bearer) | Delete translation record (ownership enforced) |
| `GET` | `/api/learning/lessons` | No | List curriculum lessons |
| `GET` | `/api/learning/progress` | Yes (Bearer) | Get user curriculum progress summary & scores |
| `POST` | `/api/learning/progress` | Yes (Bearer) | Record lesson completion & score |
| `POST` | `/api/inference/classify` | No | Classify 758-D feature vector with Adaptive Motion Gate |
| `GET` | `/api/inference/model-info` | No | Inspect loaded M1 model metadata & SHA-256 verification |

---

## 4. AI Inference Service (M1 Champion & Adaptive Motion Gate)

The inference service provides server-side parity with the frontend browser classifier:
- **Model**: M1 Softmax Logistic Regression (`frontend/public/models/sign_classifier.json`).
- **Feature Vector**: Strictly 758 normalized MediaPipe landmarks + kinematic features.
- **SHA-256 Verification**: Checked on startup against approved hash `c8068a3fedd85645d736d16ebde450bd442970784adba40fdacb8970a4535274`.
- **Path A (Dynamic)**: Motion energy $v \ge 0.05$, confidence threshold $0.45$.
- **Path B (Static)**: Motion energy $v < 0.05$, restricted to approved static vocabulary (`21.good`, `34.stop`, `45.hand`), gated by:
  1. Active Hand Presence $\ge 0.50$
  2. Spatial Pose Stability $\le 0.055$ (Anti-Jitter)
  3. Y-Centroid $\le 0.75$ (Anti-Resting Hands on desk/lap)
  4. Vocabulary Membership
  5. Confidence $\ge 0.50$ and Margin $\ge 0.10$

---

## 5. Local Setup & Execution (Windows PowerShell)

### Prerequisites
- Python 3.12+ (Python 3.14 tested and verified)
- PostgreSQL 15+ (optional for local testing; SQLite in-memory fallback enabled for testing)

### Step 1: Clone and Navigate
```powershell
cd D:\SignBridgeAI\backend
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```powershell
Copy-Item .env.example .env
# Edit .env to set DATABASE_URL and JWT_SECRET_KEY as needed
```

### Step 5: Run Database Migrations (PostgreSQL)
When PostgreSQL is running:
```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

To view the generated SQL without executing against a live database:
```powershell
.\.venv\Scripts\alembic.exe upgrade head --sql
```

### Step 6: Run Backend Test Suite
```powershell
.\.venv\Scripts\pytest.exe tests -v
```

### Step 7: Launch Development Server
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- Health Check: `http://127.0.0.1:8000/api/health`
- Swagger Documentation: `http://127.0.0.1:8000/api/docs`
