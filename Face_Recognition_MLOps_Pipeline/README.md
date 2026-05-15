# MSML 605 — Face Recognition Pipeline

MSAI605-PCS2 / MSML605-PCS2: Computing Systems for Machine Learning — Spring 2026

A containerized face recognition system built on the [LFW dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html). An admin runs training passes that are tracked in MLflow; users query the best model for predictions.

---

## Architecture

```
┌─────────────────┐   POST /jobs/retrain   ┌──────────────────┐   POST /models/reload  ┌───────────────┐
│  preprocessing  │ ─────────────────────► │    retraining    │ ──────────────────────► │   inference   │
│    :8001        │                         │     :8002        │                          │    :8003      │
└─────────────────┘                         └──────────────────┘                          └───────────────┘
        │                                           │                                            │
        │                                           │ log runs / register model                  │ serve predictions
        │                                           ▼                                            │
        │                                   ┌──────────────┐                                    │
        └──── artifacts (local ./storage) ──►   MLflow UI  │◄───────────────────────────────────┘
                                            │    :5001     │
                                            └──────────────┘
```

**4 Docker containers:**

| Service | Port | Responsibility |
|---|---|---|
| `preprocessing` | 8001 | Downloads LFW, fits PCA + scaler, saves artifacts, triggers retraining |
| `retraining` | 8002 | Trains MLP, logs metrics to MLflow, promotes best model |
| `inference` | 8003 | Serves predictions using the current best model bundle |
| `mlflow` | 5001 | Experiment tracking UI and model registry |

**Data flow:**
- Services communicate via small JSON over HTTP (control plane)
- Large artifacts (data, weights, preprocessors) are read/written to `./storage` (data plane)
- No large arrays travel over HTTP

---

## Prerequisites

- Docker + Docker Compose
- Python 3.9+ with the following packages installed locally (for the demo scripts only):

```bash
pip install scikit-learn httpx
```

---

## Admin workflow

The containers stay up persistently — spin them up once, then trigger training passes as needed from a separate terminal.

### Step 1 — Start the stack (one time)

```bash
docker compose up --build
```

Starts all 4 containers: `preprocessing`, `retraining`, `inference`, and `mlflow`.

> First build takes several minutes because PyTorch is large. Subsequent builds are fast — pip layers are cached and only changed code layers re-run.

### Step 2 — Trigger a training pass (separate admin terminal)

```bash
python scripts/admin_train.py
```

**What it does:** Sends a request to the preprocessing service, which downloads LFW, fits a StandardScaler + PCA, then automatically chains into MLP training. Metrics and model artifacts are logged to MLflow, and the best model is promoted to `production`. Takes 3–5 minutes.

Optional flags:
```bash
python scripts/admin_train.py --min-faces 40 --epochs 30
```

### Step 3 — View results in MLflow

Open **http://localhost:5001** in your browser.

- **Experiments → face_recognition_lfw** — every training run with full metrics, parameters, and artifacts
- **Models → face_recognition_model** — registered model versions; the best run is aliased `production`

---

## User workflow

With the stack already running, users just run the prediction script — no container management needed.

### Run predictions

```bash
python scripts/user_predict.py
```

**What it does:** Fetches LFW locally, picks N random face images, sends each as raw pixel values to the inference service (`localhost:8003`), and prints a results table comparing actual vs. predicted identity with confidence scores.

Example output:

```
=================================================================
  Face Recognition — User Prediction Demo
=================================================================
  Inference service : http://localhost:8003
  Samples           : 15

Actual                       Predicted                        Conf Match
-------------------------------------------------------------------------
Lleyton Hewitt               Lleyton Hewitt                  96.2% YES
George W Bush                George W Bush                   97.2% YES
Tony Blair                   Tony Blair                      85.2% YES
David Beckham                Tony Blair                      30.7% no
Colin Powell                 Colin Powell                    96.6% YES
Jean Chretien                Jean Chretien                   86.3% YES
Gerhard Schroeder            Gerhard Schroeder               88.4% YES
Ariel Sharon                 Ariel Sharon                    37.4% YES
...
-------------------------------------------------------------------------
  Accuracy on this sample: 14/15 = 93.3%
```

Optional flags:
```bash
python scripts/user_predict.py --n 20 --seed 7
```

---

## Project Structure

```
.
├── docker-compose.yml
├── .env.example               # Copy to .env to override defaults
├── scripts/
│   ├── admin_train.py         # Admin: trigger a training pass
│   └── user_predict.py        # User: predict faces from LFW
├── services/
│   ├── preprocessing/         # FastAPI service — port 8001
│   ├── retraining/            # FastAPI service — port 8002
│   ├── inference/             # FastAPI service — port 8003
│   └── mlflow/                # MLflow tracking server — port 5001
└── shared/                    # Code shared across all services
    ├── artifacts/             # Logical storage key naming
    ├── config/                # Pydantic settings (env-driven)
    ├── core/                  # ML model, training loop, data loading
    ├── schemas/               # Cross-service request/response DTOs
    └── storage/               # Storage abstraction (local + S3)
```

---

## API Endpoints

| Service | Method | Path | Description |
|---|---|---|---|
| preprocessing | POST | `/jobs/preprocess` | Run a full preprocessing job |
| preprocessing | GET | `/health` | Health check |
| retraining | POST | `/jobs/retrain` | Run a training job from a manifest URI |
| retraining | GET | `/health` | Health check |
| inference | POST | `/predict/image` | Predict from raw pixel values |
| inference | POST | `/predict` | Predict from a pre-computed PCA vector |
| inference | POST | `/models/reload` | Reload the latest model bundle from storage |
| inference | GET | `/health` | Health check |

Interactive docs: `http://localhost:8001/docs`, `:8002/docs`, `:8003/docs`

---

## Model

- **Architecture:** 2-layer MLP — `input → 512 → 256 → n_classes` with BatchNorm, ReLU, Dropout
- **Preprocessing:** StandardScaler → PCA (128 components, whitened)
- **Training:** AdamW, CosineAnnealingLR, CrossEntropyLoss (label smoothing 0.1), early stopping (patience 8)
- **Dataset:** LFW (`min_faces_per_person=30`, `resize=0.4`), ~34 classes, ~2370 samples

---

## Switching to AWS S3

Storage is abstracted behind a backend interface. To use S3 instead of the local filesystem:

1. Copy `.env.example` to `.env`
2. Set the S3 variables:
   ```
   STORAGE_BACKEND=s3
   S3_BUCKET=my-bucket-name
   S3_REGION=us-east-1
   ```
3. Add `boto3>=1.34.0` to each service's `requirements.txt`
4. Provide AWS credentials via environment variables or an IAM role:
   ```
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```
5. Rebuild: `docker compose up --build`

No code changes are required — only environment configuration.

---

## Configuration

All settings are environment-driven. See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `STORAGE_ROOT` | `./storage` | Root path for local storage |
| `MLFLOW_TRACKING_URI` | *(auto)* | Set by docker-compose to `http://mlflow:5000` |
| `PREPROCESS_TRIGGER_RETRAIN` | `true` | Auto-trigger retraining after preprocessing |
| `RETRAIN_NOTIFY_INFERENCE` | `true` | Auto-reload inference after retraining |
