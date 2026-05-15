#!/usr/bin/env python3
"""
Admin script: trigger a full training pass and report results.

Usage:
    python scripts/admin_train.py
    python scripts/admin_train.py --min-faces 40 --epochs 30
    python scripts/admin_train.py --host http://localhost:8001 --mlflow-ui http://localhost:5001
"""

import argparse
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required.  Run: pip install httpx")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trigger a training pass on the face recognition pipeline.")
    p.add_argument("--host", default="http://localhost:8001", help="Preprocessing service base URL")
    p.add_argument("--mlflow-ui", default="http://localhost:5001", help="MLflow UI URL (for display only)")
    p.add_argument("--min-faces", type=int, default=None, help="Min faces per person (default: service default of 30)")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default: service default of 50)")
    p.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds (default: 600)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    payload: dict = {}
    if args.min_faces is not None:
        payload["min_faces_per_person"] = args.min_faces
    if args.epochs is not None:
        payload["epochs"] = args.epochs

    preprocess_url = args.host.rstrip("/") + "/jobs/preprocess"

    print("=" * 60)
    print("  Face Recognition — Admin Training Pass")
    print("=" * 60)
    print(f"  Preprocessing service : {args.host}")
    print(f"  MLflow UI             : {args.mlflow_ui}")
    if payload:
        print(f"  Overrides             : {payload}")
    print()
    print("Starting preprocessing + training pipeline...")
    print("(This may take several minutes while LFW is fetched and the model trains.)")
    print()

    start = time.time()
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(preprocess_url, json=payload)
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to {args.host}")
        print("Make sure the stack is running:  docker compose up")
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"ERROR: Request timed out after {args.timeout}s")
        print("Training may still be running in the background.")
        sys.exit(1)

    elapsed = time.time() - start

    if response.status_code != 200:
        print(f"ERROR: Preprocessing service returned HTTP {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    print(f"Pipeline completed in {elapsed:.1f}s")
    print()

    # Preprocessing result
    print("── Preprocessing ─────────────────────────────────────")
    print(f"  Job ID   : {data.get('job_id', 'n/a')}")
    print(f"  Status   : {data.get('status', 'n/a')}")
    manifest = data.get("manifest", {})
    if manifest:
        print(f"  Classes  : {manifest.get('n_classes', '?')}")
        print(f"  Train N  : {manifest.get('train_size', '?')}")
        print(f"  Test N   : {manifest.get('test_size', '?')}")
        print(f"  PCA dims : {manifest.get('n_pca_components', '?')}")
    print()

    # Retrain chained automatically — surface what we know
    retrain_triggered = data.get("retrain_triggered", False)
    if retrain_triggered:
        print("── Retraining ────────────────────────────────────────")
        print("  Retraining job was triggered automatically.")
        print()
        print(f"  View training metrics in MLflow UI:")
        print(f"  {args.mlflow_ui}")
        print()
        print("  Experiment : face_recognition_lfw")
        print("  Look for the run tagged  promoted=true  to see the current best model.")
    else:
        print("  Note: PREPROCESS_TRIGGER_RETRAIN is disabled; retraining was not triggered.")

    print()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
