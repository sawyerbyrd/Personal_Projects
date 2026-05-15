#!/usr/bin/env python3
"""
User demo script: predict faces from the LFW dataset.

Fetches LFW locally, picks N random samples, sends each to the inference
service, and prints a results table showing actual vs predicted identity.

Usage:
    python scripts/user_predict.py
    python scripts/user_predict.py --n 20
    python scripts/user_predict.py --host http://localhost:8003 --n 5 --seed 0
"""

import argparse
import random
import sys

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required.  Run: pip install httpx")
    sys.exit(1)

try:
    from sklearn.datasets import fetch_lfw_people
except ImportError:
    print("ERROR: scikit-learn is required.  Run: pip install scikit-learn")
    sys.exit(1)


# Must match the values used during training (see shared/config/training.py defaults).
DEFAULT_MIN_FACES = 30
DEFAULT_RESIZE = 0.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo: predict faces from LFW using the inference service.")
    p.add_argument("--host", default="http://localhost:8003", help="Inference service base URL")
    p.add_argument("--n", type=int, default=10, help="Number of faces to predict (default: 10)")
    p.add_argument("--min-faces", type=int, default=DEFAULT_MIN_FACES,
                   help=f"Min faces per person filter — must match training value (default: {DEFAULT_MIN_FACES})")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sample selection (default: 42)")
    return p.parse_args()


def check_service(host: str) -> None:
    url = host.rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(url)
            r.raise_for_status()
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to inference service at {host}")
        print("Make sure the stack is running:  docker compose up")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Health check failed — {exc}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("  Face Recognition — User Prediction Demo")
    print("=" * 65)
    print(f"  Inference service : {args.host}")
    print(f"  Samples           : {args.n}")
    print(f"  min_faces_per_person (must match training): {args.min_faces}")
    print()

    print("Checking inference service health...")
    check_service(args.host)
    print("  Service is up.")
    print()

    print(f"Fetching LFW dataset (min_faces_per_person={args.min_faces}, resize={DEFAULT_RESIZE})...")
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_faces,
        resize=DEFAULT_RESIZE,
        color=False,
    )
    X, y, class_names = lfw.data, lfw.target, list(lfw.target_names)
    print(f"  Loaded {len(X)} samples, {len(class_names)} identities.")
    print()

    rng = random.Random(args.seed)
    n = min(args.n, len(X))
    indices = rng.sample(range(len(X)), n)

    predict_url = args.host.rstrip("/") + "/predict/image"

    # Column widths
    COL_ACTUAL    = 28
    COL_PREDICTED = 28
    COL_CONF      = 8
    COL_MATCH     = 6

    header = (
        f"{'Actual':<{COL_ACTUAL}} "
        f"{'Predicted':<{COL_PREDICTED}} "
        f"{'Conf':>{COL_CONF}} "
        f"{'Match':<{COL_MATCH}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    correct = 0
    with httpx.Client(timeout=30) as client:
        for sample_idx in indices:
            pixels = X[sample_idx].tolist()
            actual_name = class_names[y[sample_idx]]

            try:
                resp = client.post(predict_url, json={"pixels": pixels})
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPStatusError as exc:
                print(f"  HTTP {exc.response.status_code}: {exc.response.text}")
                continue
            except Exception as exc:
                print(f"  Request failed: {exc}")
                continue

            predicted_name = result.get("class_name") or f"class_{result['class_index']}"
            confidence = result["confidence"]
            match = predicted_name == actual_name
            if match:
                correct += 1
            match_str = "YES" if match else "no"

            # Truncate long names to fit columns
            actual_disp    = actual_name[:COL_ACTUAL - 1]
            predicted_disp = predicted_name[:COL_PREDICTED - 1]

            print(
                f"{actual_disp:<{COL_ACTUAL}} "
                f"{predicted_disp:<{COL_PREDICTED}} "
                f"{confidence:>{COL_CONF}.1%} "
                f"{match_str:<{COL_MATCH}}"
            )

    print(sep)
    print(f"  Accuracy on this sample: {correct}/{n} = {correct/n:.1%}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
