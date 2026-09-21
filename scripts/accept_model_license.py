#!/usr/bin/env python3
"""
Record or inspect a model license acceptance.

This is the only supported way to unlock weight downloads, and it is meant to
be run by a person who has just read the terms. Run it without --yes to see
what would be agreed to without recording anything.

Examples:
    python scripts/accept_model_license.py --list
    python scripts/accept_model_license.py --model Qwen/Qwen-Image-2.1
    python scripts/accept_model_license.py --model Qwen/Qwen-Image-2.1 --yes
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.runtime.model_access import (  # noqa: E402
    ACCEPTANCE_FILE,
    MODEL_LICENSES,
    ModelLicenseNotAccepted,
    _read_record,
    accept,
    is_accepted,
    terms_for,
)


def print_terms(model_id: str) -> None:
    """Print where the terms live and whether this machine has accepted them."""
    info = terms_for(model_id)
    state = "accepted" if is_accepted(model_id) else "NOT accepted"
    license_label = info["license"] or "not recorded in this checkout"
    print(f"model      : {info['model_id']}")
    print(f"license    : {license_label}")
    print(f"terms at   : {info['url']}")
    print(f"status     : {state}")
    if info["requires_review"]:
        print("\nThis checkout has no verified license record for this model, so it")
        print("cannot be unlocked from here. Add the record to MODEL_LICENSES once")
        print("you have read the terms upstream.")


def write_record(record: dict) -> None:
    ACCEPTANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCEPTANCE_FILE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record or inspect a model license acceptance.")
    parser.add_argument("--model", help="model id to inspect or accept")
    parser.add_argument("--list", action="store_true", help="list every known model")
    parser.add_argument("--yes", action="store_true", help="record the acceptance")
    parser.add_argument("--revoke", action="store_true", help="drop a recorded acceptance")
    args = parser.parse_args()

    if args.list or not args.model:
        for model_id in MODEL_LICENSES:
            print_terms(model_id)
            print()
        return 0

    key = terms_for(args.model)["model_id"]

    if args.revoke:
        record = _read_record()
        record.pop(key, None)
        write_record(record)
        print(f"Revoked acceptance for {key}.")
        return 0

    print_terms(args.model)
    if not args.yes:
        print("\nNothing recorded. Re-run with --yes once you have read the terms above.")
        return 0

    try:
        entry = accept(key, accepted_by="cli")
    except ModelLicenseNotAccepted as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\nRecorded: {entry['accepted_at']}  {entry['license']}")
    print(f"Written to: {ACCEPTANCE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
