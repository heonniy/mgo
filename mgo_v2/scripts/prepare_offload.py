#!/usr/bin/env python3
import argparse
import json
from mgo_v2.model_loader import prepare_checkpoint_store

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(prepare_checkpoint_store(args.model, args.output), indent=2))
