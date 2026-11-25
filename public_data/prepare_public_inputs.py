"""Convert the released CSV files into the inputs evaluation/validate_models_10000.py reads.

The released descriptors are already standardized over the 10,000 users and rounded to two decimals; they are copied as
they are, never standardized again. Each public user_key gets an integer index built from the sorted public keys alone.

Usage: python public_data/prepare_public_inputs.py --data data --out public_inputs
Writes: global_features.parquet (target_user_id plus the six generator inputs, named and ordered as in
weights/hyperpeft_lora/feature_names.json), labels_curiosity.csv, labels_politeness.csv, labels_sentiment.csv (question
rate, profanity and sentiment cohorts with the label names the validator expects), and user_key_index.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PLAIN_TO_INPUT = {"first_person_ratio": "gstat_firstperson_ratio", "punctuation_ratio": "gstat_punct_ratio",
                  "question_ratio": "gstat_question_ratio", "top_subreddit_share": "gstat_user_sr_max_share",
                  "subreddit_entropy": "gstat_user_subreddit_entropy", "type_token_ratio": "gstat_user_ttr"}
COHORTS = {"cohort_question_rate.csv": ("labels_curiosity.csv", {"high": "inquisitive", "low": "declarative"}),
           "cohort_profanity.csv": ("labels_politeness.csv", {"high": "vulgar", "low": "polite"}),
           "cohort_sentiment.csv": ("labels_sentiment.csv", {"positive": "empath", "negative": "rage"})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="public_inputs")
    ap.add_argument("--feature_names", default="weights/hyperpeft_lora/feature_names.json")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = json.loads(Path(a.feature_names).read_text())
    d = pd.read_csv(data / "descriptors.csv").rename(columns=PLAIN_TO_INPUT)
    missing = [c for c in order if c not in d.columns]
    if missing:
        raise SystemExit(f"descriptors.csv lacks {missing}")
    index = {k: i for i, k in enumerate(sorted(d["user_key"]))}
    d.insert(0, "target_user_id", d["user_key"].map(index).astype("int64"))
    d[["target_user_id"] + order].to_parquet(out / "global_features.parquet", index=False)
    pd.DataFrame({"user_key": list(index), "target_user_id": list(index.values())}).to_csv(out / "user_key_index.csv", index=False)
    for src, (dst, names) in COHORTS.items():
        c = pd.read_csv(data / src)
        bad = sorted(set(c["label"]) - set(names))
        if bad:
            raise SystemExit(f"{src}: unexpected labels {bad}")
        pd.DataFrame({"target_user_id": c["user_key"].map(index).astype("int64"), "label": c["label"].map(names)}).to_csv(out / dst, index=False)
    print(f"wrote {out}: {len(d)} users, inputs {order}, cohorts {[v[0] for v in COHORTS.values()]}")


if __name__ == "__main__":
    main()
