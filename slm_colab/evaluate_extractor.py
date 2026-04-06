import argparse
import json
import re
from pathlib import Path

from huggingface_hub import InferenceClient

REQ_FIELDS = ["customer_name", "customer_site", "requested_forecast_qty"]
ALL_FIELDS = [
    "customer_name",
    "customer_site",
    "requested_forecast_qty",
    "forecast_period_start",
    "forecast_period_end",
    "uom",
    "item_id",
    "item_description",
]


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def norm(v):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 4)
    return str(v).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model repo id")
    parser.add_argument("--token", required=True, help="HF token")
    parser.add_argument(
        "--valid-json",
        default="4thaxis/slm_ollama/synthetic/valid_pairs.json",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    rows = json.loads(Path(args.valid_json).read_text(encoding="utf-8"))
    rows = rows[: min(args.max_samples, len(rows))]

    client = InferenceClient(model=args.model, token=args.token)

    total = len(rows)
    parse_ok = 0
    req_pass = 0
    field_match_total = 0
    field_match_hits = 0

    system = (
        "Extract structured fields and return strict JSON with keys: "
        "customer_name, customer_site, requested_forecast_qty, forecast_period_start, "
        "forecast_period_end, uom, item_id, item_description, confidence, missing_fields, evidence."
    )

    for row in rows:
        prompt = row["input_text"]
        target = row

        msg = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        out = client.chat_completion(messages=msg, max_tokens=400, temperature=0.0)
        text = out.choices[0].message.content
        pred = extract_json(text)

        if pred is None:
            continue

        parse_ok += 1

        if all(pred.get(f) not in (None, "") for f in REQ_FIELDS):
            req_pass += 1

        for f in ALL_FIELDS:
            field_match_total += 1
            if norm(pred.get(f)) == norm(target.get(f)):
                field_match_hits += 1

    report = {
        "total_samples": total,
        "json_parse_success_rate": parse_ok / total if total else 0,
        "required_fields_pass_rate": req_pass / total if total else 0,
        "field_exact_match_rate": field_match_hits / field_match_total if field_match_total else 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
