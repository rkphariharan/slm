import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SYSTEM_MAP = (
    "You map uploaded planning file headers to Tahoe SCP schema. "
    "Return strict JSON with keys: dataset_name, target_table, mappings, "
    "required_columns_missing, unmapped_headers, confidence."
)


HEADER_MAP: Dict[str, str] = {
    "item": "item_no",
    "part_no": "item_no",
    "sku_no": "item_no",
    "item_id": "item_id",
    "prd_lvl_member_name": "item_no",
    "org_lvl_member_name": "org_id",
    "customer name": "customer_id",
    "cus_lvl_member_name": "customer_id",
    "customer site name": "customer_site_id",
    "cus_site_lvl_member_name": "customer_site_id",
    "tim_lvl_member_value": "shipment_date",
    "value_number": "shipped_qty",
    "order_type_flag": "order_type_flag",
    "deleted_flag": "deleted_flag",
    "deleted_flag ": "deleted_flag",
    "site number": "customer_site_id",
    "site name": "customer_site_id",
    "account number": "customer_id",
    "source system code": "source_system",
    "*organization code": "org_id",
    "organization code": "org_id",
    "*resource code": "resource_code",
    "resource code": "resource_code",
    "*start time": "capacity_date",
    "*capacity units": "max_capacity_units",
    "*availability": "available_hours",
    "effective date": "bucket_start_date",
    "disable date": "bucket_end_date",
}


TABLE_REQUIRED: Dict[str, List[str]] = {
    "stg_shipment_raw": ["item_no", "org_id", "shipment_date", "shipped_qty"],
    "stg_forecast_raw": ["item_no", "org_id", "customer_id", "bucket_start_date", "forecast_qty"],
    "stg_capacity_raw": ["org_id", "capacity_date", "resource_code", "available_hours"],
    "dim_customer_site": ["customer_site_id"],
    "dim_item": ["item_no"],
}


@dataclass
class HeaderBundle:
    file_name: str
    sheet_name: Optional[str]
    headers: List[str]
    target_table: str


def norm(text: str) -> str:
    return str(text or "").strip().lower()


def infer_target_table(file_name: str) -> str:
    n = norm(file_name)
    if "forecast_output" in n or "forecast" in n:
        return "stg_forecast_raw"
    if "shipment" in n:
        return "stg_shipment_raw"
    if "resourceavailability" in n or "resource" in n:
        return "stg_capacity_raw"
    if "customer site" in n:
        return "dim_customer_site"
    if "item master" in n:
        return "dim_item"
    return "stg_shipment_raw"


def read_headers(path: Path) -> List[HeaderBundle]:
    bundles: List[HeaderBundle] = []
    suffix = path.suffix.lower()

    if suffix == ".csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
        bundles.append(HeaderBundle(path.name, None, cols, infer_target_table(path.name)))
        return bundles

    if suffix in {".xlsx", ".xlsm"}:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            try:
                cols = list(pd.read_excel(path, sheet_name=sheet, nrows=0).columns)
            except Exception:
                continue
            if len(cols) == 0:
                continue
            bundles.append(HeaderBundle(path.name, sheet, cols, infer_target_table(path.name)))
        return bundles

    return bundles


def build_mapping(headers: List[str], target_table: str) -> Tuple[List[Dict], List[str], List[str], float]:
    mapped: List[Dict] = []
    mapped_targets: List[str] = []
    unmapped: List[str] = []

    for header in headers:
        key = norm(header)
        target = HEADER_MAP.get(key)
        if target:
            mapped.append({"source_header": header, "target_column": target})
            mapped_targets.append(target)
        else:
            unmapped.append(header)

    required = TABLE_REQUIRED.get(target_table, [])
    required_missing = [c for c in required if c not in mapped_targets]

    if len(required) == 0:
        confidence = 0.7
    else:
        confidence = max(0.2, min(1.0, (len(required) - len(required_missing)) / len(required)))

    return mapped, required_missing, unmapped, round(confidence, 3)


def build_prompt(bundle: HeaderBundle, headers: List[str]) -> str:
    sheet_text = f"Sheet: {bundle.sheet_name}\n" if bundle.sheet_name else ""
    header_lines = "\n".join([f"- {h}" for h in headers])
    return (
        f"Map this dataset to SCP staging table columns.\n"
        f"Dataset: {bundle.file_name}\n"
        f"{sheet_text}"
        f"Candidate target table: {bundle.target_table}\n"
        f"Headers:\n{header_lines}\n"
        "Return only JSON."
    )


def make_record(bundle: HeaderBundle, headers: List[str]) -> Dict:
    mappings, required_missing, unmapped, confidence = build_mapping(headers, bundle.target_table)

    assistant = {
        "dataset_name": bundle.file_name,
        "target_table": bundle.target_table,
        "mappings": mappings,
        "required_columns_missing": required_missing,
        "unmapped_headers": unmapped,
        "confidence": confidence,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MAP},
            {"role": "user", "content": build_prompt(bundle, headers)},
            {"role": "assistant", "content": json.dumps(assistant)},
        ]
    }


def variants(headers: List[str], seed: int) -> List[List[str]]:
    r = random.Random(seed)
    out = [headers]

    shuffled = headers[:]
    r.shuffle(shuffled)
    out.append(shuffled)

    lowered = [h.lower() for h in headers]
    out.append(lowered)

    with_noise = headers[:] + ["custom_upload_field_1", "business_note"]
    r.shuffle(with_noise)
    out.append(with_noise)

    return out


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    workspace = Path(__file__).resolve().parents[2]

    synthetic_dir = workspace / "4thaxis" / "synthetic"
    demo_dataset_dir = workspace / "4thaxis" / "demo" / "dataset"
    out_dir = workspace / "slm" / "mixed_dataset_v2"

    old_train = load_jsonl(synthetic_dir / "train_chat.jsonl")
    old_valid = load_jsonl(synthetic_dir / "valid_chat.jsonl")

    map_records: List[Dict] = []
    seed_counter = 100

    for path in sorted(demo_dataset_dir.iterdir()):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xlsm"}:
            continue
        for bundle in read_headers(path):
            for v in variants(bundle.headers, seed_counter):
                map_records.append(make_record(bundle, v))
                seed_counter += 1

    random.Random(42).shuffle(map_records)
    split = int(len(map_records) * 0.9)
    map_train = map_records[:split]
    map_valid = map_records[split:]

    mixed_train = old_train + map_train
    mixed_valid = old_valid + map_valid
    random.Random(7).shuffle(mixed_train)
    random.Random(9).shuffle(mixed_valid)

    write_jsonl(out_dir / "train_chat_mixed_v2.jsonl", mixed_train)
    write_jsonl(out_dir / "valid_chat_mixed_v2.jsonl", mixed_valid)

    metadata = {
        "old_train_rows": len(old_train),
        "old_valid_rows": len(old_valid),
        "new_mapping_rows_total": len(map_records),
        "new_mapping_train_rows": len(map_train),
        "new_mapping_valid_rows": len(map_valid),
        "mixed_train_rows": len(mixed_train),
        "mixed_valid_rows": len(mixed_valid),
        "sources": {
            "synthetic": str(synthetic_dir),
            "demo_dataset": str(demo_dataset_dir),
        },
    }
    (out_dir / "mixed_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
