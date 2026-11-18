"""Verify the sealed benchmark records and the README regions rendered from them.

Standard library only. Run from the repository root:

    python benchmarks/verify_sealed_records.py

This checks integrity, not provenance: every sealed record must be canonical
JSON whose ``closed_record_sha256`` recomputes and whose file bytes match the
pinned digest, the publication gate must bind the campaign report, and every
governed region of ``benchmarks/README.md`` (and the rendered campaign appendix
README) must match the bytes rendered from those records at the
``vllm-0300-campaign-archive`` tag. Full re-validation against the campaign
plans, ledgers, and validators runs only at that tag.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ARCHIVE_TAG = "vllm-0300-campaign-archive"

BENCHMARK_ROOT = Path(__file__).resolve().parent

CAMPAIGN_REPORT = "appendix/vllm-0271-publication-v1/campaign-report.json"
PUBLICATION_GATE = "appendix/vllm-0271-publication-v1/benchmark-publication-gate.json"
QUALITY_V2_REPORT = "appendix/vllm-0300-quality-v2/quality-v2-report.json"
AUXILIARY_RERUN = "appendix/vllm-0300-auxiliary-latency-rerun/auxiliary-latency-rerun.json"

# Sealing rule per record: "omit" hashes the record without its
# closed_record_sha256 key; "blank" hashes it with that key set to "".
SEALED_RECORDS: dict[str, tuple[str, str]] = {
    CAMPAIGN_REPORT: (
        "omit",
        "6ffa458da80589f46e548ddeb69069a9645d714535eda47263a105190570bf63",
    ),
    QUALITY_V2_REPORT: (
        "blank",
        "12294503dadef15a4621ecbd95763f39cca5710fd3f75f946dd711b4e01a79f6",
    ),
    AUXILIARY_RERUN: (
        "blank",
        "d2dbd64fdc23370d2cdc6a87931f5b488e94713742d3600acb693ad51e5c53b7",
    ),
}

PINNED_FILES: dict[str, str] = {
    PUBLICATION_GATE: "6efb580d3b3003cadbb21c0ed431792558a9e8fbce85b727fb10650129c335a3",
    "appendix/vllm-0271-publication-v1/README.md": (
        "bb5ea8a4e65dc40711ca6cf3219ba5473b9eb4edca89bc8ae8f8fd98c6e35159"
    ),
}

# SHA-256 of the UTF-8 bytes between each region's begin and end markers in
# benchmarks/README.md, in document order.
GOVERNED_REGIONS: dict[str, str] = {
    "cachet:vllm-0271-publication-table:status": (
        "36a7356bc4928ac41f5dc8df2408ca1a8358e21a902f635aa3c09b607e810c40"
    ),
    "cachet:vllm-0271-publication-table:campaign-disclosures": (
        "8445ab2f18ec1d9a9db1664f427a906e9da3b2542615cf7d082e416bb29442c6"
    ),
    "cachet:vllm-0271-publication-table:core-latency": (
        "1a2f45e97f7b7d19f3c192636da66a4e0d94743b015db3748625cada359dc502"
    ),
    "cachet:vllm-0271-publication-table:latency-estimands": (
        "bbfdfc04a522798c37ab8ba9d994616146439487208d82bc63d1e6cba5d678ec"
    ),
    "cachet:vllm-0271-publication-table:dataset-scores": (
        "0675c12aeca09ac0aad798d384dcd941fc953daf74411f0bb4060f5574e68116"
    ),
    "cachet:vllm-0271-publication-table:niah-grid": (
        "0648d280f4daf294d3bbaecc0e4ff67b34a86b3b76fd90b6b7c3bd0f755cc189"
    ),
    "cachet:vllm-0300-quality-v2": (
        "081a6394342996c8229c2fb167c3f0f7fbbba5f5b8d9dd2af05a1a7b585e92e9"
    ),
    "cachet:vllm-0271-publication-table:precision": (
        "7953d23330fac6a711ecca652d2ce6fddfe880d5e21a29abbc7180daf0be4c26"
    ),
    "cachet:vllm-0271-publication-table:storage": (
        "bce9b7a0fe82b272192847ff803264ad2b2503f360f052b2da3f2359f5406793"
    ),
    "cachet:vllm-0271-publication-table:hardware": (
        "7a97e62dfa7e10b57fc33499804051673b419fbddd47b6e4a1e663663e93044a"
    ),
    "cachet:vllm-0300-auxiliary-latency-rerun": (
        "3aacb90db80c8ccada6016d09d74318483376808126d6952eeb804fefc1fa85e"
    ),
    "cachet:vllm-0271-publication-table:platform": (
        "6f03eea04a909e20963720a8f53e02fe1750fe1e4a94875282d94acf183d58ee"
    ),
    "cachet:vllm-0271-publication-table:resource-cache": (
        "2b7acc16511d97eb827a5cd281116e29ecf696a3f9a649b3f09c9083e24b92f0"
    ),
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def closed_record_sha256(record: dict[str, Any], rule: str) -> str:
    body = dict(record)
    if rule == "omit":
        body.pop("closed_record_sha256", None)
    elif rule == "blank":
        body["closed_record_sha256"] = ""
    else:
        raise ValueError(f"unknown sealing rule {rule!r}")
    return canonical_sha256(body)


def load_canonical_record(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            value[key] = item
        return value

    record = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a JSON object")
    canonical = (
        json.dumps(record, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError(f"{label} is not canonical sorted-key JSON")
    return record


def governed_region_sha256(readme: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    previous_end = -1
    for region_id in GOVERNED_REGIONS:
        begin = f"<!-- {region_id}:begin -->"
        end = f"<!-- {region_id}:end -->"
        if readme.count(begin) != 1 or readme.count(end) != 1:
            raise ValueError(f"README must contain exactly one {region_id} region")
        begin_at = readme.index(begin)
        end_at = readme.index(end)
        if not previous_end < begin_at < end_at:
            raise ValueError(f"README region {region_id} is out of order")
        observed[region_id] = hashlib.sha256(
            readme[begin_at + len(begin) : end_at].encode("utf-8")
        ).hexdigest()
        previous_end = end_at + len(end)
    return observed


def verify(benchmark_root: Path = BENCHMARK_ROOT) -> list[str]:
    """Return a list of problems; an empty list means every check passed."""

    issues: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    contents: dict[str, bytes | None] = {}

    def read(relative_path: str) -> bytes | None:
        if relative_path not in contents:
            try:
                contents[relative_path] = (benchmark_root / relative_path).read_bytes()
            except OSError as exc:
                issues.append(
                    f"{relative_path} cannot be read: {exc.strerror or type(exc).__name__}"
                )
                contents[relative_path] = None
        return contents[relative_path]

    for relative_path, (rule, file_sha256) in SEALED_RECORDS.items():
        raw = read(relative_path)
        if raw is None:
            continue
        if hashlib.sha256(raw).hexdigest() != file_sha256:
            issues.append(f"{relative_path} bytes differ from the sealed record")
        try:
            record = load_canonical_record(raw, relative_path)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if record.get("closed_record_sha256") != closed_record_sha256(record, rule):
            issues.append(f"{relative_path} closed_record_sha256 does not recompute")
        records[relative_path] = record
    for relative_path, file_sha256 in PINNED_FILES.items():
        raw = read(relative_path)
        if raw is not None and hashlib.sha256(raw).hexdigest() != file_sha256:
            issues.append(f"{relative_path} bytes differ from the sealed rendering")
    gate_raw = read(PUBLICATION_GATE)
    try:
        gate = None if gate_raw is None else load_canonical_record(gate_raw, PUBLICATION_GATE)
    except ValueError as exc:
        issues.append(str(exc))
        gate = None
    if gate is not None:
        report = records.get(CAMPAIGN_REPORT)
        if report is not None and gate.get("benchmark_payload_digest") != report.get(
            "closed_record_sha256"
        ):
            issues.append("publication gate does not bind the campaign report digest")
    readme_raw = read("README.md")
    if readme_raw is None:
        return issues
    try:
        observed = governed_region_sha256(readme_raw.decode("utf-8"))
    except ValueError as exc:
        issues.append(str(exc))
    else:
        for region_id, expected in GOVERNED_REGIONS.items():
            if observed[region_id] != expected:
                issues.append(f"README region {region_id} differs from its sealed rendering")
    return issues


def main() -> int:
    issues = verify()
    for issue in issues:
        print(f"FAIL: {issue}", file=sys.stderr)
    if issues:
        return 1
    print(
        f"OK: {len(SEALED_RECORDS)} sealed records, {len(PINNED_FILES)} pinned files, "
        f"{len(GOVERNED_REGIONS)} README regions (full re-validation: git checkout {ARCHIVE_TAG})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
