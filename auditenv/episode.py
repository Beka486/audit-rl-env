"""Episode container and on-disk format.

Layout of an exported episode::

    <episode_dir>/public/manifest.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv  policy.json
    <episode_dir>/private/truth.json    provenance.json        (evaluator only)
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
NULLABLE_LEDGER = {"entity_id", "invoice_id", "payment_id"}


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def ledger_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(LEDGER_COLUMNS)
    for r in rows:
        w.writerow(["null" if r[c] is None else r[c] for c in LEDGER_COLUMNS])
    return buf.getvalue()


def fx_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FX_COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in FX_COLUMNS])
    return buf.getvalue()


def parse_ledger_csv(text: str) -> List[Dict[str, Any]]:
    rd = csv.DictReader(io.StringIO(text))
    if rd.fieldnames != LEDGER_COLUMNS:
        raise ValueError(f"ledger.csv columns must be {LEDGER_COLUMNS}")
    out = []
    for r in rd:
        row: Dict[str, Any] = dict(r)
        for c in ("debit_minor", "credit_minor"):
            row[c] = int(row[c])
        for c in NULLABLE_LEDGER:
            if row[c] == "null":
                row[c] = None
        out.append(row)
    return out


def parse_fx_csv(text: str) -> List[Dict[str, Any]]:
    rd = csv.DictReader(io.StringIO(text))
    if rd.fieldnames != FX_COLUMNS:
        raise ValueError(f"fx.csv columns must be {FX_COLUMNS}")
    return [dict(r) for r in rd]


def jsonl(rows: List[Dict[str, Any]]) -> str:
    return "".join(dumps(r) + "\n" for r in rows)


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class Episode:
    manifest: Dict[str, Any]
    policy: Dict[str, Any]
    ledger: List[Dict[str, Any]]
    invoices: List[Dict[str, Any]]
    payments: List[Dict[str, Any]]
    fx: List[Dict[str, Any]]
    truth: Dict[str, Any]
    provenance: Dict[str, Any]

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    def public(self) -> Dict[str, Any]:
        return {"manifest": copy.deepcopy(self.manifest), "policy": copy.deepcopy(self.policy),
                "ledger": copy.deepcopy(self.ledger), "invoices": copy.deepcopy(self.invoices),
                "payments": copy.deepcopy(self.payments), "fx": copy.deepcopy(self.fx)}

    def clean_public(self) -> Dict[str, Any]:
        """Records as they were before error injection (reconstructed from provenance)."""
        pub = self.public()
        added = set(self.provenance["added_ledger_rows"])
        pub["ledger"] = [r for r in pub["ledger"] if r["row_id"] not in added]
        pub["ledger"] += copy.deepcopy(self.provenance["removed_ledger_rows"])
        added_docs = set(self.provenance["added_documents"])
        pub["invoices"] = [d for d in pub["invoices"] if d["doc_id"] not in added_docs]
        pub["payments"] = [d for d in pub["payments"] if d["doc_id"] not in added_docs]
        return pub

    # ---------------------------------------------------------------- files
    def public_files(self) -> Dict[str, str]:
        return {
            "manifest.json": json.dumps(self.manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            "policy.json": json.dumps(self.policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            "ledger.csv": ledger_csv(self.ledger),
            "invoices.jsonl": jsonl(self.invoices),
            "payments.jsonl": jsonl(self.payments),
            "fx.csv": fx_csv(self.fx),
        }

    def private_files(self) -> Dict[str, str]:
        return {
            "truth.json": json.dumps(self.truth, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            "provenance.json": json.dumps(self.provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        }

    def public_sha256(self) -> str:
        h = hashlib.sha256()
        for name, text in sorted(self.public_files().items()):
            h.update(name.encode())
            h.update(b"\0")
            h.update(text.encode("utf-8"))
        return h.hexdigest()

    def full_sha256(self) -> str:
        h = hashlib.sha256(self.public_sha256().encode())
        for name, text in sorted(self.private_files().items()):
            h.update(name.encode())
            h.update(text.encode("utf-8"))
        return h.hexdigest()

    def write(self, directory: str | Path) -> Path:
        d = Path(directory)
        (d / "public").mkdir(parents=True, exist_ok=True)
        (d / "private").mkdir(parents=True, exist_ok=True)
        for name, text in self.public_files().items():
            (d / "public" / name).write_text(text, encoding="utf-8")
        for name, text in self.private_files().items():
            (d / "private" / name).write_text(text, encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "Episode":
        """Load an episode directory: ``public/`` + ``private/`` (as written by ``write``) or all eight
        files directly in ``directory``. provenance.json is optional for playing and grading."""
        d = Path(directory)
        pub = d / "public" if (d / "public" / "manifest.json").exists() else d
        priv = d / "private" if (d / "private" / "truth.json").exists() else d
        prov_path = priv / "provenance.json"

        def read(p: Path) -> str:
            return p.read_text(encoding="utf-8")

        return cls(
            manifest=json.loads(read(pub / "manifest.json")),
            policy=json.loads(read(pub / "policy.json")),
            ledger=parse_ledger_csv(read(pub / "ledger.csv")),
            invoices=parse_jsonl(read(pub / "invoices.jsonl")),
            payments=parse_jsonl(read(pub / "payments.jsonl")),
            fx=parse_fx_csv(read(pub / "fx.csv")),
            truth=json.loads(read(priv / "truth.json")),
            provenance=json.loads(read(prov_path)) if prov_path.exists() else {},
        )


def load_public(directory: str | Path) -> Dict[str, Any]:
    """Read only the public part of an exported episode (what an auditor may see)."""
    pub = Path(directory) / "public"
    if not (pub / "manifest.json").exists():
        pub = Path(directory)
    return {
        "manifest": json.loads((pub / "manifest.json").read_text(encoding="utf-8")),
        "policy": json.loads((pub / "policy.json").read_text(encoding="utf-8")),
        "ledger": parse_ledger_csv((pub / "ledger.csv").read_text(encoding="utf-8")),
        "invoices": parse_jsonl((pub / "invoices.jsonl").read_text(encoding="utf-8")),
        "payments": parse_jsonl((pub / "payments.jsonl").read_text(encoding="utf-8")),
        "fx": parse_fx_csv((pub / "fx.csv").read_text(encoding="utf-8")),
    }
