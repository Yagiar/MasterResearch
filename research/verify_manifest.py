#!/usr/bin/env python3
"""Верификация research/manifest.json против диска: все files-хэши и gitignored_inputs.

Ключи `research/...` резолвятся от корня репо, прочие — от MasterDiploma/ (как в make_manifest.py).
Выход: построчно расхождения; итоговая строка «OK» с кодом 0 или «РАСХОЖДЕНИЯ» с кодом 1.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def resolve(key: str) -> Path:
    if key.startswith(("research/", "MasterDiploma/")):
        return ROOT / key
    return MD / key


def tree_digest(p: Path):
    """Эталонная свёртка как в make_manifest.py: «relative:sha256\\n» по сортированным файлам."""
    files = sorted(q for q in p.rglob("*") if q.is_file())
    agg = hashlib.sha256()
    for q in files:
        agg.update(f"{q.relative_to(p)}:{sha256(q)}\n".encode())
    return agg.hexdigest(), len(files)


def main() -> int:
    m = json.loads((ROOT / "research/manifest.json").read_text(encoding="utf-8"))
    bad = 0
    for key, ent in m["files"].items():
        p = resolve(key)
        if not p.is_file():
            print(f"РАСХОЖДЕНИЕ: {key}: отсутствует на диске")
            bad += 1
        elif sha256(p) != ent.get("sha256"):
            print(f"РАСХОЖДЕНИЕ: {key}: хэш диска != манифесту")
            bad += 1
    for key, ent in m.get("gitignored_inputs", {}).items():
        p = resolve(key)
        st = ent.get("state")
        if st == "absent":
            if p.exists():
                print(f"РАСХОЖДЕНИЕ: {key}: в манифесте absent, на диске есть")
                bad += 1
            continue
        if "tree_sha256" in ent:
            if not p.is_dir():
                print(f"РАСХОЖДЕНИЕ: {key}: каталог отсутствует")
                bad += 1
                continue
            dig, n = tree_digest(p)
            if dig != ent["tree_sha256"] or n != ent.get("files"):
                print(f"РАСХОЖДЕНИЕ: {key}: дерево != манифесту")
                bad += 1
        elif "sha256" in ent:
            if not p.is_file():
                print(f"РАСХОЖДЕНИЕ: {key}: файл отсутствует")
                bad += 1
            elif sha256(p) != ent["sha256"]:
                print(f"РАСХОЖДЕНИЕ: {key}: gitignored-хэш != манифесту")
                bad += 1
    print(f"OK: files {len(m['files'])}, gitignored_inputs {len(m.get('gitignored_inputs', {}))}, расхождений {bad}"
          if not bad else f"ИТОГО расхождений: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
