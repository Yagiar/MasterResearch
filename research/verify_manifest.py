# Проверка целостности research/manifest.json: пересчёт sha256 всех файлов.
# Ключи манифеста мапятся либо от корня репо, либо от MasterDiploma/ (см. make_manifest.py).
# Запуск от корня репо: research/.venv/bin/python research/verify_manifest.py
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "MasterDiploma"

def main() -> int:
    m = json.loads((ROOT / "research/manifest.json").read_text())
    bad, missing, ok = [], [], 0
    for key, meta in m["files"].items():
        path = next((c for c in (ROOT / key, MD / key) if c.exists()), None)
        if path is None:
            missing.append(key)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]:
            ok += 1
        else:
            bad.append(key)
    print(f"manifest: проверено {len(m['files'])}, ok {ok}, несовпадения {bad}, отсутствуют {missing}")
    return 0 if not bad and not missing else 1

if __name__ == "__main__":
    sys.exit(main())
