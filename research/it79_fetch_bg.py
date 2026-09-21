"""it-79: отбор и загрузка 400 HD-фонов Open Images validation (чистые, CC, без aerial).

CPU-only, вне git (данные в MasterDiploma/train/data/_prepared/hi-res-bg-v1/).
Детерминизм: сид 20260921; отбор = License~creativecommons ∧ ¬aerial-класс ∧ уникальность
OriginalMD5 (первое вхождение); пред-фильтр min-стороны по метаданным невозможен
(OriginalSize = размер файла, не пиксели — зарегистрированное уточнение к PLANNED),
поэтому размерный фильтр фактический (PIL).
Загрузка: ПОСЛЕДОВАТЕЛЬНАЯ, 0,4 с/запрос, на HTTP 429 — экспоненциальный backoff и повтор
того же URL (8-worker-версия сожгла очередь 429-ми; статусы final: accepted/small/too-big/
broken-img/dup; «missing/fail» при рестарте возвращаются в очередь).
"""
from __future__ import annotations

import csv
import random
import time
import urllib.error
import urllib.request
from hashlib import md5
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "MasterDiploma/train/data/_prepared/hi-res-bg-v1"
IMG = DATA / "images"
META = DATA / "_meta"
MANIFEST = DATA / "manifest.csv"
MIN_SIDE = 768
N_MAIN = 400
ATTEMPT_CAP = 41000  # весь пул
DELAY = 0.4
AERIAL_NAMES = [
    "Airplane", "Helicopter", "Drone", "Unmanned aerial vehicles", "UAV",
    "Kite", "Parachute", "Hot air balloon", "Balloon", "Model aircraft",
]
URLS_CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
BBOX_CSV = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
CLS_CSV = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
FINAL = ("accepted", "small", "too-big", "dup-local-md5", "surplus")


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1000:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    print(f"meta ok: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)


def get_bytes(url: str) -> tuple[bytes | None, str]:
    """None,reason: 'ok' → None байты не бывают; иначе тело или код ошибки."""
    back = 5.0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-fp-measure/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cl = r.headers.get("Content-Length")
                if cl and int(cl) > 25 * 1024 * 1024:
                    return None, "too-big"
                buf = r.read(25 * 1024 * 1024 + 1)
            if len(buf) > 25 * 1024 * 1024:
                return None, "too-big"
            return buf, "ok"
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None, "dead-404"
            if e.code == 429:
                time.sleep(back)
                back *= 4
                continue
            time.sleep(2)
            if attempt == 4:
                return None, f"http-{e.code}"
        except Exception as e:
            time.sleep(2)
            if attempt == 4:
                return None, f"fail:{type(e).__name__}"
    return None, "dead-429x5"


def main() -> None:
    fetch(URLS_CSV, META / "urls.csv")
    fetch(BBOX_CSV, META / "bbox.csv")
    fetch(CLS_CSV, META / "classes.csv")

    name2code = {}
    for row in csv.reader(open(META / "classes.csv", encoding="utf-8")):
        if len(row) >= 2:
            name2code[row[1].strip().lower()] = row[0]
    codes = {name2code[n.lower()] for n in AERIAL_NAMES if n.lower() in name2code}
    print("aerial codes:", sorted(codes), flush=True)
    banned, birds = set(), set()
    bird_code = name2code.get("bird", "__none__")
    for row in csv.reader(open(META / "bbox.csv", encoding="utf-8")):
        if len(row) < 3 or row[0] == "ImageID":
            continue
        if row[2] in codes:
            banned.add(row[0])
        elif row[2] == bird_code:
            birds.add(row[0])

    seen_md5, elig = set(), []
    for row in csv.DictReader(open(META / "urls.csv", encoding="utf-8")):
        iid, lic, m = row["ImageID"], row["License"], row["OriginalMD5"]
        if "creativecommons" not in lic.lower() or iid in banned or not m or m in seen_md5:
            continue
        seen_md5.add(m)
        elig.append((iid, row["OriginalURL"], lic))
    print(f"eligible (CC ∧ ¬aerial ∧ ¬dup-md5): {len(elig)}", flush=True)

    pool = elig[:]
    random.Random(20260921).shuffle(pool)
    by_id = {t[0]: t for t in pool}
    IMG.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    accepted: list[str] = []
    if MANIFEST.exists():
        for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            if any(row["status"].startswith(p) for p in FINAL) or row["status"].startswith("broken-img"):
                kept.append(row)
                if row["status"] == "accepted":
                    accepted.append(f"{row['ImageID']}.jpg")
    print(f"resume: принято {len(accepted)}, final-строк {len(kept)} (missing/fail возвращены в очередь)", flush=True)

    from PIL import Image
    seen_local_md5 = {md5((IMG / a).read_bytes()).hexdigest() for a in accepted if (IMG / a).exists()}
    processed = {r["ImageID"] for r in kept}
    t0 = time.time()
    for iid, url, lic in pool:
        if len(accepted) >= N_MAIN or len(processed) >= ATTEMPT_CAP:
            break
        if iid in processed:
            continue
        processed.add(iid)
        rec = {"ImageID": iid, "url": url, "license": lic, "status": "", "w": "", "h": "",
               "bird": int(iid in birds)}
        buf, st = get_bytes(url)
        time.sleep(DELAY)
        dst = IMG / f"{iid}.jpg"
        if st != "ok":
            rec["status"] = st
        else:
            dst.write_bytes(buf)
            try:
                with Image.open(dst) as im:
                    w, h = im.size
                    im.load()
                rec["w"], rec["h"] = w, h
                d = md5(buf).hexdigest()
                if min(w, h) < MIN_SIDE:
                    dst.unlink()
                    rec["status"] = "small"
                elif d in seen_local_md5:
                    dst.unlink()
                    rec["status"] = "dup-local-md5"
                elif len(accepted) >= N_MAIN:
                    dst.unlink()
                    rec["status"] = "surplus"
                else:
                    seen_local_md5.add(d)
                    accepted.append(dst.name)
                    rec["status"] = "accepted"
            except Exception as e:
                dst.unlink(missing_ok=True)
                rec["status"] = f"broken-img:{type(e).__name__}"
        kept.append(rec)
        if len(kept) % 100 == 0:
            with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
                mw = csv.DictWriter(mf, fieldnames=list(kept[0].keys()))
                mw.writeheader()
                mw.writerows(kept)
        if len(accepted) % 50 == 0 and rec["status"] == "accepted":
            print(f"принято {len(accepted)}/{N_MAIN} (попыток {len(processed)}, {time.time()-t0:.0f}s)", flush=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
        mw = csv.DictWriter(mf, fieldnames=list(kept[0].keys()))
        mw.writeheader()
        mw.writerows(kept)
    (DATA / "filelist.txt").write_text("\n".join(sorted(accepted)) + "\n", encoding="utf-8")
    dead = sum(r["status"].startswith(("dead-404", "http-", "fail:", "dead-429")) for r in kept)
    small = sum(r["status"] == "small" for r in kept)
    print(f"ACCEPTED={len(accepted)}/{N_MAIN} attempts={len(processed)} dead={dead} small={small}")
    print("filelist:", DATA / "filelist.txt")


if __name__ == "__main__":
    main()
