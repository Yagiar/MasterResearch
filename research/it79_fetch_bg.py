"""it-79: отбор и загрузка 400 HD-фонов Open Images validation (чистые, CC, без aerial).

CPU-only, вне git (данные в MasterDiploma/train/data/_prepared/hi-res-bg-v1/).
Детерминизм: сид 20260921; отбор = License~creativecommons ∧ ¬aerial-класс ∧ уникальность
OriginalMD5 (первое вхождение); пред-фильтр min-стороны по метаданным невозможен
(OriginalSize = размер файла, не пиксели — зарегистрированное уточнение к PLANNED),
поэтому размерный фильтр фактический (PIL).
Загрузка: ПОСЛЕДОВАТЕЛЬНАЯ, 0,4 с/запрос, через curl-субпроцесс (edge Flickr персистентно
троттлит python-urllib 429-ми; curl в том же окне даёт 200), на 429 — мягкий бэкофф 10/20/30 с
и повтор того же URL (8-worker-версия сожгла очередь 429-ми; статусы final: accepted/small/
too-big/dup/dead-404/broken-img; «missing/fail» при рестарте возвращаются в очередь).
"""
from __future__ import annotations

import csv
import random
import subprocess
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
DELAY = 4.0  # run-6: сонрежим против Flickr-бана (апробация 1 запросом в 6 с)
AERIAL_NAMES = [
    "Airplane", "Helicopter", "Drone", "Unmanned aerial vehicles", "UAV",
    "Kite", "Parachute", "Hot air balloon", "Balloon", "Model aircraft",
]
URLS_CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
BBOX_CSV = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
CLS_CSV = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
FINAL = ("accepted", "small", "too-big", "dup-local-md5", "surplus")
DEAD = DATA / "dead.txt"


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
    """Закачка через curl-субпроцесс: edge Flickr троттлит python-urllib персистентными 429
    (подтверждено 21.09: urllib в бэкоффе стоит минуты, curl в этом же окне даёт 200 за 0,2 с).
    Мягкий арифметический бэкофф 10/20/30 с; >25 MiB → too-big (--max-filesize, rc=63)."""
    tmpf = DATA / "_curl.tmp"
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-s", "-m", "40", "-A", "research-fp-measure/1.0",
             "--max-filesize", str(25 * 1024 * 1024), "-o", str(tmpf), "-w", "%{http_code}", url],
            capture_output=True, text=True)
        code = r.stdout.strip()
        if code == "200" and r.returncode == 0:
            buf = tmpf.read_bytes()
            tmpf.unlink(missing_ok=True)
            if len(buf) > 25 * 1024 * 1024:
                return None, "too-big"
            return buf, "ok"
        tmpf.unlink(missing_ok=True)
        if r.returncode == 63:
            return None, "too-big"
        if code in ("404", "410"):
            return None, "dead-404"
        if code == "429":
            print(f"429: {url.split('/')[-1][:24]} (попытка {attempt + 1})", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        time.sleep(2)
        if attempt == 3:
            return None, f"http-{code or r.returncode}"
    return None, "dead-429x4"


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
    dead_ids: set[str] = set()
    if DEAD.exists():
        dead_ids = {l.strip() for l in open(DEAD, encoding="utf-8") if l.strip()}
    retry_ids: set[str] = set()
    if MANIFEST.exists():
        for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            if any(row["status"].startswith(p) for p in FINAL) or row["status"].startswith("broken-img"):
                kept.append(row)
                if row["status"] == "accepted":
                    accepted.append(f"{row['ImageID']}.jpg")
            elif row["status"] in ("missing", "dead-404", "http-404", "http-410"):
                # curl-подтверждённые 404 (21.09): не дёргаем второй раз
                dead_ids.add(row["ImageID"])
            else:
                retry_ids.add(row["ImageID"])
    print(f"resume: принято {len(accepted)}, final-строк {len(kept)}, dead-подтверждённых {len(dead_ids)} "
          f"(missing/fail возвращены в очередь: {len(retry_ids)})", flush=True)

    from PIL import Image
    seen_local_md5 = {md5((IMG / a).read_bytes()).hexdigest() for a in accepted if (IMG / a).exists()}
    processed = {r["ImageID"] for r in kept}
    t0 = time.time()
    bad_streak = 0
    for iid, url, lic in pool:
        if len(accepted) >= N_MAIN or len(processed) >= ATTEMPT_CAP:
            break
        if iid in processed:
            continue
        processed.add(iid)
        if iid in dead_ids:
            kept.append({"ImageID": iid, "url": url, "license": lic, "status": "dead-confirmed",
                         "w": "", "h": "", "bird": int(iid in birds)})
            continue
        rec = {"ImageID": iid, "url": url, "license": lic, "status": "", "w": "", "h": "",
               "bird": int(iid in birds)}
        buf, st = get_bytes(url)
        time.sleep(DELAY * random.uniform(0.8, 1.4))
        if st == "ok":
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= 15:
                print(f"circuit-open: 15 не-OK подряд, дрем 1200 с (принято {len(accepted)})", flush=True)
                time.sleep(1200)
                bad_streak = 0
        if st != "ok" and not st.startswith("dead-429") and iid in retry_ids:
            dead_ids.add(iid)  # смерть при второй проверке → больше не дёргаем
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
            DEAD.write_text("\n".join(sorted(dead_ids)) + "\n", encoding="utf-8")
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
    DEAD.write_text("\n".join(sorted(dead_ids)) + "\n", encoding="utf-8")
    (DATA / "filelist.txt").write_text("\n".join(sorted(accepted)) + "\n", encoding="utf-8")
    dead = sum(r["status"].startswith(("dead-404", "http-", "fail:", "dead-429")) for r in kept)
    small = sum(r["status"] == "small" for r in kept)
    print(f"ACCEPTED={len(accepted)}/{N_MAIN} attempts={len(processed)} dead={dead} small={small}")
    print("filelist:", DATA / "filelist.txt")


if __name__ == "__main__":
    main()
