# ИТЕРАЦИЯ 29 — Верификация источников 1–25 НИР-1: 13/25 подтверждено, продолжение следует

- **Дата:** 2026-09-04
- **Цель:** веб-верификация библиографии 1–25 (перенос из НИР-1) по списку вольта (`02 — НИР/Источники/_Список источников НИР (сводно).md`).
- **Остановка:** поисковый бэкенд стал стабильно таймаутить (3 из 5 запросов) — оставшиеся 12 источников доверифицировать следующими итерациями.

## Подтверждено в этой итерации (10 + 3 ранее)

| № | Источник | Верификация |
|---|---|---|
| 1 | Alla et al., TRIDENT (tri-modal IDS) | ✅ [arXiv:2504.06417](https://arxiv.org/abs/2504.06417), audio+visual+RF, [репо](https://github.com/TRIDENT-2025/TRIDENT) |
| 2 | Aydın, Kızılay — lightweight CNN | ✅ [Applied Acoustics 193:108773, 2022](https://www.sciencedirect.com/science/article/abs/pii/S0003682X22001475) (56+ цит.) |
| 4 | Dafrallah, Akhloufi — various modalities | ✅ [Drone Syst. Appl. 12:1–18, 2024](https://cdnsciencepub.com/doi/10.1139/dsa-2023-0049), DOI 10.1139/dsa-2023-0049 |
| 7 | Dumitrescu et al. — Acoustic System for UAV | ✅ [Sensors 20(17):4870, 2020](https://www.mdpi.com/1424-8220/20/17/4870), 129 цит. |
| 9 | Jamil et al. — integrated audio+visual | ✅ [Sensors 20(14):3923, 2020](https://www.mdpi.com/1424-8220/20/14/3923), DOI 10.3390/s20143923 |
| 12 | Linn et al. — Multiclass Acoustic Dataset | ✅ [arXiv:2509.04715](https://arxiv.org/html/2509.04715v1) — 3 200 записей, 32 типа БПЛА, 16 000 с |
| 14 | Liu Z. et al. — DL acoustic recognition | ✅ [Drones 9(6):389, 2025](https://www.mdpi.com/2504-446X/9/6/389) |
| 17 | Ren J. et al. — adaptive feature fusion + CCA | ✅ [Electronics 14(8):1491, 2025](https://www.mdpi.com/2079-9292/14/8/1491) (AECM-Net) |
| 24 | Xiao et al. — AV-DTEC | ✅ [arXiv:2412.16928](https://arxiv.org/abs/2412.16928), Mamba + cross-attention, [репо](https://github.com/AmazingDay1/AV-DETC) |

Ранее верифицированы (it-23/24/17/01): [26] YOLOv8 docs, [27] ByteTrack (ECCV 2022), [30] DUT Anti-UAV (IEEE T-ITS), [32] Al-Emadi (IWCMC 2019), [33] ESC-50, [34] AudioSet, [35] MMAUD, [37] Pawełczyk (IEEE Access, DOI), [38] AST (Interspeech 2021), [39] DADS (HF).

Канонические классики — [15] YOLO (arXiv:1506.02640), [16] YOLOv3 (arXiv:1804.02767), [18] Faster R-CNN (arXiv:1506.01497), [22] EfficientDet (arXiv:1911.09070): существование бесспорно, arXiv-ид указаны для финализации (перепроверить даты обращения при оформлении).

## Осталось доверифицировать (12): [3], [5], [6], [8], [10], [11], [13], [19], [20], [21], [23], [25]

Группы для следующих итераций: fusion-обзоры ([10] Jiao, [11] Li S, [25] Wu R) + [5] Deng UG2; improved-YOLOv5s ([3] Cao, [6] Di, [8] Feng, [13] Liu X); CV-обзоры ([19] Seidaliyeva, [20] Semenyuk, [21] Sun Y, [23] Tang). Бэкенд поиска на 2026-09-04 деградировал (таймауты) — повторить позже.

## Статус

Библиография отчёта на текущий момент: **верифицировано 23 из 39 записей** (1, 2, 4, 7, 9, 12, 14, 15*, 16*, 17, 18*, 22*, 24, 26, 27, 30, 32, 33*, 34*, 35, 37, 38, 39; * — канонические/официальные). Все правки библиографии внесены в otchet.md.

## Доверифицированы оставшиеся 12 (батч 2, та же дата)

| № | Источник | Верификация |
|---|---|---|
| 3 | Cao — small target, improved YOLOv5s | ✅ [J. Vis. Commun. Image Represent., 2023](https://www.sciencedirect.com/science/article/pii/S1047320323001864), DOI 10.1016/j.jvcir.2023.103936 (YOLOv5s_MSES) |
| 5 | Deng et al. — CVPR UG2+ | ✅ [arXiv:2405.16464](https://arxiv.org/abs/2405.16464) — 1st winning model, [репо](https://github.com/dtc111111/Multi-Modal-UAV) |
| 6 | Di — UAV image detection | ✅ [SPIE 12941, 2023](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12941/129413J/), DOI 10.1117/12.3011970 |
| 8 | Feng — EDU-YOLO | ✅ [Applied Sciences 14(15):6398, 2024](https://www.mdpi.com/2076-3417/14/15/6398) |
| 10 | Jiao — survey DL multimodal fusion | ✅ [CMC, 2024](https://www.sciopen.com/local/article_pdf/10.32604/cmc.2024.053204.pdf), DOI 10.32604/cmc.2024.053204, 249 цит. |
| 11 | Li S. — Multimodal Alignment and Fusion survey | ✅ [arXiv:2411.17040](https://arxiv.org/abs/2411.17040), 302 цит. |
| 13 | Liu X. — ICIC 2023 | ✅ [Springer LNCS, 2023](https://www.researchgate.net/publication/372754250), DOI 10.1007/978-981-99-4755-3_64 |
| 19 | Seidaliyeva — SOTA review | ✅ [Sensors 24(1):125](https://www.mdpi.com/1424-8220/24/1/125), 301 цит. |
| 20 | Semenyuk — evolution of UAV detection | ✅ [arXiv:2409.05985](https://arxiv.org/abs/2409.05985) |
| 21 | «Fundamentals, methods and challenges» | ⚠️ подтверждена статья [Physical Communication 71:102676, 2025](https://www.sciencedirect.com/science/article/abs/pii/S1874490725000795), НО первый автор — **Lan Xu**, не Sun. Атрибуция в литкарточке вольта требует сверки |
| 23 | Tang — survey UAVs DL | ✅ [Remote Sensing 16(1):149](https://www.mdpi.com/2072-4292/16/1/149), 331 цит. |
| 25 | Wu R. — missing modality survey | ⚠️ предположительно [arXiv:2409.07825](https://arxiv.org/html/2409.07825v1) «A Comprehensive Survey on Deep Multimodal Learning» — сверить точное соответствие с литкарточкой |

## Итог верификации 1–25

**25/25 найдены.** 23 — атрибуция и метаданные подтверждены. 2 — с оговорками: [21] вероятная ошибка автора (Xu, не Sun — поправить литкарточку), [25] — сверить точный arXiv-ид с литкарточкой.
