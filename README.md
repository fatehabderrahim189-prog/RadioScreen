# RadioScreen

**An educational AI prototype for chest X-ray screening.**
Author: Fateh

> ⚕️ **Disclaimer:** This is an educational and research prototype only.
> It is **not** a certified medical device and **must not** be used for real
> clinical diagnosis. It does not replace a licensed physician or radiologist.
> All data used here is synthetically generated.

## Why chest X-rays?
Widely available public research (e.g. NIH ChestX-ray14, CheXpert) and a
well-established academic baseline (CheXNet) make this a strong,
well-scoped area to learn applied medical AI without overreaching into
unvalidated clinical claims.

## What's actually in this repo
| Folder | What it is | Runs as-is? |
|---|---|---|
| `poc/` | Feature-based screening pipeline on **synthetic** data | ✅ Yes (`pip install scikit-learn numpy` then run) |
| `reference/` | DenseNet-121 transfer-learning design for a **real** dataset | 📄 Reference only — needs GPU + licensed dataset |
| `backend/` | FastAPI service serving the PoC model | ✅ Yes |
| `frontend/` | Clinical review interface (upload → score) | ✅ Yes (static HTML) |

## Quick start
```bash
# 1. Proof of concept (no server needed)
cd poc && pip install scikit-learn numpy && python3 radioscreen_poc.py

# 2. Backend API
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend
open frontend/index.html in a browser
```

## Roadmap to a real (non-demo) system
1. Sign a data-use agreement for a licensed public dataset.
2. Train `reference/train_cnn_reference.py` on real data with patient-level splits.
3. Evaluate with clinical metrics (sensitivity @ fixed specificity, calibration).
4. Clinical validation + regulatory review before any real-world use.

## License
MIT — see LICENSE. Note the license does not change the disclaimer above.
