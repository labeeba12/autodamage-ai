# AutoDamage AI

> Intelligent car damage detection and repair cost estimation — built for the Pakistani automotive market.

AutoDamage AI uses three fine-tuned YOLOv11 models to automatically detect vehicle damage, identify affected parts, classify the car model, and estimate repair costs in PKR from a single photo.

---

## Features

- **Damage Detection** — Detects 6 damage types: dent, scratch, crack, glass shatter, lamp broken, tire flat
- **Parts Segmentation** — Identifies and segments 22 car parts with pixel-level masks
- **Car Model Recognition** — Recognizes 23 Pakistani car models (Suzuki, Toyota, Honda, Daihatsu)
- **Cost Estimation** — Repair vs replace recommendation with PKR cost ranges
- **Paint Detection** — Detects paint color and finish type with repainting cost
- **Damage Heatmap** — Top-down car diagram showing damaged zones by severity
- **Print Report** — Generate a professional PDF damage report

---

## AI Models

| Model | Architecture | Task | Classes | Performance |
|-------|-------------|------|---------|-------------|
| Parts Segmentation | YOLOv11n-seg | Instance Segmentation | 22 parts | **94.47% Mask mAP50** |
| Damage Detection | YOLOv11m-seg | Instance Segmentation | 6 types | **77.7% Mask mAP50** |
| Car Classifier | YOLOv11n-cls | Image Classification | 23 models | **98.4% Top-1 Accuracy** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Models | Ultralytics YOLOv11 |
| Backend | FastAPI + Python 3.10 |
| Image Processing | OpenCV + NumPy |
| Cost Estimation | XGBoost + Pandas |
| Frontend | HTML + CSS + JavaScript |
| Training | Google Colab (T4 GPU) |

---

## Project Structure

```
autodamage-ai/
├── api/
│   ├── main.py          # FastAPI app + endpoints
│   └── schemas.py       # Pydantic response models
├── src/
│   ├── detection/
│   │   ├── segmenter.py     # YOLO inference (all 3 models)
│   │   └── validator.py     # Image validation
│   ├── logic/
│   │   ├── overlap_engine.py  # Damage-to-part matching (IoU + proximity)
│   │   └── severity_calc.py   # Severity grading
│   ├── cost/
│   │   ├── decision_engine.py # Repair vs replace logic
│   │   ├── paint_estimator.py # Paint cost estimation
│   │   └── xgboost_regressor.py
│   └── utils/
│       ├── visualizer.py      # Before/after image
│       ├── heatmap_generator.py
│       ├── color_detector.py
│       └── logger.py
├── frontend/
│   ├── index.html         # Landing page
│   ├── analyzer.html      # Main analyzer UI
│   ├── about.html         # About page
│   └── app.js             # Frontend logic
├── models/                # Trained YOLO weights (not in repo)
├── data/
│   └── baseline_prices.csv  # PKR pricing table (51 entries)
├── tests/                 # Unit tests (43 passing)
├── config.py
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Git Bash (Windows) or Terminal (Mac/Linux)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/autodamage-ai.git
cd autodamage-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
source venv/bin/activate       # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Download trained weights and place in `models/` folder:
```
models/
├── car_parts_seg.pt      # YOLOv11n-seg — 22 car parts
├── car_damage_seg.pt     # YOLOv11m-seg — 6 damage types
├── car_classifier.pt     # YOLOv11n-cls — 23 Pakistani car models
└── xgboost_cost_model.json
```

> Model weights are not included in the repository due to file size. Contact the team for access.

### 5. Run the API
```bash
python -m api.main
```

### 6. Open the frontend
Open `frontend/index.html` in your browser, or visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## API Reference

### `POST /estimate`
Upload a car image to get a complete damage report.

**Request**: `multipart/form-data` with `file` field (JPG/PNG/WEBP)

**Response**:
```json
{
  "success": true,
  "total_min": 14300,
  "total_max": 57200,
  "currency": "PKR",
  "overall_severity": "severe",
  "total_damages": 2,
  "damages": [
    {
      "damage_type": "dent",
      "part_name": "Front-bumper",
      "severity_label": "severe",
      "action": "replace",
      "confidence": 0.92,
      "min_cost": 14300,
      "max_cost": 57200
    }
  ],
  "car_model": "Toyota_corolla_2016",
  "car_model_confidence": 0.94,
  "paint_color": "Silver",
  "paint_code": "metallic",
  "paint_cost_min": 5000,
  "paint_cost_max": 12000
}
```

### `GET /health`
Returns API status.

### `GET /result-image/{filename}`
Returns the before/after comparison image.

### `GET /heatmap/{filename}`
Returns the damage heatmap image.

---

## Pipeline

```
Image Upload
    ↓
Image Validation (resolution + blur check)
    ↓
3-Model Inference (parts seg + damage seg + classifier)
    ↓
Car Validation (confidence + parts check)
    ↓
Overlap Engine (IoU matching + proximity fallback)
    ↓
Severity Calculation (area-based grading)
    ↓
Cost Estimation (XGBoost + rule-based pricing)
    ↓
Report Generation (heatmap + before/after + JSON)
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 43 tests pass covering:
- Severity calculation
- Decision engine (repair vs replace)
- Overlap engine (IoU + matching)
- Image validation
- Pipeline components

---

## Supported Car Models

23 Pakistani car models including:

| Brand | Models |
|-------|--------|
| Suzuki | Mehran, Alto 2007, Alto 2019, Cultus 2018, Cultus 2019, Swift, WagonR 2015, Highroof, Kyber |
| Toyota | Corolla 2007, Corolla 2011, Corolla 2016, Aqua, Vitz, Passo, Hiace 2012, Vigo |
| Honda | City Aspire, City 2000, Civic 2007, Civic 2018 |
| Daihatsu | Core, Mira |

---

## Known Limitations

- Parts model performs best on standard front/side/rear views. Angled shots use proximity-based fallback matching.
- Car classifier trained on 23 Pakistani models — vehicles outside this set will match to the closest known model.
- Cost estimates are approximations based on baseline pricing data. Actual workshop prices may vary.

---

## Team


1) Labeeba Munawar
2) Tehreem Raouf 

---

## License

This project was developed as a Final Year Project. All rights reserved.