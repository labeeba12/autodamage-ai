# AutoDamage AI

> Intelligent car damage detection and repair cost estimation for the Pakistani automotive market.

AutoDamage AI analyzes a single car photo, detects damage and affected parts, grades severity, and estimates repair cost ranges in PKR using YOLO segmentation and an XGBoost cost engine.

---

## Features

- **Damage detection** — detects 6 damage types: dent, scratch, crack, glass shatter, lamp broken, tire flat
- **Parts segmentation** — identifies 22 car parts with pixel-level masks
- **Car model selection** — user selects a Pakistani vehicle model from the frontend dropdown for cost estimation
- **Cost estimation** — repair vs replace recommendation with PKR cost ranges
- **Paint detection** — estimates paint color, paint code, and repaint cost range
- **Damage heatmap** — generates a top-down heatmap image showing damage zones
- **Visual report** — creates before/after comparison images and returns structured JSON output

---

## AI Models

| Model | Architecture | Task | Classes | Notes |
|-------|-------------|------|---------|-------|
| Parts Segmentation | YOLOv11n-seg | Instance segmentation | 22 parts | Detects vehicle body parts and masks |
| Damage Detection | YOLOv11m-seg | Instance segmentation | 6 types | Detects visible damage regions |
| Cost Estimation | XGBoost | Regression | N/A | Estimates damage repair/replacement cost |


---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Models | Ultralytics YOLOv11 + XGBoost |
| Backend | FastAPI + Python 3.10 |
| Image Processing | OpenCV + NumPy |
| Cost Estimation | XGBoost + pandas |
| Frontend | HTML + CSS + JavaScript |
| Environment | Python virtualenv |

---

## Project Structure

```
autodamage-ai/
├── api/
│   ├── main.py          # FastAPI app + endpoints
│   └── schemas.py       # Pydantic response models
├── src/
│   ├── detection/
│   │   ├── segmenter.py     # YOLO segmentation inference
│   │   └── validator.py     # image validation logic
│   ├── logic/
│   │   ├── feature_builder.py  # builds XGBoost feature vectors
│   │   ├── overlap_engine.py   # matches damage to parts
│   │   └── severity_calc.py    # severity grading
│   ├── cost/
│   │   ├── cost_regressor.py   # loads XGBoost model
│   │   ├── decision_engine.py  # repair/replace and cost rules
│   │   ├── paint_estimator.py  # paint cost lookup
│   │   └── price_lookup.py     # baseline pricing data
│   └── utils/
│       ├── visualizer.py      # before/after image generator
│       ├── heatmap_generator.py
│       ├── color_detector.py
│       └── logger.py
├── frontend/
│   ├── index.html         # Landing page
│   ├── analyzer.html      # Main analyzer UI
│   ├── about.html         # About page
│   └── app.js             # Frontend logic
├── models/                # Trained weights and saved XGBoost model
├── data/
│   ├── baseline_prices.csv
│   └── paint_codes.csv
├── tests/                 # Unit tests
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
dcd autodamage-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
source venv/bin/activate      # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model files
Download the trained model files and place them into the `models/` folder:

```
models/
├── car_parts_seg.pt
├── car_damage_seg.pt
├── xgboost_cost_model.json
```

> Model files are not included in the repository due to size.

### 5. Run the API
```bash
python -m api.main
```

### 6. Open the frontend
Open `frontend/index.html` in your browser, or use `frontend/analyzer.html`.

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## API Reference

### `POST /estimate`
Uploads a car image and returns a structured damage report.

**Request**: `multipart/form-data` containing:
- `file`: image JPEG/PNG/WEBP
- `car_model`: selected model value from the frontend dropdown

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
      "pixel_area": 4521,
      "area_ratio": 0.032,
      "overlap_ratio": 0.91,
      "min_cost": 14300,
      "max_cost": 57200
    }
  ],
  "result_image_path": "temp_outputs/result_abcdef12.jpg",
  "heatmap_path": "temp_outputs/heatmap_abcdef12.jpg",
  "paint_color": "Silver",
  "paint_code": "metallic",
  "paint_hex": "#C0C0C0",
  "paint_cost_min": 5000,
  "paint_cost_max": 12000,
  "car_model": "Toyota_corolla_2016",
  "car_model_confidence": 1.0,
  "car_model_top3": []
}
```

### `GET /health`
Returns API status and version.

### `GET /result-image/{filename}`
Returns the generated before/after comparison image.

### `GET /heatmap/{filename}`
Returns the generated damage heatmap image.

---

## Pipeline

```
Image upload
    ↓
Image validation (resolution + blur check)
    ↓
2-model inference (parts segmentation + damage segmentation)
    ↓
Damage-to-part matching (IoU + proximity fallback)
    ↓
Severity calculation (area-based grading)
    ↓
Cost estimation (rule-based + XGBoost regression)
    ↓
Visualization generation (before/after + heatmap)
    ↓
JSON damage report output
```

---

## Running Tests

```bash
pytest tests/ -v
```

The suite covers:
- severity calculation
- decision engine (repair vs replace)
- overlap engine (damage-part matching)
- image validation
- pipeline components

---

## Supported Car Models

The frontend offers a dropdown for 23 Pakistani models, including:

| Brand | Models |
|-------|--------|
| Suzuki | Mehran, Alto 2007, Alto 2019, Cultus 2018, Cultus 2019, Swift, WagonR 2015, Highroof, Kyber |
| Toyota | Corolla 2007, Corolla 2011, Corolla 2016, Aqua, Vitz, Passo, Hiace 2012, Vigo |
| Honda | City Aspire, City 2000, Civic 2007, Civic 2018 |
| Daihatsu | Cuore, Mira |

---

## Known Limitations

- Car model is selected manually from the frontend dropdown rather than being automatically classified.
- Parts segmentation works best on standard front/side/rear views; very angled shots may fall back to proximity matching.
- Cost estimates are approximations based on baseline pricing data and may differ from actual workshop quotes.

---

## Team

- Labeeba Munawar
- Tehreem Raouf

---

## License

This project was developed as a Final Year Project. All rights reserved.
