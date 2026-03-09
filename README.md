# AutoDamage AI

An intelligent car damage detection and cost estimation system
built with YOLOv11 segmentation models and FastAPI.

## What It Does
- Detects car damage (scratch, dent, crack, glass shatter, lamp broken, tire flat)
- Identifies affected car parts (22 part classes)
- Maps damage to specific parts using mask overlap geometry
- Estimates repair/replace costs in PKR
- Classifies severity as Minor / Moderate / Severe

## Tech Stack
- **AI Models**: YOLOv11 Segmentation (Ultralytics)
- **Backend**: FastAPI + Python
- **Image Processing**: OpenCV
- **Cost Engine**: Pandas + XGBoost
- **Dataset**: CarDD (Car Damage Detection) + Custom Parts Dataset

## Project Structure
```
autodamage-ai/
├── api/           # FastAPI endpoints
├── src/
│   ├── detection/ # YOLO inference
│   ├── logic/     # Overlap + severity engine
│   ├── cost/      # Cost estimation
│   └── utils/     # Visualizer + logger
├── models/        # Trained YOLO weights
├── data/          # Pricing tables
└── tests/         # Unit tests
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/autodamage-ai.git
cd autodamage-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place your trained YOLO weights in the `models/` folder:
```
models/
├── car_parts_seg.pt
└── car_damage_seg.pt
```

### 5. Run the API
```bash
python api/main.py
```

### 6. Open API docs
Visit: http://localhost:8000/docs

## API Usage

### POST /estimate
Upload a car image to get damage report and cost estimate.

**Request**: multipart/form-data with image file

**Response**:
```json
{
  "success": true,
  "total_min": 33000,
  "total_max": 52000,
  "currency": "PKR",
  "overall_severity": "moderate",
  "total_damages": 3,
  "damages": [
    {
      "damage_type": "scratch",
      "part_name": "Front-bumper",
      "severity_label": "moderate",
      "action": "repair",
      "min_cost": 7500,
      "max_cost": 22500
    }
  ]
}
```

## Models
| Model | Architecture | Classes | mAP50 |
|-------|-------------|---------|-------|
| Car Parts | YOLOv11n-seg | 22 | 94.47% |
| Car Damage | YOLOv11m-seg | 6 | 76.3% |

## Team
- Developers: Labeeba Munawar & Tehreem Raouf
- Project: Final Year Project (FYP)
- Year: 2026