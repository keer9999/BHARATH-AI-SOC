# API Reference — Flask Web Application

## Base URL

```
http://<host>:5000
```

Default host: `localhost` (or the RPi 5 IP on your LAN)

---

## Authentication

The API has **no authentication by default** — it is designed for local/LAN use. If deploying externally, add an nginx reverse proxy with basic auth or token-based authentication.

---

## Endpoints

### GET `/`

Renders the main web dashboard.

**Response**: HTML page with live camera stream, chat interface, and analytics.

---

### GET `/video_feed`

MJPEG camera stream with real-time YOLOv8 detection overlays.

**Response Headers:**
```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

**Usage in HTML:**
```html
<img src="/video_feed" width="640" height="480" />
```

**Usage with VLC:**
```bash
vlc http://localhost:5000/video_feed
```

---

### GET `/api/status`

Returns system health, sensor readings, and camera status.

**Response: `200 OK`**
```json
{
  "status": "ok",
  "timestamp": "2026-11-14T10:30:00Z",
  "camera": {
    "connected": true,
    "device": 0,
    "resolution": "1280x720",
    "fps": 18.4
  },
  "sensors": {
    "pico_connected": true,
    "last_reading_age_s": 1.2,
    "distance_cm": 23.4,
    "fill_pct": 46.8,
    "temp_c": 28.1,
    "humidity_pct": 65.2,
    "gas_ppm": 142.3,
    "moisture_pct": 18.7
  },
  "model": {
    "loaded": true,
    "name": "yolov8n",
    "inference_ms": 54
  },
  "llm": {
    "loaded": true,
    "model": "llama-3.2-1b-instruct-q4_k_m.gguf"
  },
  "uptime_s": 3600
}
```

---

### GET `/api/detections`

Returns recent detection history from the database.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Maximum number of records to return |
| `offset` | int | 0 | Pagination offset |
| `class` | string | (all) | Filter by waste class: `plastic`, `glass`, `metal`, `paper`, `organic` |
| `since` | ISO 8601 | (all) | Filter detections after this timestamp |
| `session_id` | string | (all) | Filter by session ID |

**Response: `200 OK`**
```json
{
  "count": 2,
  "total": 1247,
  "detections": [
    {
      "id": 1247,
      "session_id": "sess_20261114_103000",
      "timestamp": "2026-11-14T10:30:00.123Z",
      "waste_class": "plastic",
      "confidence": 0.87,
      "bbox": {"x1": 120, "y1": 80, "x2": 340, "y2": 310},
      "sensor_snapshot": {
        "fill_pct": 46.8,
        "temp_c": 28.1,
        "gas_ppm": 142.3
      },
      "disposal_advice": "Place in the blue recycling bin. Rinse before disposal."
    },
    {
      "id": 1246,
      "session_id": "sess_20261114_103000",
      "timestamp": "2026-11-14T10:29:58.901Z",
      "waste_class": "organic",
      "confidence": 0.92,
      "bbox": {"x1": 50, "y1": 200, "x2": 180, "y2": 400},
      "sensor_snapshot": {
        "fill_pct": 46.8,
        "temp_c": 28.1,
        "gas_ppm": 142.3
      },
      "disposal_advice": "Compostable waste — place in the brown bin or compost pile."
    }
  ]
}
```

---

### GET `/api/stats`

Returns aggregated waste composition statistics.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `today` | `today`, `week`, `month`, `all` |
| `session_id` | string | (all) | Filter by session ID |

**Response: `200 OK`**
```json
{
  "period": "today",
  "total_detections": 342,
  "composition": {
    "plastic": {"count": 134, "pct": 39.2},
    "organic": {"count": 98, "pct": 28.7},
    "paper": {"count": 56, "pct": 16.4},
    "metal": {"count": 34, "pct": 9.9},
    "glass": {"count": 20, "pct": 5.8}
  },
  "avg_confidence": 0.83,
  "peak_hour": "12:00",
  "fill_level": {
    "current_pct": 46.8,
    "avg_pct": 38.2,
    "max_pct": 71.5
  },
  "alerts_triggered": 2
}
```

---

### POST `/api/chat`

Send a message to the Llama 3.2 1B on-device assistant for disposal guidance.

**Request Body (JSON):**
```json
{
  "message": "How should I dispose of an old lithium battery?",
  "context": {
    "detected_class": "metal",
    "confidence": 0.85
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | User's question (max 500 chars) |
| `context.detected_class` | string | ❌ | Most recently detected waste class |
| `context.confidence` | float | ❌ | Detection confidence |

**Response: `200 OK`**
```json
{
  "reply": "Lithium batteries are hazardous waste and should NOT go in regular recycling or trash bins. Please take them to a certified e-waste collection point or battery drop-off location. Many electronics stores and municipal waste centres accept them for free.",
  "inference_ms": 1240,
  "model": "llama-3.2-1b-instruct-q4_k_m"
}
```

**Error Response: `429 Too Many Requests`**
```json
{
  "error": "LLM busy — previous request still processing. Retry in 2s."
}
```

---

### GET `/api/report/pdf`

Generates and returns a PDF waste management report.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `today` | `today`, `week`, `month` |
| `session_id` | string | (all) | Filter by session |

**Response: `200 OK`**
```
Content-Type: application/pdf
Content-Disposition: attachment; filename="waste_report_20261114.pdf"
```

Returns PDF file containing:
- Waste composition pie chart
- Detection timeline
- Sensor readings summary
- Bin fill-level history
- Alert log

---

### GET `/api/report/csv`

Exports detection data as a CSV file.

**Query Parameters:** Same as `/api/report/pdf`

**Response: `200 OK`**
```
Content-Type: text/csv
Content-Disposition: attachment; filename="waste_detections_20261114.csv"
```

**CSV Columns:**
```
id,timestamp,session_id,waste_class,confidence,x1,y1,x2,y2,fill_pct,temp_c,humidity_pct,gas_ppm,moisture_pct
```

---

### POST `/api/alert/config`

Configure alert thresholds.

**Request Body (JSON):**
```json
{
  "fill_threshold_pct": 80,
  "gas_ppm_threshold": 300,
  "temp_c_threshold": 40,
  "email_alerts": true,
  "alert_email": "admin@example.com"
}
```

**Response: `200 OK`**
```json
{
  "status": "updated",
  "config": {
    "fill_threshold_pct": 80,
    "gas_ppm_threshold": 300,
    "temp_c_threshold": 40,
    "email_alerts": true,
    "alert_email": "admin@example.com"
  }
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `200` | OK |
| `400` | Bad Request — missing or invalid parameters |
| `404` | Not Found |
| `429` | Too Many Requests — LLM rate limited |
| `500` | Internal Server Error — check logs |
| `503` | Service Unavailable — model not loaded |

---

## CORS

CORS is not enabled by default. If accessing from a different origin:

```python
# In web_app.py, add:
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/chat` | 1 concurrent request (LLM is single-threaded) |
| `/api/report/pdf` | 1 concurrent generation |
| All others | No limit |

---

## Example: Python Client

```python
import requests

BASE = "http://raspberry-pi-5.local:5000"

# Get system status
status = requests.get(f"{BASE}/api/status").json()
print(f"Fill level: {status['sensors']['fill_pct']:.1f}%")

# Get today's stats
stats = requests.get(f"{BASE}/api/stats", params={"period": "today"}).json()
print(f"Most common: {max(stats['composition'], key=lambda k: stats['composition'][k]['count'])}")

# Ask the LLM
reply = requests.post(f"{BASE}/api/chat", json={
    "message": "What bin does plastic go in?",
    "context": {"detected_class": "plastic"}
}).json()
print(reply["reply"])
```

