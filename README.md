
# 🚀 E-Scooter Nachfrage Heatmap

Interaktive D3.js-Visualisierung + Machine Learning Vorhersage (XGBoost) der E-Scooter-Nachfrage in Berliner Stadtteilen.

---

## 📁 Projektstruktur

```text
scooter-crm/
├── backend/
│   ├── ml_api.py              # FastAPI-Backend mit ML-Modell
│   └── xgb_scooter_model.joblib  # Vortrainiertes XGBoost-Modell
├── frontend/
│   └── index.html             # D3.js-Web-Interface
```

---

## 🧑‍💻 Voraussetzungen

* Python 3.10+
* Node.js **nicht notwendig**
* [Homebrew](https://brew.sh/) (für macOS, zum Installieren von OpenMP)
* (empfohlen) virtuelles Python-Environment

---

## 🔧 Setup

### 1. Repository klonen

```bash
git clone <repo-url>
cd scooter-crm
```

### 2. Python-Umgebung aufsetzen

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Falls keine `requirements.txt` vorhanden:

```bash
pip install fastapi uvicorn scikit-learn pandas xgboost joblib
```

### 3. (macOS) XGBoost-Fehler beheben

Installiere OpenMP, falls noch nicht vorhanden:

```bash
brew install libomp
```

---

## ▶️ Backend starten (ML-API)

```bash
uvicorn ml_api:app --reload
```

Läuft unter: `http://localhost:8000/predict`

---

## 🌐 Frontend starten

### Option A: mit Live Server (VS Code)

* Öffne `frontend/index.html` in VS Code
* Rechtsklick → **"Open with Live Server"**

### Option B: über Python HTTP-Server

```bash
cd frontend
python3 -m http.server 5500
```

Dann im Browser öffnen: [http://localhost:5500](http://localhost:5500)

---

## ✅ Features

* Nachfrage-Vorhersage pro Stadtteil anhand:

  * Uhrzeit
  * Wetter
  * Datum (inkl. Feiertagserkennung)
* D3.js-Heatmap mit Zoom & Tooltip
* Datenquelle: eigenes ML-Modell (`xgb_scooter_model.joblib`)
