
# AI Disease Prediction Using XGBoost Algorithm

A beginner-friendly Flask health tracking and educational screening project.

## Important medical/model note

This project is for education and health-information organization. It is **not a medical diagnostic system**.

The included web app can run without trained model files. Its displayed risk cards are explicitly labeled as **Educational demonstration estimates — not clinically validated**.

`train_models.py` creates synthetic XGBoost models only to demonstrate the project pipeline. Synthetic data is not a substitute for a validated clinical dataset. Do not use the resulting models for medical decisions and do not claim accuracy from them.

The visible form intentionally does **not** ask for Pregnancy or Glucose.

## Features

- Age, weight, height and automatic BMI
- Optional BP, heart rate, temperature, sleep, water, activity, stress and smoking
- 20+ symptoms with severity/duration/location/description fields
- Educational screening estimate cards
- "Why did I get this result?" explanation
- Daily Health Journal stored in SQLite
- Calendar with recorded dates
- Previous-date record viewing
- Trend charts for weight/BMI/BP/heart rate/temperature/sleep/water/activity/stress
- My Health Summary
- "Tell us about yourself"
- Friendly local health assistant with history-aware context
- Responsive brown/hazel UI
- No API key required

## Run on Windows

1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal.
4. Create a virtual environment:

```powershell
python -m venv .venv
```

5. Activate it:

```powershell
.venv\Scripts\activate
```

6. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

7. Optional: create synthetic demonstration XGBoost model files:

```powershell
python train_models.py
```

This step is optional for the current UI and is clearly labeled as synthetic educational modeling.

8. Start Flask:

```powershell
python app.py
```

9. Open:

`http://127.0.0.1:5000`

To stop Flask, press **Ctrl + C** in the terminal.

## SQLite

The database file `health_journal.db` is created automatically when the app starts. It is intentionally ignored by Git because it can contain personal health entries.

## GitHub

Do not upload `.venv` or the SQLite database.

```powershell
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Later:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI_Disease_Prediction_XGBoost_Complete
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Project structure

```text
AI_Disease_Prediction_XGBoost_Complete/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── models/
│   └── README.txt
└── data/
    └── .gitkeep
```

## Enhanced version
This copy adds a more context-aware local health assistant and lets the journal explicitly use any manually selected date and time, including past dates. The original ZIP is unchanged.

The assistant is intentionally offline and educational; it is not a medical diagnostic system. It uses the latest saved record and recent chat context to answer common questions and apply safety-oriented responses.
