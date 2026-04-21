# E-Waste Classification Web App

Professional Streamlit frontend for your trained e-waste image classifier (portfolio-ready UI).

## What this app includes

- Clean and modern UI (not "template-looking")
- Image upload + camera capture
- Predicted class + confidence score
- Class-wise confidence chart
- Smart fallback if model files are missing

## Project structure

- `app.py` - Streamlit frontend and inference flow
- `export_artifacts.py` - helper to save model and class names from notebook
- `requirements.txt` - Python dependencies
- `artifacts/` - model and labels for deployment

## 1) Save artifacts from your notebook

At the end of `Final_Submission_FiLe.ipynb`, run:

```python
from export_artifacts import save_artifacts
save_artifacts(model, class_names)
```

This creates:

- `artifacts/ewaste_classifier.keras`
- `artifacts/class_names.json`

## 2) Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) Fast deployment (free tier friendly)

Use Streamlit Community Cloud or Hugging Face Spaces:

1. Push this folder to GitHub.
2. Ensure `artifacts/ewaste_classifier.keras` and `artifacts/class_names.json` are present in repo.
3. Deploy app entrypoint: `app.py`.

## Demo tips for 10-minute runtime windows

- Keep app startup fast by using one optimized model (`.keras`).
- Avoid retraining in deployment environment.
- Keep one clear test image ready to show prediction instantly.
- Restart only when required, not between every small check.
