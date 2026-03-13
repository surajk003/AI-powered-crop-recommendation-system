import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np
import os
import json
import requests
import logging
from pathlib import Path

# ==========================
# 🌾 Load joblib model and preprocessor (RandomForest path)
# ==========================
MODEL_FILE = os.getenv("CROP_MODEL_FILE", "crop_model.joblib")
PRE_FILE = os.getenv("CROP_PRE_FILE", "crop_preprocessor.joblib")
META_FILE = os.getenv("CROP_META_FILE", "crop_model.joblib.meta.json")

# Read unique crop names from CSV to map indices to actual names
def load_crop_names():
    try:
        # Try to read first few rows to get header
        with open("Crop_recommendation.csv", "r") as f:
            import csv
            reader = csv.reader(f)
            next(reader)  # Skip header
            crops = set()
            for row in reader:
                if row:  # Ensure row has data
                    crops.add(row[-1].strip().title())  # Last column is label
            return {str(i): name for i, name in enumerate(sorted(crops))}
    except Exception:
        # Fallback crop names if CSV unavailable
        return {
            "0": "Cotton", "1": "Jute", "2": "Maize",
            "3": "Mothbeans", "4": "Mungbean", "5": "Pigeonpeas",
            "6": "Rice", "7": "Wheat", "8": "Apple",
            "9": "Banana", "10": "Black Gram", "11": "Chickpea",
            "12": "Coconut", "13": "Coffee", "14": "Grapes",
            "15": "Lentil", "16": "Mango", "17": "Orange",
            "18": "Papaya", "19": "Pomegranate", "20": "Sugarcane",
            "21": "Tea"
        }

# Load crop name mapping
CROP_NAMES = load_crop_names()

try:
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PRE_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("✅ Joblib model and preprocessor loaded successfully.")
except Exception as e:
    print(f"❌ Could not load model/preprocessor/meta: {e}")
    raise SystemExit(1)


# ==========================
# 🌦 Gemini Crop Details
# ==========================
def get_crop_details(crop_name):
    """Return simple local crop details (offline-friendly).

    This avoids calling external LLM/HTTP when running the packaged exe.
    The dictionary below is minimal; extend as needed.
    """
    local = {
        "0": {  # cotton
            "duration": "150-180 days",
            "environment": "Warm, humid climate; 21-30°C",
            "fertilizers": "NPK balanced; micronutrients important",
            "pests": "Bollworms, aphids, whiteflies",
        },
        "1": {  # jute
            "duration": "120-150 days",
            "environment": "Hot, humid climate; 25-35°C",
            "fertilizers": "Nitrogen-rich, organic matter",
            "pests": "Stem weevils, mites",
        },
        "2": {  # maize
            "duration": "70-110 days",
            "environment": "Warm, well-drained soils",
            "fertilizers": "Phosphorus at planting, Nitrogen split",
            "pests": "Stem borers, armyworms",
        },
        "3": {  # mothbeans
            "duration": "75-90 days",
            "environment": "Semi-arid regions; drought tolerant",
            "fertilizers": "Low requirement; phosphorus important",
            "pests": "Pod borers, leaf miners",
        },
        "4": {  # mungbean
            "duration": "60-75 days",
            "environment": "Warm climate; 25-35°C",
            "fertilizers": "Phosphorus and potassium",
            "pests": "Pod borers, whiteflies",
        },
        "5": {  # pigeonpeas
            "duration": "120-180 days",
            "environment": "Semi-arid tropics; 18-30°C",
            "fertilizers": "Phosphorus-based; minimal nitrogen",
            "pests": "Pod borers, pod flies",
        },
        "rice": {
            "duration": "120-150 days",
            "environment": "Warm, flooded fields; 20-35°C",
            "fertilizers": "Nitrogen-rich; split applications",
            "pests": "Stem borer, leaf folder",
        },
        "wheat": {
            "duration": "90-120 days",
            "environment": "Cooler climates; 10-25°C",
            "fertilizers": "Balanced NPK with micronutrients",
            "pests": "Aphids, rusts",
        },
        "maize": {
            "duration": "70-110 days",
            "environment": "Warm, well-drained soils",
            "fertilizers": "Phosphorus at planting, Nitrogen split",
            "pests": "Stem borers, armyworms",
        },
    }
    key = str(crop_name).strip().lower()
    # Try direct lookup first, then try as index
    details = local.get(key)
    if not details and key.isdigit():
        details = local.get(str(int(key)))
    if not details:
        return {"info": f"No local details for crop '{crop_name}'."}
    return details


def _read_local_config():
    cfg = {}
    # look for crop_config.json in current folder
    p = Path("crop_config.json")
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    return cfg


def call_gemini_for_crop(crop_name):
    """Call Gemini-like REST endpoint using an API key from env or crop_config.json.

    This function is best-effort: on any failure it returns a dict with an 'error'
    or falls back to returning {'info': text}.
    """
    logger = logging.getLogger("gemini")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("gemini_api.log")
    fh.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(fh)

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-pro")
    if not api_key:
        cfg = _read_local_config()
        api_key = cfg.get("GEMINI_API_KEY")
        model = cfg.get("GEMINI_MODEL", model)

    if not api_key or api_key == "your-api-key-here":
        logger.debug("No valid Gemini API key found; skipping remote call.")
        return {"error": "Please add your Gemini API key to crop_config.json"}

    # Use newer Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    
    prompt = (
        f"You are an expert agronomist. Provide detailed information about the crop '{crop_name}' in JSON format with these fields:\n"
        "- duration: growing period and seasons\n"
        "- environment: ideal climate and conditions\n"
        "- fertilizers: recommended nutrients and timing\n"
        "- pests: common problems and solutions\n\n"
        "Format the response as valid JSON only, no other text."
    )

    body = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
        }
    }

    try:
        resp = requests.post(url, json=body, timeout=15)
        logger.debug(f"Gemini request status: {resp.status_code} body: {resp.text}")
        if resp.status_code != 200:
            return {"error": f"Gemini API error (HTTP {resp.status_code}): {resp.text}"}
    except Exception as e:
        logger.exception("Gemini HTTP request failed")
        return {"error": f"Gemini request failed: {e}"}

    try:
        d = resp.json()
    except Exception:
        return {"info": resp.text}

    # Extract text from Gemini v1 response format
    text = None
    try:
        if isinstance(d, dict):
            # Navigate the response structure
            candidates = d.get("candidates", [])
            if candidates and isinstance(candidates[0], dict):
                content = candidates[0].get("content", {})
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    if parts and isinstance(parts[0], dict):
                        text = parts[0].get("text", "")
            
            # Also try newer format with 'text' directly
            if not text and "text" in d:
                text = d["text"]
    except Exception as e:
        logger.exception("Failed to parse Gemini response")
        text = None

    if not text:
        # fallback: store raw json
        return {"info": json.dumps(d, indent=2)}

    # try to parse JSON from text
    try:
        return json.loads(text)
    except Exception:
        return {"info": text}


def format_crop_details(details):
    """Format Gemini JSON response for display."""
    if not details:
        return "No additional details available."
    if "error" in details:
        return details["error"]

    parts = []
    if "duration" in details:
        parts.append(f"🕒 Duration: {details['duration']}")
    if "environment" in details:
        parts.append(f"🌤 Environment: {details['environment']}")
    if "fertilizers" in details:
        parts.append(f"🌱 Fertilizers: {details['fertilizers']}")
    if "pests" in details:
        parts.append(f"🐛 Pests: {details['pests']}")
    if "info" in details:
        parts.append(f"ℹ️ {details['info']}")
    return "\n\n".join(parts)


# ==========================
# 🌼 Modern Themed App
# ==========================
class CropApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌾 Smart Crop Recommendation System")
        self.state("zoomed")  # Full screen on open
        self.configure(bg="#E8F5E9")

        # Create main frame
        self.main_frame = ttk.Frame(self, padding=30)
        self.main_frame.pack(expand=True, fill="both")

        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="🌱 AI-Powered Crop Recommendation System",
            font=("Segoe UI", 28, "bold"),
            anchor="center",
        )
        title_label.pack(pady=(0, 20))

        # Frame for input fields
        self.input_frame = ttk.LabelFrame(
            self.main_frame,
            text="Enter Soil & Climate Parameters",
            padding=20,
            labelanchor="n",
        )
        self.input_frame.pack(padx=40, pady=20, fill="x")

        self.feature_names = meta.get(
            "features", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        )
        self.entries = {}

        # Layout: 3 columns per row
        for i, feat in enumerate(self.feature_names):
            ttk.Label(
                self.input_frame,
                text=f"{feat}:",
                font=("Segoe UI", 12)
            ).grid(row=i // 3, column=(i % 3) * 2, sticky="e", padx=10, pady=10)

            entry = ttk.Entry(self.input_frame, width=20, font=("Segoe UI", 11))
            entry.grid(row=i // 3, column=(i % 3) * 2 + 1, padx=10, pady=10)
            self.entries[feat] = entry

        # Buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(pady=20)

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10)
        ttk.Button(btn_frame, text="🔍 Predict Crop", command=self.predict_crop).grid(
            row=0, column=0, padx=15
        )
        ttk.Button(btn_frame, text="🧹 Clear", command=self.clear_fields).grid(
            row=0, column=1, padx=15
        )
        ttk.Button(btn_frame, text="❌ Exit", command=self.destroy).grid(
            row=0, column=2, padx=15
        )

        # Gemini toggle (optional)
        self.use_gemini_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(
            btn_frame,
            text="Use Gemini for details",
            variable=self.use_gemini_var,
        )
        chk.grid(row=0, column=3, padx=15)

        # Prediction Result
        self.result_label = ttk.Label(
            self.main_frame,
            text="",
            font=("Segoe UI", 18, "bold"),
            foreground="#2E7D32",
            anchor="center",
        )
        self.result_label.pack(pady=10)

        # Details section
        details_frame = ttk.LabelFrame(
            self.main_frame,
            text="📘 Crop Details (Powered by Gemini AI)",
            padding=20,
            labelanchor="n",
        )
        details_frame.pack(padx=40, pady=10, fill="both", expand=True)

        self.details_box = tk.Text(
            details_frame,
            height=16,
            wrap="word",
            font=("Segoe UI", 12),
            bg="#FAFAFA",
            relief="flat",
        )
        self.details_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Footer
        footer = ttk.Label(
            self.main_frame,
            text="Developed with 💚 using TensorFlow, Gemini AI, and Tkinter",
            font=("Segoe UI", 10),
            foreground="gray",
            anchor="center",
        )
        footer.pack(pady=10)

    # =====================
    # Helper functions
    # =====================
    def clear_fields(self):
        for e in self.entries.values():
            e.delete(0, tk.END)
        self.result_label.config(text="")
        self.details_box.delete("1.0", tk.END)

    def predict_crop(self):
        try:
            data = {f: float(self.entries[f].get()) for f in self.feature_names}
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return

        df = pd.DataFrame([data])
        try:
            Xp = preprocessor.transform(df)
            preds = model.predict(Xp)
            if preds.ndim == 2:
                idx = int(np.argmax(preds, axis=1)[0])
            else:
                idx = int(preds[0])
            crop_id = meta.get("classes", [])[idx]
            # Get human-readable crop name
            crop_name = CROP_NAMES.get(str(crop_id), f"Crop {crop_id}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to predict: {e}")
            return

        self.result_label.config(text=f"🌾 Recommended Crop: {crop_name}")

        self.details_box.delete("1.0", tk.END)
        # Fetch details: prefer Gemini if user enabled it, otherwise local
        if self.use_gemini_var.get():
            self.details_box.insert(tk.END, "Fetching crop details from Gemini...\n")
            self.update_idletasks()
            details = call_gemini_for_crop(crop_name)  # Use readable name
            # if remote call failed, fallback to local
            if not details or "error" in details:
                # include a note about fallback
                fallback = get_crop_details(crop_id)  # Try by ID first
                if not fallback or "info" in fallback:
                    fallback = get_crop_details(crop_name)  # Try by name
                details = details if details and "error" not in details else {"info": "(Gemini failed, showing local info)\n\n" + json.dumps(fallback)}
            # If Gemini returned an 'info' string, try to parse or show it
            if isinstance(details, dict) and "info" in details:
                formatted = format_crop_details(details)
            else:
                formatted = format_crop_details(details)
        else:
            details = get_crop_details(crop_id)  # Try by ID first
            if not details or "info" in details:
                details = get_crop_details(crop_name)  # Try by name
            formatted = format_crop_details(details)
        self.details_box.delete("1.0", tk.END)
        self.details_box.insert(tk.END, formatted)


# ==========================
# 🚀 Run App
# ==========================
if __name__ == "__main__":
    app = CropApp()
    app.mainloop()
