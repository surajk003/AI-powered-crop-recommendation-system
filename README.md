# 🌾 AI Powered Crop Recommendation System

An intelligent agriculture support system that recommends the **most suitable crop to grow based on soil nutrients and environmental conditions**.

The system uses a **Machine Learning Random Forest model** to analyze soil parameters such as **Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall**.

It provides farmers or agricultural researchers with **data-driven crop recommendations** through a simple **Tkinter graphical interface**.

---

# 🚀 Features

✔ AI-based crop recommendation
✔ Machine Learning prediction using Random Forest
✔ Soil nutrient analysis (N, P, K)
✔ Climate parameter analysis
✔ Interactive graphical interface
✔ Offline crop information system
✔ Optional Gemini AI crop details
✔ Fast prediction using trained ML model

---

# 🧠 Technologies Used

| Technology    | Purpose                           |
| ------------- | --------------------------------- |
| Python        | Main programming language         |
| Tkinter       | GUI development                   |
| Scikit-learn  | Machine learning model            |
| Random Forest | Crop prediction model             |
| Pandas        | Data processing                   |
| NumPy         | Numerical computation             |
| Joblib        | Model saving and loading          |
| Gemini AI API | Crop insights and recommendations |

---

# 🌱 Input Parameters

The model predicts the best crop using the following parameters:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

These parameters help determine the **optimal crop suitable for the given soil and climate conditions**.

---

# ⚙️ How the System Works

1️⃣ User enters soil nutrient values and environmental parameters.

2️⃣ The trained **Random Forest model** processes the input data.

3️⃣ The system predicts the **best crop to cultivate**.

4️⃣ The GUI displays the **recommended crop**.

5️⃣ The system also shows **additional crop information** such as:

* Growing duration
* Ideal environment
* Fertilizer requirements
* Common pests

6️⃣ Optionally, **Gemini AI** can generate more detailed crop insights.

---

# 🖥️ Application Interface

The GUI allows users to:

* Enter soil and climate parameters
* Predict the most suitable crop
* View crop cultivation details
* Clear input fields easily

---

# 📂 Project Structure

```
Crop_Recommendation_System
│
├── crop_model.joblib
├── crop_preprocessor.joblib
├── crop_model.joblib.meta.json
├── Crop_recommendation.csv
├── main.py
├── requirements.txt
├── README.md
```

---

# 📦 Installation

Clone the repository

```
git clone https://github.com/yourusername/Crop-Recommendation-System.git
```

Move to project directory

```
cd Crop-Recommendation-System
```

Install required libraries

```
pip install pandas numpy scikit-learn joblib requests
```

---

# ▶️ Run the Application

Run the Python program

```
python main.py
```

The **GUI interface will open**, allowing you to input soil parameters and get crop recommendations.

---

# 🌾 Example Crops Supported

The system can recommend crops such as:

* Rice
* Wheat
* Maize
* Cotton
* Coffee
* Mango
* Banana
* Pomegranate
* Chickpea
* Coconut
* Tea
* Grapes
* Papaya
* Lentil

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes**.
Actual crop selection should also consider **local agricultural expertise and field conditions**.

---

# 🎯 Future Improvements

* Real-time weather API integration
* Soil sensor IoT integration
* Mobile application interface
* Deep learning crop prediction models
* Farmer advisory system

---

# 👨‍💻 Author

**Suraj Konda**

B.Tech Student | AI | IoT | Smart Agriculture Enthusiast

---

# ⭐ Support

If you find this project helpful, please **star ⭐ this repository on GitHub**.
