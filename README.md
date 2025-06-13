🔹 Project Overview
The Fraud Detection Web Application is a machine learning-based system built using Python and Flask. It enables users to upload transaction data in CSV format, automatically identifies fraudulent transactions using a trained model, and presents results clearly through a web interface. Visual insights such as correlation heatmaps and feature importance charts are also generated.

🔹 Key Features
📁 CSV upload interface on the homepage

⚙️ Preprocessing: dropping irrelevant columns, encoding categorical features

📏 Scaling data using a saved StandardScaler

🧠 Prediction using a pre-trained ML model (fraud_detection_model.pkl)

🧾 Result table with fraud entries highlighted in red

🔍 Smart search with dropdown filter for column-specific queries

📊 Visual insights:

Correlation heatmap

Feature importance bar chart

Dataset summary:

Total transactions

Fraud count

Non-fraud count

Fraud rate (%)

🔹 Technologies Used
Python – Core development language

Flask – Backend web framework

Pandas – Data processing

Scikit-learn – ML model training and prediction

Matplotlib & Seaborn – Data visualizations

Jinja2 – HTML rendering via Flask

🔹 Model Details
The model is trained offline using Logistic Regression or Decision Tree.
It is saved as fraud_detection_model.pkl, and a corresponding scaler is saved as scaler.pkl. These files are loaded during prediction to maintain consistency in feature scaling and output accuracy.

🔹 Conclusion
This project is a complete, scalable, and user-friendly fraud detection solution.
It combines the power of machine learning with intuitive frontend design, making it well-suited for real-world financial and e-commerce fraud analysis.
