from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

app = Flask(__name__)

# Directories
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature config
CATEGORICAL_COLUMNS = ['Payment Method', 'Product Category', 'Customer Location', 'Device Used']
DROP_COLUMNS = ['Transaction ID', 'Customer ID', 'Transaction Date', 'IP Address', 'Shipping Address', 'Billing Address']
TARGET = 'Is Fraudulent'

# Preprocessing function
def preprocess_dataframe(df):
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csvFile' not in request.files:
        return render_template('result.html', error="No file uploaded.")

    file = request.files['csvFile']
    if not file or not file.filename.endswith('.csv'):
        return render_template('result.html', error="Invalid file type. Upload a .csv")

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_dataset.csv')
        file.save(file_path)

        # Load and preprocess
        df = pd.read_csv(file_path)
        df_original = df.copy()
        df_processed = preprocess_dataframe(df)

        if TARGET in df_processed.columns:
            df_processed.drop(columns=[TARGET], inplace=True)

        df_scaled = scaler.transform(df_processed)
        predictions = model.predict(df_scaled)

        # Append predictions
        label_map = {0: "Non-Fraud", 1: "Fraud"}
        df_original['Prediction'] = [label_map[p] for p in predictions]

        # Dataset analysis
        total = len(df_original)
        fraud_count = (df_original['Prediction'] == 'Fraud').sum()
        nonfraud_count = total - fraud_count
        fraud_percent = round((fraud_count / total) * 100, 2)

        # Generate heatmap
        heatmap_path = os.path.join(STATIC_FOLDER, 'correlation_dynamic.png')
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        # Generate feature importances
        importance_path = os.path.join(STATIC_FOLDER, 'feature_importance_dynamic.png')
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = df_processed.columns
            importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            importance_series.plot(kind='bar')
            plt.title("Feature Importances")
            plt.ylabel("Importance Score")
            plt.tight_layout()
            plt.savefig(importance_path)
            plt.close()

        return render_template(
            'result.html',
            table_headers=df_original.columns.tolist(),
            table_rows=df_original.to_dict(orient='records'),
            heatmap_path='/static/correlation_dynamic.png',
            importance_path='/static/feature_importance_dynamic.png',
            total=total,
            fraud_count=fraud_count,
            nonfraud_count=nonfraud_count,
            fraud_percent=fraud_percent
        )

    except Exception as e:
        traceback.print_exc()
        return render_template('result.html', error=str(e))

@app.route('/search', methods=['GET', 'POST'])
def search_data():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_dataset.csv')
    if not os.path.exists(file_path):
        return render_template('search.html', results=[], message="❌ No uploaded dataset found. Please upload and predict first.")

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return render_template('search.html', results=[], message="⚠️ Failed to read dataset. Please upload a valid CSV.")

    if request.method == 'POST':
        query = request.form.get('query')
        column = request.form.get('column')

        if not query:
            return render_template('search.html', results=[], message="Please enter a search term.")

        if column == "all":
            results = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
        else:
            if column not in df.columns:
                return render_template('search.html', results=[], message=f"Column '{column}' not found in dataset.")
            results = df[df[column].astype(str).str.contains(query, case=False, na=False)]

        if results.empty:
            return render_template('search.html', results=[], message="No matching records found.")

        return render_template('search.html', results=results.to_dict(orient='records'))

    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)
