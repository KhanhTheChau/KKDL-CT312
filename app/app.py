from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
from preprocess import DataPreprocessor, prepare_csv_df, validate_single_input

app = Flask(__name__)
app.secret_key = 'bean_prediction_secret_2025'


MODEL_DIR = './train/models'
TRAIN_CSV_PATH = './train/data/X_train.csv'


FEATURE_NAMES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
    'ShapeFactor3', 'ShapeFactor4', 'ShapeFactor5'
]

# Load models
models = {}
model_mapping = {
    'KNeighbors': 'knn_model.pkl',
    'NaiveBayes': 'naive_bayes_model.pkl',
    'LogisticRegression': 'logistic_regression_model.pkl',
    'RandomForest': 'random_forest_model.pkl',
    'SupportVectorMachine': 'svm_model.pkl'
}

for key, filename in model_mapping.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models[key] = pickle.load(f)
        print(f"[OK] Đã tải mô hình: {key}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy mô hình: {path}")


preprocessor = DataPreprocessor()
if os.path.exists(TRAIN_CSV_PATH):
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        preprocessor.fit(train_df, FEATURE_NAMES)
        print(f"[OK] Scaler đã fit với {len(FEATURE_NAMES)} cột.")
    except Exception as e:
        print(f"[LỖI] Không thể fit scaler: {e}")
else:
    print(f"[CẢNH BÁO] Không tìm thấy {TRAIN_CSV_PATH}. Scaler chưa được fit!")


@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    prediction_result = None
    data = {'model_name': 'RandomForest'}
    is_csv_upload_success = False
    csv_base64 = None
    auto_download = False
    download_filename = 'ket_qua_du_doan_hat_dau.csv'

    if request.method == 'POST':
        model_name = request.form.get('model_name', 'RandomForest')
        data['model_name'] = model_name
        selected_model = models.get(model_name)

        if not selected_model:
            message = f"<strong>LỖI:</strong> Mô hình '{model_name}' chưa được tải."
            return render_template('index.html', message=message, data=data)

        csv_file = request.files.get('csv_file')
        if csv_file and csv_file.filename.endswith('.csv'):
            try:
                csv_df = pd.read_csv(csv_file)
                processed_df = prepare_csv_df(csv_df, FEATURE_NAMES)

                if processed_df.empty:
                    message = "<strong>CẢNH BÁO:</strong> Tệp CSV không có dữ liệu hợp lệ (thiếu cột hoặc không phải số)."
                else:
                    X_scaled = preprocessor.transform_csv(processed_df)
                    predictions = selected_model.predict(X_scaled)
                    result_df = processed_df.copy()
                    result_df['Class_Predicted'] = predictions

                    # Tạo CSV và mã hóa base64
                    output = io.StringIO()
                    result_df.to_csv(output, index=False)
                    csv_str = output.getvalue()
                    csv_base64 = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
                    output.close()

                    is_csv_upload_success = True
                    auto_download = True  # Kích hoạt tự động tải
                    message = f"<strong>THÀNH CÔNG:</strong> Đã xử lý <strong>{len(result_df)}</strong> mẫu. Tải xuống tự động..."

            except Exception as e:
                message = f"<strong>LỖI:</strong> Xử lý CSV thất bại: {str(e)}"
        else:
            input_data = {feat: request.form.get(feat) for feat in FEATURE_NAMES}
            numeric_data, error = validate_single_input(input_data, FEATURE_NAMES)

            if error:
                message = f"<strong>LỖI:</strong> {error}"
            else:
                try:
                    scaled_input = preprocessor.transform_single(input_data)
                    prediction = selected_model.predict([scaled_input])[0]
                    prediction_result = str(prediction)
                    message = "<strong>THÀNH CÔNG:</strong> Dự đoán hoàn tất!"
                except Exception as e:
                    message = f"<strong>LỖI:</strong> Dự đoán thất bại: {str(e)}"

    template_data = {
        'message': message,
        'prediction_result': prediction_result,
        'data': data,
        'is_csv_upload_success': is_csv_upload_success,
        'csv_base64': csv_base64,
        'auto_download': auto_download,
        'download_filename': download_filename
    }

    return render_template('index.html', **template_data)



if __name__ == '__main__':
    os.makedirs('./train/data', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True)