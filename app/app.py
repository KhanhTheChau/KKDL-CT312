from flask import Flask, render_template, request, redirect, url_for, Response
import pickle 
import pandas as pd
import io
import os

# Import lớp tiền xử lý
from preprocess import CSVPreprocessor

# ====================== CẤU HÌNH ======================
PORT = 5000
TRAIN_DIR = '../train'
app = Flask(__name__)

BEAN_CLASSES = [
    "Seker (Đậu Đường)",
    "Barbunya (Đậu Thổ Nhĩ Kỳ)",
    "Bombay (Đậu Bombay)",
    "Cali (Đậu Calypso)",
    "Horoz (Đậu Horoz)",
    "Sira (Đậu Sira)",
    "Dermason (Đậu Dermason)"
]

MODEL_PATHS = {
    'KNeighbors': 'knn_model.pkl',
    'NaiveBayes': 'naive_bayes_model.pkl',
    'LogisticRegression': 'logistic_regression_model.pkl',
    'RandomForest': 'random_forest_model.pkl',
    'SupportVectorMachine': 'svm_model.pkl',
}

FEATURE_NAMES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 
    'ShapeFactor4'
]

# ====================== DỰ ĐOÁN ======================
def predict_bean_class(data, model_name):
    model_filename = MODEL_PATHS.get(model_name)
    if not model_filename:
        return f"LỖI: KHÔNG TÌM THẤY TỆP MÔ HÌNH '{model_name}'"

    full_model_path = os.path.join(TRAIN_DIR, model_filename)

    # Chỉ lấy 16 thuộc tính cần thiết
    numeric_data = [float(data[f]) for f in FEATURE_NAMES if f in data]

    if len(numeric_data) != 16:
        return "LỖI: THIẾU DỮ LIỆU ĐẦU VÀO (Cần 16 thuộc tính)"

    try:
        with open(full_model_path, 'rb') as f:
            model = pickle.load(f)
        prediction_result = model.predict([numeric_data])[0]

        # Nếu mô hình trả int index, dùng BEAN_CLASSES
        if isinstance(prediction_result, int) and 0 <= prediction_result < len(BEAN_CLASSES):
            return BEAN_CLASSES[prediction_result]
        return str(prediction_result)
    except FileNotFoundError:
        return f"LỖI: TỆP MÔ HÌNH '{full_model_path}' KHÔNG TỒN TẠI"
    except Exception as e:
        return f"LỖI DỰ ĐOÁN: {str(e)}"

# ====================== GIAO DIỆN FLASK ======================
@app.route('/', methods=['GET', 'POST'])
def prediction_interface():
    prediction_result = None
    data = None
    message = None
    model_name = 'RandomForest'
    df = None

    if request.method == 'POST':
        model_name = request.form.get('model_name', 'RandomForest')
        csv_has_header = request.form.get('csv_has_header') == 'on'
        csv_file = request.files.get('csv_file')
        input_features = {}
        data_source = "Thủ công"

        # ---------------------- CSV UPLOAD ----------------------
        if csv_file and csv_file.filename and csv_file.filename.endswith('.csv'):
            try:
                # Đọc file CSV từ người dùng
                csv_data = csv_file.read().decode('utf-8')
                if csv_has_header:
                    df = pd.read_csv(io.StringIO(csv_data))
                else:
                    df = pd.read_csv(io.StringIO(csv_data), header=None, names=FEATURE_NAMES)

                if df.empty:
                    message = "LỖI: Tệp CSV trống."
                elif not all(f in df.columns for f in FEATURE_NAMES):
                    missing = [f for f in FEATURE_NAMES if f not in df.columns]
                    message = f"LỖI: Thiếu cột bắt buộc: {', '.join(missing)}"
                else:
                    # TIỀN XỬ LÝ CSV
                    preprocessor = CSVPreprocessor(n_components=5)
                    df_processed = preprocessor.preprocess_csv(io.StringIO(csv_data))

                    # Dùng hàng đầu tiên của file đã tiền xử lý để dự đoán
                    first_row_data = df_processed.iloc[0].to_dict()
                    input_features = {k: str(v) for k, v in first_row_data.items() if k in FEATURE_NAMES}
                    data_source = "CSV (đã tiền xử lý)"
            except Exception as e:
                message = f"LỖI XỬ LÝ CSV: {str(e)}"

        # ---------------------- INPUT FORM ----------------------
        if not input_features and 'LỖI' not in str(message):
            exclude_keys = ['model_name', 'csv_file', 'csv_has_header']
            input_features = {k: v for k, v in request.form.items() if k not in exclude_keys}
            data_source = "Thủ công"

        # ---------------------- DỰ ĐOÁN ----------------------
        if input_features:
            try:
                prediction_result = predict_bean_class(input_features, model_name)
                if "LỖI" in prediction_result:
                    message = f"Lỗi dự đoán ({data_source}): {prediction_result}"
                elif data_source.startswith("CSV") and df is not None:
                    df['Predicted_Class'] = ''
                    df.loc[0, 'Predicted_Class'] = prediction_result
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    response = Response(output.getvalue(), mimetype="text/csv")
                    response.headers["Content-Disposition"] = "attachment; filename=bean_prediction_result.csv"
                    return response
                else:
                    message = f"Dự đoán thành công ({data_source}) bằng mô hình {model_name}: {prediction_result}"
            except Exception as e:
                message = f"Lỗi xử lý dữ liệu ({data_source}): {str(e)}"
                prediction_result = "LỖI DỰ ĐOÁN"

    if data is None:
        data = {'model_name': model_name}
    data['csv_has_header'] = request.form.get('csv_has_header', 'on')

    return render_template('index.html', message=message, data=data, prediction_result=prediction_result)

# ====================== MAIN ======================
if __name__ == '__main__':
    app.run(debug=True, port=PORT)
