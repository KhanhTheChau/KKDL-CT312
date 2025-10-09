
Phân tích tập dữ liệu Dry Bean 2025

## Thành viên nhóm
1. Châu Thế Khanh	    B2207528	
2. Nguyễn Lê Tấn Thành	B2207565	
3. Nguyễn Thị Trà My	B2207544

## Mục tiêu
Tìm hiểu dữ liệu
- Tiền xử lý dữ liệu
- Phân tích trực quan
- Xây dựng mô hình máy học (hồi quy, phân lớp, gom nhóm, …)
- Đánh giá mô hình
- Xây dựng website

## Tổng quan dự án
- Dry Bean Dataset (Dữ liệu Đậu Khô)
- Mục tiêu	Phân loại 7 loại đậu khô khác nhau.	Đây là một bài toán Phân loại Đa lớp (Multi-class) kinh điển.

## Thách thức và Bước Tiền xử lý Cần thiết
1. Chuẩn hóa
- Các đặc trưng có thang đo rất khác nhau (ví dụ: Area là hàng chục nghìn, trong khi ShapeFactor là số thập phân nhỏ).
- HÀNH ĐỘNG: Bắt buộc phải áp dụng StandardScaler hoặc MinMaxScaler cho tất cả 17 cột đặc trưng để tránh việc các đặc trưng có giá trị lớn chi phối quá trình huấn luyện mô hình (đặc biệt là với các thuật toán dựa trên khoảng cách như SVM, KNN, hoặc Neural Networks).

2. Dữ liệu không sạch hoàn toàn
- Có missing values ở một số cột (Extent: 1, Solidity: 1, roundness: 4, Compactness: 2, ShapeFactor1: 2, ShapeFactor2: 3, ShapeFactor3: 1). Số lượng ít, có thể xử lý bằng imputation (mean/median) mà không ảnh hưởng lớn.
- Lỗi chính tả cột: 'AspectRation' nên là 'AspectRatio'.
- Cột thừa: 'ShapeFactor5' lặp lại thành 'ShapeFactor5.1' (có lẽ lỗi copy), và 'Name' (chỉ là mã như B1, B2) không hữu ích cho ML – nên loại bỏ để tránh nhiễu.

3. Kiểm tra Sự Cân bằng Lớp (Class Imbalance):
- HÀNH ĐỘNG: Nếu một số loại đậu có số lượng quá ít so với các loại khác, bạn có thể cần áp dụng các kỹ thuật xử lý mất cân bằng lớp (ví dụ: SMOTE hoặc sử dụng các ma trận trọng số lớp trong quá trình huấn luyện).

4. Xử lý Tương quan Đặc trưng (Multicollinearity):
- Do các đặc trưng đều được suy ra từ hình ảnh, một số đặc trưng có thể có mối tương quan rất cao (ví dụ: Area và EquivalentDiameter đo cùng một khái niệm).
- HÀNH ĐỘNG: Bạn có thể tính ma trận tương quan. Nếu cần thiết, hãy xem xét sử dụng Phân tích Thành phần Chính (PCA) để giảm số chiều và loại bỏ sự dư thừa thông tin, giúp mô hình ổn định và nhanh hơn.

## Mô hình phù hợp:
- Baseline: Logistic Regression hoặc KNN (dễ giải thích).
- Nâng cao: Ensemble như Random Forest/XGBoost (xử lý imbalance tốt), hoặc CNN nếu kết hợp với dữ liệu hình ảnh gốc.
- Đánh giá: Sử dụng F1-score (macro) do imbalance, thay vì accuracy.
