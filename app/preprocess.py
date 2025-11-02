# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class CSVPreprocessor:
    def __init__(self, n_components: int = 5):
        """
        Khởi tạo bộ tiền xử lý CSV.
        :param n_components: số thành phần chính cho PCA.
        """
        self.n_components = n_components
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=n_components)
        self.columns_before_pca = None

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển các cột có kiểu dữ liệu là chuỗi số (string) sang kiểu số thực (float/int).
        """
        for col in df.columns:
            # Nếu cột là object nhưng có thể chuyển thành số
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except Exception:
                    pass  # bỏ qua nếu không thể chuyển
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ các hàng null nếu ít, nếu nhiều thì thay bằng median.
        """
        for col in df.columns:
            null_ratio = df[col].isnull().mean()
            if null_ratio == 0:
                continue
            elif null_ratio < 0.1:
                # Nếu ít hơn 10% giá trị bị thiếu → xóa hàng đó
                df = df.dropna(subset=[col])
            else:
                # Nếu nhiều hơn → thay thế bằng median
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
        return df

    def _scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu số bằng MinMaxScaler.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Giảm chiều dữ liệu bằng PCA.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.columns_before_pca = numeric_cols
        pca_result = self.pca.fit_transform(df[numeric_cols])
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f"PCA_{i+1}" for i in range(self.n_components)],
            index=df.index
        )
        # Kết hợp phần không phải số với kết quả PCA
        non_numeric = df.drop(columns=numeric_cols)
        return pd.concat([pca_df, non_numeric], axis=1)

    def preprocess_csv(self, file_path: str) -> pd.DataFrame:
        """
        Tiền xử lý file CSV: làm sạch, chuẩn hóa và giảm chiều.
        """
        df = pd.read_csv(file_path)

        df = self._convert_numeric_columns(df)
        df = self._handle_missing_values(df)
        df = self._scale_data(df)
        df = self._apply_pca(df)

        return df
