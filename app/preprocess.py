# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.feature_names = None

    def fit(self, df: pd.DataFrame, feature_names: List[str]):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột khi fit: {missing}")

        df_selected = df[feature_names].copy()

        # Chuẩn hóa tất cả cột thành số (hỗ trợ dấu phẩy)
        for col in df_selected.columns:
            df_selected[col] = (
                df_selected[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .replace(['', 'nan', '<NA>'], np.nan)
            )
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

        df_selected = df_selected.dropna()
        if df_selected.empty:
            raise ValueError("Không có dữ liệu hợp lệ sau khi chuẩn hóa để fit scaler.")

        self.scaler.fit(df_selected)
        self.feature_names = feature_names
        self.fitted = True

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa giống hệt fit()"""
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        df = df[self.feature_names].copy()

        for col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .replace(['', 'nan', '<NA>'], np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        return df

    def transform_single(self, data: dict) -> list:
        if not self.fitted:
            raise RuntimeError("Scaler chưa được fit!")

        df = pd.DataFrame([data])
        df = self._preprocess_df(df)

        if df.empty:
            raise ValueError("Dữ liệu đầu vào không hợp lệ sau xử lý.")

        scaled = self.scaler.transform(df).flatten().tolist()
        return scaled

    def transform_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Scaler chưa được fit!")

        df = self._preprocess_df(df)

        if df.empty:
            raise ValueError("Không có dữ liệu hợp lệ trong CSV sau xử lý.")

        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=self.feature_names)


def prepare_csv_df(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df_out = df.copy()
    df_out.columns = [str(col).strip() for col in df_out.columns]

    missing = [col for col in feature_names if col not in df_out.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")

    df_out = df_out[feature_names].copy()

    # Chuẩn hóa giống fit
    for col in df_out.columns:
        df_out[col] = (
            df_out[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .replace(['', 'nan', '<NA>'], np.nan)
        )
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')

    df_out = df_out.dropna()
    return df_out


def validate_single_input(data: dict, feature_names: List[str]) -> Tuple[Union[list, None], Union[str, None]]:
    cleaned_data = {str(k).strip(): data.get(k) for k in feature_names}
    numeric_data = []
    error = None

    for key in feature_names:
        value = cleaned_data.get(key)
        if value is None or str(value).strip() == '':
            error = f"Dữ liệu nhập bị thiếu: '{key}'"
            return None, error

        try:
            processed_value = str(value).strip().replace(',', '.')
            num = float(processed_value)
            numeric_data.append(num)
        except ValueError:
            error = f"Dữ liệu nhập không hợp lệ: '{key}' phải là một số."
            return None, error

    return numeric_data, error