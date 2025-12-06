"""
Elite Set-Piece Analytics - Data Preprocessors
معالجة البيانات لتحليلات الكرات الثابتة

This module handles data preprocessing:
- Cleaning missing values
- Normalizing coordinates
- Handling outliers
- Train/test splitting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and invalid entries
    تنظيف البيانات من خلال معالجة القيم المفقودة والإدخالات غير الصالحة
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Fill missing numerical values
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing categorical values
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('unknown')
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_len:
        print(f"⚠️ Removed {initial_len - len(df_clean)} duplicate rows")
    
    return df_clean


def normalize_coordinates(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    target_x_range: Tuple[float, float] = (0, 105),
    target_y_range: Tuple[float, float] = (0, 68)
) -> pd.DataFrame:
    """
    Normalize pitch coordinates to standard dimensions
    تطبيع إحداثيات الملعب إلى الأبعاد القياسية
    
    Args:
        df: Input DataFrame
        x_col: Name of x coordinate column
        y_col: Name of y coordinate column
        target_x_range: Target x range (default: 0-105 meters)
        target_y_range: Target y range (default: 0-68 meters)
    
    Returns:
        DataFrame with normalized coordinates
    """
    df_norm = df.copy()
    
    # Get current range
    x_min, x_max = df_norm[x_col].min(), df_norm[x_col].max()
    y_min, y_max = df_norm[y_col].min(), df_norm[y_col].max()
    
    # Normalize to target range
    df_norm[f'{x_col}_meters'] = (
        (df_norm[x_col] - x_min) / (x_max - x_min) * 
        (target_x_range[1] - target_x_range[0]) + target_x_range[0]
    )
    
    df_norm[f'{y_col}_meters'] = (
        (df_norm[y_col] - y_min) / (y_max - y_min) * 
        (target_y_range[1] - target_y_range[0]) + target_y_range[0]
    )
    
    return df_norm


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Handle outliers in numerical columns
    معالجة القيم المتطرفة في الأعمدة الرقمية
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (default: all numerical)
        method: 'iqr' for interquartile range, 'zscore' for z-score
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        DataFrame with outliers handled
    """
    df_out = df.copy()
    
    if columns is None:
        columns = df_out.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers instead of removing
            df_out[col] = df_out[col].clip(lower_bound, upper_bound)
            
        elif method == 'zscore':
            mean = df_out[col].mean()
            std = df_out[col].std()
            
            if std > 0:
                z_scores = (df_out[col] - mean) / std
                df_out.loc[abs(z_scores) > threshold, col] = mean
    
    return df_out


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    scaler_type: str = 'standard'
) -> Tuple[pd.DataFrame, pd.Series, object]:
    """
    Prepare features for model training
    تحضير الميزات لتدريب النموذج
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of target column
        scaler_type: 'standard' or 'minmax'
    
    Returns:
        Tuple of (scaled features, target, fitted scaler)
    """
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Create scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale features
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_columns,
        index=X.index
    )
    
    return X_scaled, y, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets
    تقسيم البيانات إلى مجموعات التدريب والتحقق والاختبار
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify: Whether to stratify by target
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    stratify_col = y if stratify else None
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)  # Adjust ratio
    stratify_col_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_col_temp
    )
    
    print(f"✅ Data split:")
    print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    encoding_type: str = 'onehot'
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables
    ترميز المتغيرات الفئوية
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns
        encoding_type: 'onehot' or 'label'
    
    Returns:
        Tuple of (encoded DataFrame, encoders dict)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
            
        if encoding_type == 'onehot':
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoders[col] = dummies.columns.tolist()
            
        elif encoding_type == 'label':
            encoder = LabelEncoder()
            df_encoded[f'{col}_encoded'] = encoder.fit_transform(
                df_encoded[col].astype(str)
            )
            encoders[col] = encoder
    
    return df_encoded, encoders


def create_training_pipeline(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> dict:
    """
    Complete training data pipeline
    خط أنابيب بيانات التدريب الكامل
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        target_column: Target column name
        categorical_columns: List of categorical columns to encode
        test_size: Test set proportion
        val_size: Validation set proportion
    
    Returns:
        Dictionary with all prepared data and transformers
    """
    print("🔧 Running training pipeline...")
    
    # 1. Clean data
    print("  1. Cleaning data...")
    df_clean = clean_data(df)
    
    # 2. Handle outliers
    print("  2. Handling outliers...")
    numerical_features = [col for col in feature_columns 
                         if col in df_clean.select_dtypes(include=[np.number]).columns]
    df_clean = handle_outliers(df_clean, columns=numerical_features)
    
    # 3. Encode categorical
    if categorical_columns:
        print("  3. Encoding categorical variables...")
        df_clean, encoders = encode_categorical(df_clean, categorical_columns)
        # Update feature columns with encoded ones
        for col in categorical_columns:
            if col in feature_columns:
                feature_columns.remove(col)
                if col in encoders:
                    feature_columns.extend(encoders[col])
    else:
        encoders = {}
    
    # 4. Prepare features
    print("  4. Preparing features...")
    # Filter to only existing columns
    available_features = [col for col in feature_columns if col in df_clean.columns]
    X, y, scaler = prepare_features(df_clean, available_features, target_column)
    
    # 5. Split data
    print("  5. Splitting data...")
    # Handle small datasets
    if len(X) < 10:
        X_train, X_val, X_test = X, X, X
        y_train, y_val, y_test = y, y, y
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=test_size, val_size=val_size
        )
    
    print("✅ Pipeline complete!")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'feature_columns': available_features
    }


if __name__ == "__main__":
    # Test the preprocessors
    from loaders import load_wyscout_data
    from extractors import extract_set_pieces, get_first_receiver_data
    
    print("=" * 50)
    print("Testing Data Preprocessors")
    print("=" * 50)
    
    # Load and extract data
    df = load_wyscout_data()
    set_pieces = extract_set_pieces(df)
    X, y = get_first_receiver_data(set_pieces)
    
    # Test split
    print("\nTesting data split...")
    if len(X) >= 10:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
