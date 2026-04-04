"""
Automate Preprocessing - Credit Scoring
Muhammad Rifqy Saputra (Dicoding: rifqysaputra)

File ini mengkonversi langkah preprocessing dari notebook
menjadi fungsi otomatis untuk Kriteria 1 (Skilled).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(input_path):
    """
    Load raw data dari folder namadataset_raw
    
    Parameters:
    -----------
    input_path : str or Path
        Path ke file CSV raw data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame yang sudah dimuat
    """
    logger.info(f"Loading data from: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def perform_eda(df):
    """
    Melakukan Exploratory Data Analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame untuk dianalisis
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 60)
    
    # Info dataset
    logger.info("\n1. Dataset Info:")
    logger.info(f"   - Rows: {df.shape[0]}")
    logger.info(f"   - Columns: {df.shape[1]}")
    logger.info(f"   - Column names: {list(df.columns)}")
    
    # Data types
    logger.info("\n2. Data Types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"   - {col}: {dtype}")
    
    # Missing values
    logger.info("\n3. Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            logger.info(f"   - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    if missing.sum() == 0:
        logger.info("   - No missing values found")
    
    # Target distribution
    logger.info("\n4. Target Distribution:")
    target_dist = df['target'].value_counts(normalize=True)
    for val, pct in target_dist.items():
        logger.info(f"   - Target {val}: {pct*100:.1f}%")
    
    logger.info("=" * 60)

def preprocess_data(df):
    """
    Melakukan preprocessing data
    
    Steps:
    ------
    1. Handle missing values (numerical: median, categorical: mode)
    2. Encode categorical variables dengan LabelEncoder
    3. Return processed DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame siap untuk modelling
    """
    logger.info("\nStarting preprocessing...")
    
    # Copy dataframe
    processed = df.copy()
    
    # 1. Handle missing values pada kolom numerik dengan median
    numerical_cols = ['age', 'income', 'loan_amount', 'employment_years', 'late_payments']
    logger.info(f"\n1. Handling missing values in numerical columns: {numerical_cols}")
    
    for col in numerical_cols:
        if col in processed.columns:
            # Convert to numeric
            processed[col] = pd.to_numeric(processed[col], errors='coerce')
            # Fill missing with median
            median_val = processed[col].median()
            processed[col] = processed[col].fillna(median_val)
            logger.info(f"   - {col}: filled {processed[col].isnull().sum()} missing with median {median_val}")
    
    # 2. Handle missing values pada kolom kategorikal dengan modus
    categorical_cols = ['home_ownership', 'marital_status']
    logger.info(f"\n2. Handling missing values in categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        if col in processed.columns:
            # Clean and fill
            processed[col] = processed[col].astype('string').fillna('unknown').str.strip().str.lower()
            mode_val = processed[col].mode().iloc[0] if not processed[col].mode().empty else 'unknown'
            processed[col] = processed[col].replace({'<NA>': mode_val}).fillna(mode_val)
            logger.info(f"   - {col}: filled missing with mode '{mode_val}'")
    
    # 3. Encode categorical dengan LabelEncoder
    logger.info(f"\n3. Encoding categorical variables...")
    
    label_encoders = {}
    for col in categorical_cols:
        if col in processed.columns:
            le = LabelEncoder()
            processed[col] = le.fit_transform(processed[col].astype(str))
            label_encoders[col] = le
            logger.info(f"   - {col}: encoded {len(le.classes_)} categories")
    
    logger.info(f"\n✅ Preprocessing complete!")
    logger.info(f"   Final shape: {processed.shape}")
    
    return processed

def save_processed_data(processed_df, output_path):
    """
    Menyimpan hasil preprocessing ke folder namadataset_preprocessing
    
    Parameters:
    -----------
    processed_df : pd.DataFrame
        DataFrame yang sudah diproses
    output_path : str or Path
        Path untuk menyimpan file
    """
    logger.info(f"\nSaving processed data to: {output_path}")
    
    # Create directory if not exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    processed_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved successfully!")
    logger.info(f"   - Rows: {len(processed_df)}")
    logger.info(f"   - Columns: {len(processed_df.columns)}")
    logger.info(f"   - File: {output_path}")

def main(input_file=None, output_file=None):
    """
    Main function untuk menjalankan preprocessing otomatis
    
    Parameters:
    -----------
    input_file : str, optional
        Path ke file raw data. Default: '../namadataset_raw/credit_scoring_raw.csv'
    output_file : str, optional
        Path untuk menyimpan hasil. Default: 'namadataset_preprocessing/credit_scoring_preprocessing.csv'
    """
    # Default paths
    if input_file is None:
        input_file = '../namadataset_raw/credit_scoring_raw.csv'
    
    if output_file is None:
        output_file = 'namadataset_preprocessing/credit_scoring_preprocessing.csv'
    
    logger.info("=" * 70)
    logger.info("AUTOMATE PREPROCESSING - CREDIT SCORING")
    logger.info("Muhammad Rifqy Saputra")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load data
        df = load_data(input_file)
        
        # Step 2: EDA
        perform_eda(df)
        
        # Step 3: Preprocessing
        processed = preprocess_data(df)
        
        # Step 4: Save
        save_processed_data(processed, output_file)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        return processed
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Automate preprocessing for Credit Scoring dataset'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../namadataset_raw/credit_scoring_raw.csv',
        help='Input file path (raw data)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='namadataset_preprocessing/credit_scoring_preprocessing.csv',
        help='Output file path (processed data)'
    )
    
    args = parser.parse_args()
    
    # Run main function
    main(args.input, args.output)
