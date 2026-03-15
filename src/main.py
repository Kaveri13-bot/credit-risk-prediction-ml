from data_ingestion import split_and_save_data, load_train_data, load_test_data
from data_preprocessing import split_features_target,split_train_validation, build_preprocessor, prepare_test_features 
from model_training import train_model,build_model
from feature_engineering import create_features
from model_evaluation import evaluate_model
from model_prediction import generate_test_predictions
from config import TRAIN_DATA_PATH, TEST_DATA_PATH
import pandas as pd

def run_pipeline():

    # Step 0: Create train/test split
    print("Splitting dataset...")
    split_and_save_data()

    # Load data
    print("Loading data...")
    df_train = load_train_data()
    df_test = load_test_data()
       
     # Feature engineering
    print("Performing feature engineering...")

    def engineer_features(df):
        df = df.copy()

        # Clean column names
        df.columns = df.columns.str.strip()

        # Total Income
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

        # Loan to Income Ratio
        df["LoanIncomeRatio"] = df["LoanAmount"] / df["TotalIncome"]

        # EMI approximation
        df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]

        # Dependents cleaning
        df["Dependents"] = df["Dependents"].replace("3+", 3)
        df["Dependents"] = df["Dependents"].fillna(0).astype(int)

        # Income per dependent
        df["IncomePerDependent"] = df["TotalIncome"] / (df["Dependents"] + 1)

        # Encode target (only if exists)
        if "Loan_Status" in df.columns:
            df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

        return df

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    # Split features and target
    print("Preparing features and target...")
    X, y = split_features_target(df_train)
    # Build preprocessor
    print("Building preprocessor...")
    preprocessor = build_preprocessor(X)

    # Split into training and validation
    print("Splitting train and validation...")
    X_train, X_val, y_train, y_val = split_train_validation(X, y)

    # Build and train model
    print("Training model...")
    model = build_model(preprocessor)
    trained_model = train_model(model, X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    evaluate_model(trained_model, X_val, y_val)

    # Test predictions
    print("Generating predictions...")
    X_test = prepare_test_features(df_test)
    generate_test_predictions(trained_model, X_test)


if __name__ == "__main__":
    run_pipeline()