def create_features(df):
    df = df.copy()

    # -------------------------
    # Clean column names
    # -------------------------
    df.columns = df.columns.str.strip()

    # -------------------------
    # Create Total Income
    # -------------------------
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # -------------------------
    # Loan to Income Ratio
    # -------------------------
    df["LoanIncomeRatio"] = df["LoanAmount"] / df["TotalIncome"]

    # -------------------------
    # EMI Approximation (LoanAmount / Term)
    # -------------------------
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]

    # -------------------------
    # Income per dependent
    # -------------------------
    df["Dependents"] = df["Dependents"].replace("3+", 3)
    df["Dependents"] = df["Dependents"].fillna(0).astype(int)

    df["IncomePerDependent"] = df["TotalIncome"] / (df["Dependents"] + 1)

    # -------------------------
    # Binary encode target
    # -------------------------
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    return df