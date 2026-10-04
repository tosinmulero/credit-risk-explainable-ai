"""Canonical semantic schema for the UCI credit-default dataset."""

COLUMN_NAME_MAP = {
    "ID": "CLIENT_ID",
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": "DEFAULT_NEXT_MONTH",
}

FEATURE_NAME_MAP = {key: value for key, value in COLUMN_NAME_MAP.items() if key.startswith("X")}

TARGET_NAME = "DEFAULT_NEXT_MONTH"
IDENTIFIER_NAME = "CLIENT_ID"
