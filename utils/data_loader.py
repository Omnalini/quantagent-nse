import pandas as pd

def load_stock_data(path):
    df = pd.read_csv(path)

    # CASE 1: Date column exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    # CASE 2: Date is already index (Unnamed column)
    else:
        df.index = pd.to_datetime(df.index)

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df
