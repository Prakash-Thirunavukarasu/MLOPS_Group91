import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path="data/raw/iris.csv", 
                    output_folder="data/processed"):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    # --- Step 1: Data Cleaning (Illustrative) ---
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df.fillna(df.median(numeric_only=True), inplace=True)

    print("Splitting data into training and testing sets...")

    # Define features (X) and the target (y)
    X = df.drop("species", axis=1)
    y = df["species"]

    # Split the data with an 80/20 ratio, ensuring balanced classes in splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Combine features and target back into single dataframes for easy saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # --- Step 2: Save Processed Data ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving processed data to: {output_folder}")

    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

    print("Preprocessing complete. train.csv and test.csv created.")

if __name__ == "__main__":
    preprocess_data()