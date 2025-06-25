import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- Configuration ---
PROCESSED_DATA_PATH = "final_processed_data.geojson"
# --- End Configuration ---

def main():
    print("--- Starting Step 6: Preparing Data for Machine Learning ---")

    # 1. Load our final, clean dataset
    try:
        gdf = gpd.read_file(PROCESSED_DATA_PATH)
        print("Final dataset loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load '{PROCESSED_DATA_PATH}'. {e}")
        return

    # 2. Select features (X) and the target (y)
    # We'll use the core demographic data to predict the number of shops.
    features = ['TotalPopulation', 'MedianHouseholdIncome', 'MedianAge']
    target = 'ShopCount'
    
    print(f"\nFeatures (X): {features}")
    print(f"Target (y): {target}")

    # Create our X and y dataframes, dropping any rows with missing demographic data
    df = gdf[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # 3. Normalize the Features
    # ML models work best when all features are on a similar scale (e.g., 0 to 1).
    print("\nNormalizing features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # The result is a NumPy array, let's put it back into a DataFrame for clarity
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    print("Preview of scaled features:")
    print(X_scaled_df.head())

    # 4. Split the data into training and testing sets
    # 80% for training the model, 20% for testing its performance on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    print(f"\nData split into training and testing sets:")
    print(f"  Training set size: {len(X_train)} tracts")
    print(f"  Testing set size: {len(X_test)} tracts")

    # 5. Save the prepared data arrays
    # We can save these so we don't have to repeat this step every time.
    # np.savez allows saving multiple arrays into a single file.
    output_filename = "ml_ready_data.npz"
    np.savez(output_filename, 
             X_train=X_train, 
             X_test=X_test, 
             y_train=y_train, 
             y_test=y_test)
             
    print(f"\nSUCCESS! Machine learning-ready data saved to '{output_filename}'")
    print("We are now ready to train a model.")
    print("\n--- ML Preparation Complete ---")


if __name__ == "__main__":
    main()
