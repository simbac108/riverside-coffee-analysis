import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- Configuration ---
ML_DATA_PATH = "ml_ready_data.npz"
PROCESSED_DATA_PATH = "final_processed_data.geojson" # Needed for final analysis
# --- End Configuration ---

def main():
    print("--- Starting Step 7: Training a Predictive Model ---")

    # 1. Load the prepared data for Machine Learning
    try:
        data = np.load(ML_DATA_PATH, allow_pickle=True)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        print("ML-ready data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{ML_DATA_PATH}'. Please run prepare_ml_data.py first.")
        return

    # 2. Initialize and Train the Model
    print("\nTraining the Gradient Boosting Regressor model...")
    # We use a regressor because we are predicting a number (ShopCount)
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 3. Evaluate Model Performance on the Test Set
    print("\n--- Model Performance Evaluation ---")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("  (This means on average, our model's prediction is off by ~{:.2f} shops)".format(mae))
    
    print(f"\nR-squared (R²): {r2:.4f}")
    print("  (This means our model explains ~{:.1f}% of the variance in the shop count)".format(r2 * 100))
    print("------------------------------------")


    # 4. Use the Trained Model to Find Opportunity Hotspots
    print("\n--- Finding Opportunity Hotspots with the Trained Model ---")
    
    # Load the full, original processed dataset to get GEOIDs and other info
    gdf = gpd.read_file(PROCESSED_DATA_PATH)
    # Prepare the features from the full dataset, just like we did for training
    features = ['TotalPopulation', 'MedianHouseholdIncome', 'MedianAge']
    full_df = gdf[features + ['ShopCount', 'GEOID']].dropna()
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_full_scaled = scaler.fit_transform(full_df[features])
    
    # Predict the "expected" shop count for ALL tracts
    full_df['PredictedShopCount'] = model.predict(X_full_scaled)
    
    # Calculate the "Opportunity Score"
    # A high score means the model expected significantly more shops than actually exist.
    # We define it as Predicted - Actual.
    full_df['OpportunityScore'] = full_df['PredictedShopCount'] - full_df['ShopCount']
    
    # Find the tracts with the highest opportunity score
    top_hotspots = full_df.sort_values(by='OpportunityScore', ascending=False).head(5)

    print("\nTop 5 Census Tracts Identified as 'Opportunity Hotspots':")
    print(top_hotspots[['GEOID', 'TotalPopulation', 'MedianHouseholdIncome', 'ShopCount', 'PredictedShopCount', 'OpportunityScore']])
    
    # Save the full results with predictions to a new file for further review
    output_filename = "final_analysis_with_predictions.csv"
    full_df.to_csv(output_filename, index=False)
    print(f"\nFull analysis with predictions saved to '{output_filename}'")
    print("\n--- Project Complete ---")


if __name__ == "__main__":
    main()
