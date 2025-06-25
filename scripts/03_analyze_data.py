import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROCESSED_DATA_PATH = "final_processed_data.geojson"
# --- End Configuration ---

def main():
    print("--- Starting Step 5: Quantitative Data Analysis ---")
    
    # Load the final dataset
    try:
        gdf = gpd.read_file(PROCESSED_DATA_PATH)
        print("Final dataset loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load '{PROCESSED_DATA_PATH}'. Please run master_script.py first. {e}")
        return

    # --- Analysis 1: Correlation Matrix Heatmap ---
    print("\n--- Generating Correlation Matrix ---")
    
    # Select only the numeric columns we're interested in for correlation
    numeric_columns = ['TotalPopulation', 'MedianHouseholdIncome', 'MedianAge', 'ShopCount', 'AvgRating', 'TotalReviews']
    correlation_matrix = gdf[numeric_columns].corr()

    print("Correlation Matrix:")
    print(correlation_matrix)

    # Visualize the matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Key Variables", fontsize=16)
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("\nHeatmap saved to 'correlation_heatmap.png'")
    plt.show()


    # --- Analysis 2: Scatter Plots for Key Relationships ---
    print("\n--- Generating Scatter Plots ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Visualizing Key Relationships', fontsize=16)

    # Scatter plot: Income vs. Shop Count
    sns.scatterplot(data=gdf, x='MedianHouseholdIncome', y='ShopCount', ax=ax1)
    ax1.set_title("Income vs. Number of Coffee Shops")
    ax1.set_xlabel("Median Household Income ($)")
    ax1.set_ylabel("Number of Shops in Tract")
    ax1.grid(True)

    # Scatter plot: Population vs. Shop Count
    sns.scatterplot(data=gdf, x='TotalPopulation', y='ShopCount', ax=ax2)
    ax2.set_title("Population vs. Number of Coffee Shops")
    ax2.set_xlabel("Total Population")
    ax2.set_ylabel("Number of Shops in Tract")
    ax2.grid(True)

    plt.savefig("scatter_plots.png", dpi=150)
    print("Scatter plots saved to 'scatter_plots.png'")
    plt.show()


    # --- Analysis 3: Programmatically Find "Opportunity Hotspots" ---
    print("\n--- Finding Potential 'Opportunity Hotspots' ---")

    # Define our criteria for a hotspot:
    # - Population is in the top 25% (75th percentile)
    # - Median Income is in the top 25% (75th percentile)
    # - The number of shops is 0
    pop_threshold = gdf['TotalPopulation'].quantile(0.75)
    income_threshold = gdf['MedianHouseholdIncome'].quantile(0.75)

    hotspots_df = gdf[
        (gdf['TotalPopulation'] >= pop_threshold) &
        (gdf['MedianHouseholdIncome'] >= income_threshold) &
        (gdf['ShopCount'] == 0)
    ]

    print(f"Criteria for Hotspot: Population >= {int(pop_threshold)}, Income >= ${int(income_threshold)}, and Shop Count == 0")
    print(f"\nFound {len(hotspots_df)} potential hotspot census tracts matching the criteria.")

    if not hotspots_df.empty:
        print("Top 5 Hotspots Found:")
        # Sort by population to see the most dense opportunities first
        print(hotspots_df[['GEOID', 'TotalPopulation', 'MedianHouseholdIncome', 'ShopCount']].sort_values(by='TotalPopulation', ascending=False).head())

    # Save the full list of hotspots to a file for further investigation
    hotspots_df.to_csv("opportunity_hotspots.csv", index=False)
    print("\nFull list of potential hotspots saved to 'opportunity_hotspots.csv'")
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
