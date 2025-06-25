import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import os

# --- MASTER CONFIGURATION ---
ORIGINAL_SHAPEFILE_PATH = "/home/gch93/Downloads/tl_2024/tl_2024_06_tract.shp"
SHOPS_CSV_PATH = "riverside_coffee_shops.csv"
CENSUS_VARIABLES_TO_GET = "B01003_001E,B19013_001E,B01002_001E"
CENSUS_API_KEY = ""
STATE_FIPS = "06"
COUNTY_FIPS = "065"
# --- END CONFIGURATION ---

def get_census_data():
    """Calls the Census API and returns a cleaned DataFrame."""
    print("--- Calling Census API... ---")
    api_url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get=NAME,{CENSUS_VARIABLES_TO_GET}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}+county:{COUNTY_FIPS}"
        f"&key={CENSUS_API_KEY}"
    )
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    
    rename_dict = {
        'B01003_001E': 'TotalPopulation',
        'B19013_001E': 'MedianHouseholdIncome',
        'B01002_001E': 'MedianAge',
    }
    df = df.rename(columns=rename_dict)
    
    for col in rename_dict.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['state'] = df['state'].astype(str).str.zfill(2)
    df['county'] = df['county'].astype(str).str.zfill(3)
    df['tract'] = df['tract'].astype(str).str.zfill(6)
    df['GEOID'] = df['state'] + df['county'] + df['tract']
    
    print("Census data processed successfully.")
    return df[['GEOID', 'TotalPopulation', 'MedianHouseholdIncome', 'MedianAge']]


def main():
    print("--- Starting Master Script with Weighted Rating ---")
    
    # STEP 1: Load and filter map data
    print(f"Loading original shapefile...")
    try:
        california_tracts = gpd.read_file(ORIGINAL_SHAPEFILE_PATH)
        tracts_gdf = california_tracts[california_tracts['COUNTYFP'] == COUNTY_FIPS].copy()
        tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str).str.strip()
        print("Tracts map loaded and filtered.")
    except Exception as e:
        print(f"ERROR loading shapefile: {e}")
        return

    # STEP 2: Get demographic data
    demographics_df = get_census_data()

    # STEP 3: Load and process coffee shop data
    print(f"Loading coffee shop data...")
    try:
        shops_df = pd.read_csv(SHOPS_CSV_PATH)
        shops_gdf = gpd.GeoDataFrame(
            shops_df, 
            geometry=gpd.points_from_xy(shops_df.longitude, shops_df.latitude),
            crs="EPSG:4326"
        )
        shops_gdf = shops_gdf.to_crs(tracts_gdf.crs)
        print("Coffee shop data loaded.")
    except Exception as e:
        print(f"ERROR loading coffee shop data: {e}")
        return

    # STEP 4: Perform joins and aggregations
    print("Performing spatial joins and data merges...")
    shops_with_tract_data = gpd.sjoin(shops_gdf, tracts_gdf, how="inner", predicate="within")
    aggregated_shops = shops_with_tract_data.groupby('GEOID').agg(
        ShopCount=('name', 'count'),
        AvgRating=('rating', 'mean'),
        TotalReviews=('review_count', 'sum')
    ).reset_index()

    final_gdf = tracts_gdf.merge(aggregated_shops, on='GEOID', how='left')
    final_gdf = final_gdf.merge(demographics_df, on='GEOID', how='left')

    fill_values = {'ShopCount': 0, 'AvgRating': 0, 'TotalReviews': 0}
    final_gdf.fillna(value=fill_values, inplace=True)
    
    print("Calculating weighted average rating...")
    C = shops_df['rating'].mean()
    m = shops_df['review_count'].quantile(0.75)
    v = final_gdf['TotalReviews']
    R = final_gdf['AvgRating']
    final_gdf['WeightedAvgRating'] = (v / (v + m)) * R + (m / (v + m)) * C
    
    # FIX for FutureWarning: Use this method instead of inplace=True for fillna
    final_gdf['WeightedAvgRating'] = final_gdf['WeightedAvgRating'].fillna(0)
    
    print(f"  Global average rating (C): {C:.2f}")
    print(f"  Review threshold (m): {m:.0f} reviews")
    print("All data has been merged and processed.")

    # STEP 5: Visualize the final data
    print("\nGenerating final visualization with Weighted Rating...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exploratory Data Analysis of Riverside County (with Weighted Ratings)', fontsize=20)
    
    # --- Map 1: Total Population ---
    ax1 = axes[0, 0]
    final_gdf.plot(column='TotalPopulation', ax=ax1, legend=True, cmap='viridis', legend_kwds={'label': "Population Count", 'orientation': "horizontal"})
    ax1.set_title("Total Population per Census Tract")
    ax1.set_axis_off()

    # --- Map 2: Median Household Income ---
    # FIX for NameError: Added the missing ax2 definition
    ax2 = axes[0, 1]
    final_gdf[final_gdf['MedianHouseholdIncome'] > 0].plot(column='MedianHouseholdIncome', ax=ax2, legend=True, cmap='plasma', legend_kwds={'label': "Median Household Income ($)", 'orientation': "horizontal"})
    ax2.set_title("Median Household Income per Census Tract")
    ax2.set_axis_off()

    # --- Map 3: Coffee Shop Count (IMPROVED) ---
    ax3 = axes[1, 0]
    final_gdf.plot(column='ShopCount',
             ax=ax3,
             legend=True,
             cmap='cividis',
             scheme='NaturalBreaks',
             k=5,
             legend_kwds={'title': "Number of Coffee Shops", 'loc': 'upper left'})
    ax3.set_title("Coffee Shop Count per Census Tract (Improved Scale)")
    ax3.set_axis_off()

    # --- Map 4: Weighted Average Rating (NEW) ---
    ax4 = axes[1, 1]
    final_gdf[final_gdf['AvgRating'] == 0].plot(ax=ax4, color='lightgray', edgecolor='black')
    final_gdf[final_gdf['WeightedAvgRating'] > 0].plot(
             column='WeightedAvgRating',
             ax=ax4,
             legend=True,
             cmap='magma',
             scheme='NaturalBreaks',
             k=5,
             legend_kwds={'title': "Weighted Avg Rating", 'loc': 'upper left'})
    ax4.set_title("Weighted Average Coffee Shop Rating")
    ax4.set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("final_maps_weighted_rating.png", dpi=150)
    print("SUCCESS! Final maps saved to 'final_maps_weighted_rating.png'")
    # --- Save the final processed data to a file ---
    print("Saving final unified data to 'final_processed_data.geojson'...")
    final_gdf.to_file("final_processed_data.geojson", driver='GeoJSON')
    print("SUCCESS! Final dataset saved to 'final_processed_data.geojson'.")
    plt.show()

if __name__ == "__main__":
    main()
