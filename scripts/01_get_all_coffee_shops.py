import requests
import csv
import time
import geopandas as gpd
import numpy as np

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual Google Places API Key
API_KEY = 'REMOVED_FOR_SECURITY'
# The type of place to search for
TYPE = 'cafe'
# The radius for each individual search in our grid (in meters).
# Smaller is more detailed but uses more API calls. 2000m (2km) is a good start.
RADIUS = 2000
# The file containing the boundaries of our search area
TRACTS_SHP_PATH = "riverside_county_tracts.shp"
OUTPUT_CSV_FILENAME = "riverside_coffee_shops_comprehensive.csv"
# --- END CONFIGURATION ---

def fetch_places(api_key, location_str, radius, place_type):
    """Fetches up to 60 places for a single location point."""
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location_str}&radius={radius}&type={place_type}&key={api_key}"
    )
    
    results = []
    page_count = 0
    while url and page_count < 3: # Google allows up to 3 pages (20 results each)
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching data: {e}")
            break

        for place in data.get('results', []):
            results.append({
                # place_id is the best unique identifier for de-duplication
                'place_id': place.get('place_id'), 
                'name': place.get('name'),
                'address': place.get('vicinity'),
                'latitude': place['geometry']['location']['lat'],
                'longitude': place['geometry']['location']['lng'],
                'rating': place.get('rating'),
                'review_count': place.get('user_ratings_total')
            })

        next_page_token = data.get('next_page_token')
        page_count += 1
        if next_page_token:
            time.sleep(2)  # Required delay before fetching the next page
            url = (
                f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                f"?pagetoken={next_page_token}&key={api_key}"
            )
        else:
            url = None
            
    return results

def save_to_csv(data, filename):
    """Saves a list of dictionaries to a CSV file."""
    if not data:
        print("No data to save.")
        return
        
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def create_search_grid(gdf, radius_m):
    """Creates a grid of lat/lon points covering the GeoDataFrame's total bounds."""
    # Convert radius from meters to degrees (approximate)
    # 1 degree of latitude is roughly 111,111 meters
    radius_deg = radius_m / 111111.0
    
    # Get the total bounds of our search area
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Create grid points with spacing based on the radius
    x_coords = np.arange(minx, maxx, radius_deg * 1.5) # Overlap circles slightly
    y_coords = np.arange(miny, maxy, radius_deg * 1.5)
    
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append(f"{y},{x}") # Format as "latitude,longitude"
            
    return grid_points

if __name__ == "__main__":
    if API_KEY == 'YOUR_API_KEY_HERE' or not API_KEY:
        print("ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual Google Places API Key.")
    else:
        print("--- Starting Comprehensive Data Collection ---")
        
        # 1. Load our map to define the search area
        print(f"Loading search area boundary from '{TRACTS_SHP_PATH}'...")
        try:
            tracts_gdf = gpd.read_file(TRACTS_SHP_PATH)
            # Ensure it's in a standard lat/lon CRS for bounds calculation
            tracts_gdf = tracts_gdf.to_crs("EPSG:4326")
        except Exception as e:
            print(f"ERROR: Could not load shapefile '{TRACTS_SHP_PATH}'. {e}")
            exit()

        # 2. Create the grid of search points
        search_grid = create_search_grid(tracts_gdf, RADIUS)
        print(f"Created a search grid with {len(search_grid)} points to cover Riverside County.")
        
        # 3. Iterate through the grid and fetch data
        all_places = {} # Use a dictionary to handle duplicates automatically
        for i, location_point in enumerate(search_grid):
            print(f"--> Searching grid point {i+1}/{len(search_grid)} at location {location_point}...")
            places_found = fetch_places(API_KEY, location_point, RADIUS, TYPE)
            print(f"  Found {len(places_found)} results for this point.")
            for place in places_found:
                # Use place_id as the unique key to avoid duplicates
                all_places[place['place_id']] = place
        
        # 4. Convert the dictionary of unique places back to a list
        final_results = list(all_places.values())
        
        print("\n--- Collection Complete ---")
        print(f"Retrieved a total of {len(final_results)} unique coffee shops.")
        
        # 5. Save the comprehensive data to a new CSV
        save_to_csv(final_results, OUTPUT_CSV_FILENAME)
        print(f"Saved comprehensive data to '{OUTPUT_CSV_FILENAME}'")
