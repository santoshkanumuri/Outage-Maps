import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv
import logging


# --- Configuration ---
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/api/buildings-status"
DEFAULT_MAP_CENTER = [33.58452231722856, -101.87542936763231]
DEFAULT_ZOOM = 18

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="TTU WiFi Map", layout="wide")


# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_data(url: str):
    """Fetches building data from the backend API."""
    try:
        logger.info(f"Fetching building data from {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} building records.")
        # Ensure key columns exist even if API returns empty list
        df = pd.DataFrame(data)
        required_cols = ['BuildingName', 'Status', 'Latitude', 'Longitude']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None # Add missing columns as None
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        st.error(f"Failed to fetch data from API: {e}", icon="🚨")
        return None
    except Exception as e:
        logger.exception(f"Error processing fetched data: {e}")
        st.error(f"An unexpected error occurred: {e}", icon="💥")
        return None


def create_building_map(df: pd.DataFrame):
    """Creates a Folium map with building status markers."""
    if df is None or df.empty:
        st.warning("No building data available to display.")
        # Return a basic map centered on default location
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    # Drop rows with invalid coordinates before creating GeoDataFrame
    df_valid = df.dropna(subset=['Latitude', 'Longitude']).copy()
    if len(df_valid) < len(df):
        st.warning(f"Excluded {len(df) - len(df_valid)} buildings due to missing coordinates.", icon="⚠️")

    if df_valid.empty:
        st.warning("No buildings with valid coordinates found.")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    # Create GeoDataFrame
    try:
        gdf = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid.Longitude, df_valid.Latitude),
            crs="EPSG:4326" # Standard WGS84 Latitude/Longitude
        )
        logger.info(f"Created GeoDataFrame with {len(gdf)} buildings.")
    except Exception as e:
        logger.exception("Failed to create GeoDataFrame.")
        st.error(f"Error creating geographic data: {e}", icon="❌")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)


    # Calculate map center (mean of coordinates) or use default
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    map_center = [center_lat, center_lon]

    # Create Folium map instance
    # Use tiles='CartoDB positron' or 'Stamen Toner' for less cluttered maps sometimes
    m = folium.Map(location=map_center, zoom_start=DEFAULT_ZOOM, control_scale=True)

    google_maps_tile = folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="© Google Maps",
        name="Google Maps (No Labels)"
    )

    google_maps_tile.add_to(m)

    # Define colors for status
    color_map = {'active': 'green', 'inactive': 'red'}

    # Add markers to the map
    for idx, row in gdf.iterrows():
        building_name = row['BuildingName']
        status = row['Status']
        color = color_map.get(status.lower(), 'gray') # Default to gray if status unknown

        # Create popup content
        popup_html = f"""
        <b>Building:</b> {building_name}<br>
        <b>WiFi Status:</b> <span style="color:{color};">{status.capitalize()}</span><br>
        <b>Coords:</b> ({row.geometry.y:.4f}, {row.geometry.x:.4f})
        """
        popup = folium.Popup(popup_html, max_width=300)

        # Add a CircleMarker
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x], # Folium uses [Lat, Lon]
            radius=20, # Adjust size
            color=color, # Border color
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=f"{building_name} ({status.capitalize()})" # Hover text
        ).add_to(m)

    logger.info("Added building markers to Folium map.")
    return m


# --- Streamlit App Layout ---
st.title("🏫 TTU WiFi Status Map")
st.markdown("Displays WiFi status for university buildings on an interactive map.")

# --- Fetch and Display Data ---
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

only_inactive = st.checkbox("Show only inactive", value=False)


building_data_df = fetch_data(API_ENDPOINT)

if building_data_df is not None:
    st.success(f"Loaded status for {len(building_data_df)} buildings.", icon="✅")

    # --- Display Summary ---
    status_counts = building_data_df['Status'].value_counts()
    active_count = status_counts.get('active', 0)
    inactive_count = status_counts.get('inactive', 0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active WiFi Buildings", active_count)
    with col2:
        st.metric("Inactive WiFi Buildings", inactive_count)
    # --- Display Interactive Map ---
    st.subheader("Building Status Map")

    if only_inactive:
        building_data_df = building_data_df[building_data_df['Status'] == 'inactive']
        st.warning("Showing only inactive buildings.", icon="⚠️")

    folium_map = create_building_map(building_data_df)

    # Use st_folium to render the map
    # Set a reasonable height, width can be dynamic
    st_data = st_folium(folium_map, width=None, height=800, returned_objects=[]) # use_container_width=True might work too

    # --- Optional: Display Raw Data Table ---
    with st.expander("Show locations of all buildings", expanded=False):
        # in 2 columns show active and inactive buildings separately in table format
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Active Buildings")
            st.dataframe(building_data_df[building_data_df['Status'] == 'active']['BuildingName'])
        with col2:
            st.subheader("Inactive Buildings")
            st.dataframe(building_data_df[building_data_df['Status'] == 'inactive']['BuildingName'])




else:
    st.warning("Could not load building data to display the map.")
