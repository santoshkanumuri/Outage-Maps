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
# --- !!! IMPORTANT: Ensure this matches the backend endpoint that serves BOTH statuses !!! ---
API_ENDPOINT = f"{API_BASE_URL}/api/building-status"
DEFAULT_MAP_CENTER = [33.58452231722856, -101.87542936763231] # TTU Approx Center
DEFAULT_ZOOM = 18
# --- NEW: Options for the dropdown ---
STATUS_TYPES = ["WiFi Status", "Audio Status"]

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="TTU Systems Status Map", layout="wide") # Updated title slightly

# --- Helper Functions ---
@st.cache_data(ttl=60)
def fetch_data(url: str):
    """Fetches building data (including all statuses) from the backend API."""
    try:
        logger.info(f"Fetching building data from {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} building records.")
        df = pd.DataFrame(data)
        # --- Use lowercase column names consistent with Pydantic model/JSON ---
        required_cols = ['BuildingName', 'Latitude', 'Longitude', 'WifiStatus', 'AudioStatus']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing in API response, defaulting to 'inactive'.")
                df[col] = 'inactive' # Default missing columns if necessary
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        # Attempt to show error details if available
        error_detail = ""
        try:
            error_detail = response.json().get("detail", "") if response else ""
        except Exception:
            pass # Ignore errors parsing the error response itself
        st.error(f"Failed to fetch data from API: {e}. {error_detail}", icon="🚨")
        return None
    except Exception as e:
        logger.exception(f"Error processing fetched data: {e}")
        st.error(f"An unexpected error occurred while processing data: {e}", icon="💥")
        return None

# --- MODIFIED: Function now accepts the status type to display ---
def create_building_map(df: pd.DataFrame, status_to_display: str):
    """Creates a Folium map with building markers colored by the selected status type."""
    if df is None or df.empty:
        st.warning("No building data available to display.")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    # --- Determine which status column and label to use ---
    if status_to_display == "WiFi Status":
        status_column = 'WifiStatus'
        status_label = "WiFi"
    elif status_to_display == "Audio Status":
        status_column = 'AudioStatus'
        status_label = "Audio"
    else:
        st.error("Invalid status type selected.")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    logger.info(f"Creating map for: {status_label} Status")

    # Drop rows with invalid coordinates (using lowercase column names)
    df_valid = df.dropna(subset=['Latitude', 'Longitude']).copy()
    if len(df_valid) < len(df):
        st.warning(f"Excluded {len(df) - len(df_valid)} buildings due to missing coordinates.", icon="⚠️")

    if df_valid.empty:
        st.warning("No buildings with valid coordinates found.")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    # Create GeoDataFrame (using lowercase column names)
    try:
        gdf = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid.Longitude, df_valid.Latitude),
            crs="EPSG:4326"
        )
        logger.info(f"Created GeoDataFrame with {len(gdf)} buildings.")
    except Exception as e:
        logger.exception("Failed to create GeoDataFrame.")
        st.error(f"Error creating geographic data: {e}", icon="❌")
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13)

    # Calculate map center or use default
    center_lat = gdf.geometry.y.mean() if not gdf.empty else DEFAULT_MAP_CENTER[0]
    center_lon = gdf.geometry.x.mean() if not gdf.empty else DEFAULT_MAP_CENTER[1]
    map_center = [center_lat, center_lon]

    m = folium.Map(location=map_center, zoom_start=DEFAULT_ZOOM, control_scale=True)

    # Add Google Maps Tile Layer (Kept from original)
    google_maps_tile = folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="© Google Maps",
        name="Google Maps Street" # Renamed slightly for clarity
    )
    google_maps_tile.add_to(m)

    color_map = {'active': 'green', 'inactive': 'red'}

    # Add markers to the map
    for idx, row in gdf.iterrows():
        # Use lowercase 'building_name' from DataFrame
        building_name = row['BuildingName']
        # --- Get status from the CORRECT column based on selection ---
        status = row[status_column]
        color = color_map.get(str(status).lower(), 'gray') # Ensure status is string and lowercase

        # --- Updated popup to show BOTH statuses for context ---
        popup_html = f"""
        <b>Building:</b> {building_name}<br>
        <hr style='margin: 3px 0;'>
        <b>{status_label} Status: <span style="color:{color};">{str(status).capitalize()}</span></b><br>
        <hr style='margin: 3px 0;'>
        <i>WiFi: {str(row['WifiStatus']).capitalize()}</i><br>
        <i>Audio: {str(row['AudioStatus']).capitalize()}</i><br>
        <i>Coords: ({row.geometry.y:.4f}, {row.geometry.x:.4f})</i>
        """
        popup = folium.Popup(popup_html, max_width=300)

        # --- Updated tooltip to reflect the selected status ---
        tooltip_text = f"{building_name} ({status_label}: {str(status).capitalize()})"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=20, # Kept large radius from original
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=tooltip_text
        ).add_to(m)

    logger.info(f"Added {len(gdf)} building markers to Folium map for {status_label} status.")
    return m


# --- Streamlit App Layout ---
st.title("🏫 TTU Building Systems Status Map") # Updated title

# --- Sidebar or Main Area for Controls ---
# Place controls together for better UX
with st.container():
    col1, col2, col3 = st.columns([2,2,1]) # Adjust column ratios as needed
    with col1:
        # --- NEW: Dropdown for status type selection ---
        selected_status = st.selectbox(
            "Select Status Type:",
            options=STATUS_TYPES,
            index=0 # Default to WiFi Status
        )
    with col2:
        # Keep the checkbox for filtering
        only_inactive = st.checkbox(f"Show only inactive {selected_status.split(' ')[0]} buildings", value=False)
    with col3:
         # Keep refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear() # Clear cache
            st.rerun() # Rerun the script to fetch again and redraw

st.markdown(f"Displays **{selected_status}** for university buildings. Hover over markers for details.")

# --- Fetch Data (only once) ---
building_data_df_original = fetch_data(API_ENDPOINT)

# --- Display Map and Summary ---
if building_data_df_original is not None:
    st.success(f"Loaded status data for {len(building_data_df_original)} buildings.", icon="✅")

    # --- Determine the column to use based on selection ---
    if selected_status == "WiFi Status":
        status_col_for_display = 'WifiStatus'
    else: # Audio Status
        status_col_for_display = 'AudioStatus'

    # --- Calculate Summary Metrics based on SELECTED status ---
    # Ensure the column exists and handle potential non-string types before value_counts
    if status_col_for_display in building_data_df_original.columns:
         # Convert to string and lower for consistent counting
         status_counts = building_data_df_original[status_col_for_display].astype(str).str.lower().value_counts()
         active_count = status_counts.get('active', 0)
         inactive_count = status_counts.get('inactive', 0)
    else:
         active_count = 0
         inactive_count = 0
         st.warning(f"Status column '{status_col_for_display}' not found for summary calculation.")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        # Use selected status name in metric label
        st.metric(f"Active {selected_status.split(' ')[0]} Buildings", active_count)
    with metric_col2:
        st.metric(f"Inactive {selected_status.split(' ')[0]} Buildings", inactive_count)


    # --- Filter Data for Map Display if checkbox is ticked ---
    df_to_map = building_data_df_original.copy() # Start with the full dataset
    if only_inactive:
        # Filter based on the SELECTED status column
        df_to_map = df_to_map[df_to_map[status_col_for_display].astype(str).str.lower() == 'inactive']
        st.info(f"Filtering map to show only buildings with inactive {selected_status.split(' ')[0]}.", icon="ℹ️")


    # --- Display Interactive Map ---
    st.subheader(f"Building {selected_status} Map")
    # Pass the dataframe (potentially filtered) and the selected status type
    folium_map = create_building_map(df_to_map, selected_status)
    st_folium(folium_map, width=None, height=800, returned_objects=[])


    # --- Optional: Display Raw Data Table ---
    # This expander now shows active/inactive lists based on the SELECTED status
    with st.expander(f"Show {selected_status.split(' ')[0]} Building Lists", expanded=False):
        list_col1, list_col2 = st.columns(2)
        # Use the original DataFrame for the lists, filtered by the selected status column
        active_df = building_data_df_original[
            building_data_df_original[status_col_for_display].astype(str).str.lower() == 'active'
        ][['BuildingName']].reset_index(drop=True) # Select only name, reset index for cleaner display

        inactive_df = building_data_df_original[
            building_data_df_original[status_col_for_display].astype(str).str.lower() == 'inactive'
        ][['BuildingName']].reset_index(drop=True) # Select only name

        with list_col1:
            st.subheader(f"Active {selected_status.split(' ')[0]} Buildings ({len(active_df)})")
            st.dataframe(active_df, use_container_width=True)
        with list_col2:
            st.subheader(f"Inactive {selected_status.split(' ')[0]} Buildings ({len(inactive_df)})")
            st.dataframe(inactive_df, use_container_width=True)

else:
    st.warning("Could not load building data to display the map.")
