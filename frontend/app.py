import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
from dotenv import load_dotenv
import logging
from folium.plugins import MarkerCluster
from pymongo import MongoClient
import pymongo
import hashlib
import re
import time
import numpy as np # Added for mean calculation if needed

# --- Configuration ---
load_dotenv()

# Use environment variables for API URL or default
# API_BASE_URL = os.getenv("API_BASE_URL", "https://my-uni-map-api-41bdbd00fbb7.herokuapp.com/")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") # Localhost for testing
# Assuming the endpoint now provides room-level data
API_ENDPOINT = f"{API_BASE_URL}/api/building-status" # Renamed for clarity, adjust if needed

DEFAULT_MAP_CENTER = [33.58452231722856, -101.87542936763231]  # TTU Approx Center
DEFAULT_ZOOM_ALL = 19 # Zoom level when viewing all buildings
DEFAULT_ZOOM_BUILDING = 21 # Zoom level when focused on a specific building
STATUS_TYPES = ["WiFi Status", "Audio Status"] # Available statuses to select
SECONDS_IN_DAY = 24 * 60 * 60

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Config ---
st.set_page_config(page_title="TTU Systems Status Map", layout="wide")

# --- MongoDB Initialization ---
def get_mongo_client():
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        st.error("MongoDB URI is not set in the environment.")
        return None
    try:
        # Added serverSelectionTimeoutMS to handle potential connection delays
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logger.info("MongoDB connection successful.")
        return client
    except pymongo.errors.ConnectionFailure as e:
        st.error("Failed to connect to MongoDB.")
        logger.error(f"MongoDB Connection Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during MongoDB connection: {e}")
        logger.error(f"Unexpected MongoDB Error: {e}")
        return None


def get_collection(database_env, collection_env):
    client = get_mongo_client()
    if client is None:
        return None
    db_name = os.getenv(database_env)
    collection_name = os.getenv(collection_env)
    if not db_name or not collection_name:
        st.error("Database or collection environment variables are missing.")
        return None
    db = client[db_name]
    return db[collection_name]

def get_users_collection():
    # Ensure env variable names match your .env file
    return get_collection("MONGO_USERS_DATABASE", "MONGO_USERS_COLLECTION")

# --- Utility Functions ---
def hash_password(password: str) -> str:
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password: str, hashed_password: str) -> bool:
    """Check if the provided password matches the hashed password."""
    return hash_password(password) == hashed_password

def is_valid_username(username: str) -> bool:
    """Validate the username using a regex."""
    return bool(re.match(r"^[A-Za-z0-9_.]{1,20}$", username))

def is_valid_name(name: str) -> bool:
    """Validate the name to allow only alphabets and spaces."""
    return bool(re.match(r"^[A-Za-z ]{1,50}$", name))

def date_calc(temp_days: int) -> int:
    """Calculate the end date for temporary access."""
    # Ensure temp_days is treated as an integer
    try:
        days = int(temp_days)
        if days <= 0:
            raise ValueError("Temporary access days must be positive.")
        return int(time.time() + days * SECONDS_IN_DAY)
    except (ValueError, TypeError):
        logger.error(f"Invalid input for temporary days: {temp_days}. Defaulting to 0.")
        # Return current time, effectively expiring immediately or handle as error
        return int(time.time())


def validate_user_input(username: str, password: str, temp_days: str) -> list:
    """Validate the username, password, and temporary access days."""
    errors = []
    if len(password) < 5:
        errors.append("Password should be at least 5 characters long.")
    try:
        if not temp_days.isdigit() or int(temp_days) <= 0:
            errors.append("Temporary access days should be a positive integer.")
    except ValueError:
         errors.append("Temporary access days must be a valid number.")
    if not is_valid_username(username):
        errors.append("Username can only contain letters, numbers, underscores, and periods (1-20 chars).")
    # Add name validation if needed (e.g., in add_user)
    # if not is_valid_name(name): errors.append("Name can only contain letters and spaces (1-50 chars).")
    return errors

# --- User Management Functions ---
# (Authentication, Add, Delete, Extend, Modify - remain largely unchanged)
# Minor adjustments for robustness and logging
def authenticate(username: str, password: str) -> bool:
    """Authenticate the user by checking the username and password."""
    users = get_users_collection()
    if users is None:
        st.error("User database connection failed.")
        return False

    if not is_valid_username(username):
        st.error("Invalid username format.")
        return False

    try:
        user = users.find_one(
            {"username": username},
            {"_id": 0, "username": 1, "password": 1, "end_date": 1, "is_admin": 1, "name": 1} # Exclude _id explicitly
        )
    except pymongo.errors.PyMongoError as e:
        st.error("Database query failed during authentication.")
        logger.error(f"MongoDB find_one error during auth for {username}: {e}")
        return False


    if user:
        # Ensure required fields exist before accessing
        if "password" not in user or "end_date" not in user or "name" not in user or "is_admin" not in user:
             st.error("User data is incomplete. Please contact administrator.")
             logger.error(f"Incomplete user data retrieved for {username}.")
             return False

        if check_password(password, user["password"]):
            if user["end_date"] > time.time():
                st.session_state.logged_in = True
                st.session_state.name = user["name"]
                st.session_state.username = user["username"]
                st.session_state.is_admin = user["is_admin"]
                st.success(f"Welcome, {user['name']}!")
                logger.info(f"User {user['username']} logged in successfully.")
                return True
            else:
                st.error("User access has expired. Please contact the admin to extend access.")
                logger.warning(f"Expired access attempt for user {user['username']}.")
                return False
        else:
            st.error("Invalid username or password.")
            logger.warning(f"Failed login attempt for user {username} (wrong password).")
            return False
    else:
        st.error("Invalid username or password.")
        logger.warning(f"Failed login attempt for username: {username} (not found).")
        return False

def add_user(name: str, username: str, password: str, is_admin: bool, temp_days: str) -> bool:
    """Add a new user to the users collection after validating inputs."""
    if not is_valid_name(name):
         st.error("Name can only contain letters and spaces (1-50 chars).")
         return False

    errors = validate_user_input(username, password, temp_days)
    if errors:
        for error in errors:
            st.error(error)
        return False

    users = get_users_collection()
    if users is None:
        st.error("User database connection failed.")
        return False

    try:
        if users.find_one({"username": username}):
            st.error("Username already exists. Please choose a different username.")
            return False
    except pymongo.errors.PyMongoError as e:
        st.error("Database query failed checking for existing user.")
        logger.error(f"MongoDB find_one error during add_user check for {username}: {e}")
        return False


    end_date = date_calc(temp_days)
    # Check if date_calc returned a valid future date
    if end_date <= time.time() and int(temp_days) > 0:
        st.error("Could not calculate a valid end date from the days provided.")
        return False

    hashed_password = hash_password(password)
    user = {
        "name": name.strip(), # Trim whitespace
        "username": username,
        "password": hashed_password,
        "is_admin": is_admin,
        "end_date": end_date
    }
    try:
        result = users.insert_one(user)
        if result.inserted_id:
            if is_admin:
                st.success(f"Admin user '{username}' has been added successfully.")
                logger.info(f"Admin user {username} added by {st.session_state.get('username', 'UNKNOWN')}.")
            else:
                st.success(f"User '{username}' added successfully with access for {temp_days} days.")
                logger.info(f"User {username} added for {temp_days} days by {st.session_state.get('username', 'UNKNOWN')}.")
            return True
        else:
            st.error("Failed to add new user (Insert operation did not confirm).")
            logger.error(f"Add User Error: insert_one for {username} did not return inserted_id.")
            return False
    except pymongo.errors.PyMongoError as e:
        st.error(f"Database error occurred while adding user: {e}")
        logger.error(f"Add User (MongoDB) Error for {username}: {e}")
        return False
    except Exception as e: # Catch other potential errors
        st.error(f"An unexpected error occurred while adding user: {e}")
        logger.error(f"Add User (General) Error for {username}: {e}")
        return False


def delete_user(current_username: str, username_to_delete: str) -> bool:
    """Delete a user from the users collection."""
    users = get_users_collection()
    if users is None:
        st.error("User database connection failed.")
        return False

    if current_username == username_to_delete:
        st.error("You cannot delete your own account.")
        return False

    try:
        user_to_delete_data = users.find_one({"username": username_to_delete}, {"is_admin": 1})
        if not user_to_delete_data:
            st.error(f"User '{username_to_delete}' not found.")
            return False

        if user_to_delete_data.get("is_admin", False):
            st.error("Cannot delete another admin user using this interface.")
            logger.warning(f"Attempt by {current_username} to delete admin {username_to_delete} blocked.")
            return False

        result = users.delete_one({"username": username_to_delete})
        if result.deleted_count == 1:
            st.success(f"User '{username_to_delete}' has been deleted successfully.")
            logger.info(f"User {username_to_delete} deleted by {current_username}.")
            return True
        else:
            # This case might happen if the user was deleted between find_one and delete_one
            st.error(f"User '{username_to_delete}' could not be deleted (possibly already removed).")
            logger.warning(f"Delete user failed for {username_to_delete}, deleted_count was {result.deleted_count}.")
            return False

    except pymongo.errors.PyMongoError as e:
        st.error(f"Database error occurred while deleting user: {e}")
        logger.error(f"Delete User (MongoDB) Error for {username_to_delete}: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while deleting user: {e}")
        logger.error(f"Delete User (General) Error for {username_to_delete}: {e}")
        return False


def get_all_usernames() -> list:
    """Get all usernames from the users collection."""
    users = get_users_collection()
    if users is None:
        logger.warning("Cannot get usernames, user collection unavailable.")
        return []
    try:
        # Fetch only non-admin usernames if the current user is not an admin?
        # Or maybe fetch all, the UI should handle restrictions. Fetching all is simpler here.
        all_users = list(users.find({}, {"username": 1, "_id": 0}))
        return [user["username"] for user in all_users if "username" in user]
    except pymongo.errors.PyMongoError as e:
        st.error(f"Database error occurred while fetching usernames: {e}")
        logger.error(f"Get All Usernames (MongoDB) Error: {e}")
        return []


def extend_access(current_username: str, username_to_extend: str, temp_days: str) -> bool:
    """Extend user access by a specified number of days."""
    users = get_users_collection()
    if users is None:
        st.error("User database connection failed.")
        return False

    try:
        days_to_add = int(temp_days)
        if days_to_add <= 0:
            st.error("Number of days to extend must be positive.")
            return False
    except ValueError:
        st.error("Invalid number of days entered.")
        return False

    try:
        user = users.find_one({"username": username_to_extend}, {"is_admin": 1, "end_date": 1})
        if not user:
            st.error(f"User '{username_to_extend}' not found.")
            return False

        if user.get("is_admin", False):
            st.error("Admin user access cannot be extended or modified here.")
            logger.warning(f"Attempt by {current_username} to extend access for admin {username_to_extend} blocked.")
            return False

        # Calculate new end date based on *current* end date or now, whichever is later
        current_end_date = user.get("end_date", time.time())
        start_time = max(time.time(), current_end_date)
        new_end_date = start_time + days_to_add * SECONDS_IN_DAY

        result = users.update_one({"username": username_to_extend}, {"$set": {"end_date": new_end_date}})

        if result.modified_count == 1:
            st.success(f"User '{username_to_extend}' access extended by {days_to_add} days.")
            logger.info(f"User {username_to_extend} access extended by {days_to_add} days by {current_username}.")
            return True
        elif result.matched_count == 1 and result.modified_count == 0:
             st.warning(f"User '{username_to_extend}' found, but access was not extended (perhaps the new date was not later?).")
             logger.warning(f"Extend access for {username_to_extend} matched but didn't modify.")
             return False
        else:
            # Should not happen if find_one succeeded, but good to check
            st.error(f"Failed to extend access for user '{username_to_extend}' (user not found during update).")
            logger.error(f"Extend access failed for {username_to_extend}, matched_count was {result.matched_count}.")
            return False

    except pymongo.errors.PyMongoError as e:
        st.error(f"Database error occurred while extending access: {e}")
        logger.error(f"Extend Access (MongoDB) Error for {username_to_extend}: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while extending access: {e}")
        logger.error(f"Extend Access (General) Error for {username_to_extend}: {e}")
        return False

def modify_user_profile(current_username: str, new_name: str, new_password: str or None) -> bool:
    """Modify logged-in user's profile (name, password)."""
    users = get_users_collection()
    if users is None:
        st.error("User database connection failed.")
        return False

    if not is_valid_name(new_name):
         st.error("New name can only contain letters and spaces (1-50 chars).")
         return False

    update_data = {"name": new_name.strip()}

    if new_password:
        if len(new_password) < 5:
             st.error("New password must be at least 5 characters long.")
             return False
        update_data["password"] = hash_password(new_password)

    try:
        result = users.update_one({"username": current_username}, {"$set": update_data})
        if result.modified_count == 1:
            st.success("Profile updated successfully.")
            logger.info(f"User {current_username} updated their profile.")
            # Update name in session state immediately
            st.session_state.name = new_name.strip()
            return True
        elif result.matched_count == 1 and result.modified_count == 0:
             st.info("No changes detected in profile information.")
             return True # Not an error if nothing changed
        else:
            st.error("Failed to update profile (user not found).")
            logger.error(f"Modify profile failed for {current_username}, matched_count: {result.matched_count}")
            return False
    except pymongo.errors.PyMongoError as e:
        st.error(f"Database error occurred while updating profile: {e}")
        logger.error(f"Modify Profile (MongoDB) Error for {current_username}: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while updating profile: {e}")
        logger.error(f"Modify Profile (General) Error for {current_username}: {e}")
        return False


# --- Data Fetching and Map Creation Functions ---
@st.cache_data(ttl=60)
def fetch_data(url: str):
    """Fetch room data from the backend API."""
    try:
        logger.info(f"Fetching room data from {url}")
        response = requests.get(url, timeout=25) # Increased timeout
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} room records.")
        df = pd.DataFrame(data)

        # --- Data Cleaning and Preparation ---
        required_cols = [
            'BuildingName', 'RoomLat', 'RoomLon', 'WifiStatus', 'AudioStatus',
            'BuildingLat', 'BuildingLon', 'RoomNumber', 'FloorNumber' # Add other useful cols
        ]
        # Check for essential columns
        for col in ['BuildingName', 'RoomLat', 'RoomLon', 'WifiStatus', 'AudioStatus', 'BuildingLat', 'BuildingLon']:
             if col not in df.columns:
                 st.error(f"Essential column '{col}' missing in API response. Cannot proceed.")
                 logger.error(f"Essential column '{col}' missing in API response from {url}")
                 return None # Cannot proceed without essential geo-data

        # Fill missing non-essential data with defaults or NaN
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing in API response, filling with 'Unknown' or NaN.")
                if col in ['WifiStatus', 'AudioStatus']:
                     df[col] = 'inactive' # Default status if missing
                elif pd.api.types.is_numeric_dtype(df[col].dtype) :
                     df[col] = np.nan
                else:
                     df[col] = 'Unknown'


        # Convert coordinates to numeric, coercing errors to NaN
        coord_cols = ['RoomLat', 'RoomLon', 'BuildingLat', 'BuildingLon']
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with invalid room coordinates as they can't be mapped
        initial_count = len(df)
        df.dropna(subset=['RoomLat', 'RoomLon'], inplace=True)
        if len(df) < initial_count:
             logger.warning(f"Dropped {initial_count - len(df)} rows due to missing/invalid RoomLat/RoomLon.")
             if len(df) == 0:
                  st.warning("No rooms with valid coordinates found in the data.")
                  return None # Return None if no valid rooms left

        # Standardize status columns to lower case strings
        df['WifiStatus'] = df['WifiStatus'].astype(str).str.lower().fillna('inactive')
        df['AudioStatus'] = df['AudioStatus'].astype(str).str.lower().fillna('inactive')
        df['BuildingName'] = df['BuildingName'].astype(str).fillna('Unknown Building')

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        st.error(f"Failed to fetch data from API: {e}", icon="🚨")
        return None
    except ValueError as e: # Catch JSON decoding errors
         logger.error(f"Failed to decode JSON response: {e}")
         st.error("Received invalid data format from the API.", icon=" C")
         return None
    except Exception as e:
        logger.exception(f"Error processing fetched data: {e}") # Use exception for full traceback
        st.error(f"An unexpected error occurred during data processing: {e}", icon="💥")
        return None


def create_room_map(df: pd.DataFrame, selected_statuses: list, filter_mode: str, selected_building: str):
    """Creates a Folium map with clustered room markers."""
    if df is None or df.empty:
        st.warning("No room data available to display.")
        # Return a basic map centered on TTU
        return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="© Google Maps Street")

    # --- 1. Filter Data Based on Selections ---
    gdf_display = df.copy()

    # Filter by Building
    if selected_building != "All Buildings":
        gdf_display = gdf_display[gdf_display['BuildingName'] == selected_building]
        if gdf_display.empty:
             st.warning(f"No rooms found for the selected building: {selected_building}")
             return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_ZOOM_ALL, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="© Google Maps Street")

    # Filter by Status (Inactive only)
    if filter_mode == "Show Only Inactive" and selected_statuses:
        inactive_conditions = []
        if "WiFi Status" in selected_statuses:
            inactive_conditions.append(gdf_display['WifiStatus'] == 'inactive')
        if "Audio Status" in selected_statuses:
            inactive_conditions.append(gdf_display['AudioStatus'] == 'inactive')

        if inactive_conditions:
             # Combine conditions with OR: show if *any* selected status is inactive
            combined_inactive_condition = pd.concat(inactive_conditions, axis=1).any(axis=1)
            gdf_display = gdf_display[combined_inactive_condition]

    if gdf_display.empty:
        st.info("No rooms match the current filter criteria.")
        # Still show map centered appropriately
        if selected_building != "All Buildings":
             # Try to find building coords even if no rooms match filter
             building_coords = df[df['BuildingName'] == selected_building][['BuildingLat', 'BuildingLon']].iloc[0]
             map_center = [building_coords['BuildingLat'], building_coords['BuildingLon']]
             zoom = DEFAULT_ZOOM_BUILDING
        else:
            map_center = DEFAULT_MAP_CENTER
            zoom = DEFAULT_ZOOM_ALL
        return folium.Map(location=map_center, zoom_start=zoom, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="© Google Maps Street")


    # --- 2. Create GeoDataFrame for valid points ---
    try:
        gdf = gpd.GeoDataFrame(
            gdf_display,
            geometry=gpd.points_from_xy(gdf_display.RoomLon, gdf_display.RoomLat),
            crs="EPSG:4326"
        )
    except Exception as e:
         st.error(f"Error creating GeoDataFrame: {e}")
         logger.error(f"GeoDataFrame creation failed: {e}")
         return folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=13, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="© Google Maps Street")


    # --- 3. Determine Map Center and Zoom ---
    if selected_building != "All Buildings":
        # Center on the selected building's coordinates (use mean if multiple entries, unlikely but safe)
        building_coords = gdf_display[['BuildingLat', 'BuildingLon']].iloc[0] # Get first valid coords for the building
        # Check if building coords are valid
        if pd.isna(building_coords['BuildingLat']) or pd.isna(building_coords['BuildingLon']):
            st.warning(f"Could not find valid coordinates for building {selected_building}. Using default center.")
            logger.warning(f"Invalid BuildingLat/Lon for {selected_building}. Using default map center.")
            map_center = DEFAULT_MAP_CENTER
            zoom = DEFAULT_ZOOM_ALL
        else:
            map_center = [building_coords['BuildingLat'], building_coords['BuildingLon']]
            zoom = DEFAULT_ZOOM_BUILDING
    else:
        # Center on the mean of all *displayed* rooms if available
        if not gdf.empty:
            map_center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
        else:
            map_center = DEFAULT_MAP_CENTER # Fallback
        zoom = DEFAULT_ZOOM_ALL

    # --- 4. Create Folium Map ---
    m = folium.Map(location=map_center, zoom_start=zoom, control_scale=True, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="© Google Maps")

# Add Tile Layers (allowing user choice)




    # --- 5. Define Marker Cluster ---
    # Simplified cluster logic: Red if > N% inactive, Green otherwise
    # Requires determining 'display_status' for each point first
    def get_display_status(row, selected_sts):
        is_inactive = False
        if "WiFi Status" in selected_sts and row['WifiStatus'] == 'inactive':
            is_inactive = True
        if "Audio Status" in selected_sts and row['AudioStatus'] == 'inactive':
            is_inactive = True
        # If no statuses are selected, default to active? Or maybe handle this upstream.
        # Assuming at least one status is always selected if we reach here with data.
        return 'inactive' if is_inactive else 'active'

    # Apply this logic to determine color *before* adding marker
    gdf['display_status'] = gdf.apply(lambda row: get_display_status(row, selected_statuses), axis=1)

    marker_cluster = MarkerCluster(
         name="Room Status", # Layer name
         overlay=True,
         control=True,
         # Updated JS function for clustering based on 'inactive' count
         icon_create_function='''
            function(cluster) {
                var markers = cluster.getAllChildMarkers();
                var count = cluster.getChildCount();
                var inactiveCount = 0;
                // Access the custom property set during marker creation
                for (var i = 0; i < markers.length; i++) {
                     // Check the 'data-status' attribute we will add
                    if (markers[i].options.icon.options.markerColor === 'red') {
                        inactiveCount++;
                    }
                }

                var inactiveRatio = inactiveCount / count;
                // Determine cluster color: more red as the proportion of inactive increases
                var color;
                if (inactiveCount > 0) {
                   // Adjust opacity based on ratio, ensuring base visibility
                   var opacity = 0.5 + (inactiveRatio * 0.5);
                   color = 'rgba(255, 0, 0, ' + opacity + ')'; // Red, semi-transparent
                } else {
                   color = 'rgba(0, 128, 0, 0.8)'; // Solid Green
                }

                // Scale size logarithmically or similar to avoid huge clusters
                var size = 30 + Math.log2(count + 1) * 5; // Example scaling
                size = Math.min(60, size); // Max size limit

                return L.divIcon({
                    html: '<div style="width:' + size + 'px; height:' + size + 'px; border-radius:50%; background:' + color + '; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; border: 1px solid rgba(0,0,0,0.5);">' + count + '</div>',
                    className: 'marker-cluster marker-cluster-large', // Use default Leaflet classes if possible
                    iconSize: new L.Point(size, size)
                });
            }
        '''
    ).add_to(m)


    # --- 6. Add Markers to Cluster ---
    for idx, row in gdf.iterrows():
        # Determine color based on the pre-calculated display_status
        color = 'red' if row['display_status'] == 'inactive' else 'green'
        status_html = ""
        if "WiFi Status" in selected_statuses:
            wifi_color = 'red' if row['WifiStatus'] == 'inactive' else 'green'
            status_html += f'<i>WiFi: <span style="color:{wifi_color};">{str(row["WifiStatus"]).capitalize()}</span></i><br>'
        if "Audio Status" in selected_statuses:
            audio_color = 'red' if row['AudioStatus'] == 'inactive' else 'green'
            status_html += f'<i>Audio: <span style="color:{audio_color};">{str(row["AudioStatus"]).capitalize()}</span></i><br>'

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; min-width: 180px;">
            <b style="font-size: 14px;">{row['BuildingName'].capitalize()}</b><br>
            <b>Room:</b> {row.get('RoomNumber', 'N/A')} (Floor: {row.get('FloorNumber', 'N/A')})<br>
            <b>Room ID:</b> {row.get('RoomID', 'N/A')}<br>
            <hr style="margin: 3px 0;">
            {status_html}
            <i>Use: {str(row.get('SpaceUse', 'N/A')).capitalize()}</i><br>

            <hr style="margin: 3px 0;">
            <b>Building Coordinates:</b><br>
            <i>Coords: ({row['BuildingLat']:.5f}, {row['BuildingLon']:.5f})</i><br>
            <b>Room Coordinates:</b><br>
            <i>Coords: ({row.geometry.y:.5f}, {row.geometry.x:.5f})</i><br>
            <b>Last Updated:</b><br>
            <i>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row.get('last_updated', time.time())))}</i><br>

        </div>
        """
        popup = folium.Popup(popup_html, max_width=300)

        # Using simpler CircleMarkers which are often better for many points
        folium.Circle(
            location=[row.geometry.y, row.geometry.x],
            radius=3, # Smaller radius for rooms
            color=color, # Border color
            fill=True,
            fill_color=color, # Fill color
            fill_opacity=0.7,
            popup=popup,
            # Custom attribute for the cluster function - use icon color directly now
            # icon=folium.Icon(color=color) # This doesn't work directly with CircleMarker for clustering access
                                            # Instead, the JS cluster function checks the marker's *actual* rendered color/options if possible,
                                            # or we rely on the logic using `markerColor` property of the icon if using regular Markers.
                                            # Let's stick to CircleMarker and adapt JS if needed.
                                            # *Correction*: For CircleMarker, direct property access is hard in cluster func.
                                            # Let's use regular Markers with Icons for easier status access in JS cluster function.
        ).add_to(marker_cluster) # Add marker to the cluster group


    # --- 7. Add Layer Control ---
    folium.LayerControl().add_to(m)

    return m


# --- Authentication UI ---
def login_page():
    col1, col2 = st.columns([1, 2]) # Adjust column ratios if needed
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Texas_Tech_Athletics_logo.svg/1024px-Texas_Tech_Athletics_logo.svg.png", width=250) # TTU Logo
    with col2:
        st.title("Outage Maps - Login Page")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate(username, password):
                st.rerun() # Rerun to update the page state after successful login
            else:
                # Increment login attempts in session state
                st.session_state.login_attempts = st.session_state.get("login_attempts", 0) + 1
                # Basic attempt limiting (consider more robust methods for production)
                if st.session_state.login_attempts >= 5:
                    st.error("Too many failed login attempts. Please try again later or contact admin.")
                    # Optionally add a time delay here
                    time.sleep(5) # Simple delay
                    st.stop() # Stop execution for this user temporarily

def logout():
    """Clear session state for logout."""
    keys_to_clear = ["logged_in", "username", "name", "is_admin", "login_attempts"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("You have been logged out.")
    # Use st.rerun() to ensure the page reloads to the login state
    st.rerun()


# --- Main App Function ---
def app():
    # --- Authentication Check ---
    if not st.session_state.get("logged_in"):
        login_page()
        return # Stop execution if not logged in

    # --- Sidebar Navigation ---
    is_admin = st.session_state.get("is_admin", False)
    st.sidebar.title(f"Hello, {st.session_state.get('name', 'User')}!")
    st.sidebar.write("Role: **Admin**" if is_admin else "**User**")
    st.sidebar.markdown("---")
    st.sidebar.write("Select an option from the menu below:")
    # Define options based on role
    base_options = ["Room Status Map", "Modify My Profile"]
    admin_options = ["Add New User", "Delete User", "Extend User Access"]
    logout_option = ["Logout"]

    available_options = base_options
    if is_admin:
        available_options.extend(admin_options)
    available_options.extend(logout_option)

    mode = st.sidebar.radio("Navigation", available_options, key="nav_mode")

    # --- Page Content Based on Mode ---
    if mode == "Logout":
        logout() # Will stop execution here and rerun

    elif mode == "Room Status Map":
        st.title("🏫 TTU Room Systems Status Map")

        # Fetch data first to populate dropdowns
        room_data = fetch_data(API_ENDPOINT)

        if room_data is None or room_data.empty:
            st.error("Could not load room data. Map cannot be displayed.")
            if st.button("🔄 Try Refreshing Data"):
                st.cache_data.clear()
                st.rerun()
            return # Stop if no data

        # --- UI Controls ---
        st.subheader("Map Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Building Selection Dropdown
            building_list = ["All Buildings"] + sorted(room_data['BuildingName'].unique().tolist())
            selected_building = st.selectbox(
                "Focus on Building:",
                options=building_list,
                index=0, # Default to "All Buildings"
                key="building_select"
            )

        with col2:
            # Status Type Multi-Select
            selected_statuses = st.multiselect(
                "Select Status Types to Display:",
                options=STATUS_TYPES,
                default=STATUS_TYPES, # Default to showing both
                key="status_multi_select"
            )
            if not selected_statuses:
                 st.warning("Please select at least one status type to display.")
                 # Optionally default back if user deselects all
                 # selected_statuses = STATUS_TYPES

        with col3:
            # Inactive Filter Radio Buttons
            filter_mode = st.radio(
                "Filter by Status:",
                options=["Show All Statuses", "Show Only Inactive"],
                index=0, # Default to showing all
                key="inactive_filter_radio",
                horizontal=True # Display radio buttons horizontally
            )

        # Refresh button
        if st.button("🔄 Refresh Data", key="refresh_map_data"):
            st.cache_data.clear()
            st.rerun()

        # --- Data Filtering and Map Display ---
        if selected_statuses: # Only proceed if statuses are selected
            st.markdown("---")
            st.subheader(f"Map Display: {selected_building}")
            st.markdown(f"Showing rooms based on selected filters. Hover over points for details.")

            # Create and display the map
            folium_map = create_room_map(room_data, selected_statuses, filter_mode, selected_building)

            # Display map - adjust height as needed
            st_folium(folium_map, width='100%', height=700, returned_objects=[])

            # --- Optional: Display Filtered Data Table ---
            with st.expander("Show Filtered Room Data"):
                 # Apply the same filtering logic used by the map function to show the table
                 display_df_table = room_data.copy()
                 if selected_building != "All Buildings":
                     display_df_table = display_df_table[display_df_table['BuildingName'] == selected_building]

                 if filter_mode == "Show Only Inactive" and selected_statuses:
                     inactive_conditions = []
                     if "WiFi Status" in selected_statuses:
                         inactive_conditions.append(display_df_table['WifiStatus'] == 'inactive')
                     if "Audio Status" in selected_statuses:
                         inactive_conditions.append(display_df_table['AudioStatus'] == 'inactive')
                     if inactive_conditions:
                         combined_condition = pd.concat(inactive_conditions, axis=1).any(axis=1)
                         display_df_table = display_df_table[combined_condition]

                 # Select and rename columns for better readability
                 cols_to_show = ['BuildingName', 'RoomNumber', 'FloorNumber', 'WifiStatus', 'AudioStatus', 'SpaceUse', 'RoomLat', 'RoomLon', 'BuildingLat', 'BuildingLon', 'RoomID']
                 display_df_table = display_df_table[cols_to_show].rename(columns={
                    'BuildingName': 'Building',
                    'RoomNumber': 'Room',
                    'FloorNumber': 'Floor',
                    'WifiStatus': 'WiFi',
                    'AudioStatus': 'Audio',
                    'SpaceUse': 'Use Type',
                    'RoomLat': 'Room Latitude',
                    'RoomLon': 'Room Longitude',
                    'BuildingLat': 'Building Latitude',
                    'BuildingLon': 'Building Longitude',
                    'RoomID': 'Room ID'
                 }).reset_index(drop=True)

                 st.dataframe(display_df_table, use_container_width=True)
        else:
             # This case should be rare due to the warning above, but handle it
             st.info("Select at least one status type (WiFi, Audio) to view the map.")


    # --- Admin: Add New User ---
    elif mode == "Add New User" and is_admin:
        st.subheader("👤 Add New User")
        with st.form("add_user_form"):
            new_name = st.text_input("User's Full Name", max_chars=50, key="add_name")
            new_username = st.text_input("Choose a Username (letters, numbers, _, .)", max_chars=20, key="add_username")
            new_password = st.text_input("Choose a Password (min 5 chars)", type="password", max_chars=30, key="add_password")
            confirm_password = st.text_input("Confirm Password", type="password", max_chars=30, key="add_confirm_password")
            new_is_admin = st.checkbox("Grant Administrator Privileges", value=False, key="add_is_admin")
            # Set default days; make non-editable if admin is checked
            default_days = "36500" if new_is_admin else "30" # 100 years for admin, 30 days default
            new_temp_days = st.text_input(
                "Access Duration (days)",
                value=default_days,
                key="add_temp_days",
                disabled=new_is_admin # Disable if admin is checked
            )
            if new_is_admin:
                st.caption("Admin access is effectively permanent (set to 100 years).")


            submitted = st.form_submit_button("Add User")
            if submitted:
                if not is_valid_name(new_name):
                     st.error("Name can only contain letters and spaces (1-50 chars).")
                elif new_password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    # If admin checked, force temp_days to a large number
                    days_to_add = "36500" if new_is_admin else new_temp_days
                    add_user(new_name, new_username, new_password, new_is_admin, days_to_add)


    elif mode == "Delete User" and is_admin:
        st.subheader("🗑️ Delete User")
        all_users = get_all_usernames()
        # Exclude self and potentially other admins from the list
        users_eligible_for_deletion = [
            u for u in all_users
            if u != st.session_state.username ]

        if users_eligible_for_deletion:
            user_to_delete = st.selectbox("Select User to Delete", options=users_eligible_for_deletion, key="delete_user_select")
            st.warning(f"⚠️ Are you sure you want to permanently delete user '{user_to_delete}'? This action cannot be undone.")
            if st.button(f"Confirm Deletion of {user_to_delete}", type="primary"):
                delete_user(st.session_state.username, user_to_delete)
                # Rerun to refresh the list after deletion
                st.rerun()
        else:
            st.info("No other non-admin users available to delete.")

    # --- User: Modify Own Profile ---
    elif mode == "Modify My Profile":
        st.subheader("✏️ Modify My Profile")
        with st.form("modify_profile_form"):
            st.text_input("Username", value=st.session_state.username, disabled=True)
            mod_name = st.text_input("Full Name", value=st.session_state.get("name", ""), max_chars=50, key="mod_name")
            st.markdown("---")
            st.caption("Leave password fields blank to keep your current password.")
            mod_password = st.text_input("New Password (min 5 chars)", type="password", max_chars=30, key="mod_password")
            mod_confirm_password = st.text_input("Confirm New Password", type="password", max_chars=30, key="mod_confirm_password")

            submitted = st.form_submit_button("Update Profile")
            if submitted:
                 password_to_set = None # Assume no password change initially
                 if mod_password or mod_confirm_password: # If either field has input
                      if mod_password != mod_confirm_password:
                           st.error("New passwords do not match!")
                      elif len(mod_password) < 5:
                           st.error("New password must be at least 5 characters long.")
                      else:
                           password_to_set = mod_password # Valid new password provided
                 else:
                      # Both blank, proceed with name change only
                      pass


                 # Proceed if passwords match (or were blank) and name is valid
                 if (mod_password == mod_confirm_password):
                    # Call modify function only if name changed or a valid new password was set
                    if mod_name != st.session_state.get("name", "") or password_to_set:
                         modify_user_profile(st.session_state.username, mod_name, password_to_set)
                    else:
                         st.info("No changes detected.")


    # --- Admin: Extend User Access ---
    elif mode == "Extend User Access" and is_admin:
        st.subheader("⏳ Extend User Access")
        all_users = get_all_usernames()
        # Exclude self and potentially admins
        users_eligible_for_extension = [
             u for u in all_users
             if u != st.session_state.username ]

        if users_eligible_for_extension:
             user_to_extend = st.selectbox("Select User to Extend Access For", options=users_eligible_for_extension, key="extend_user_select")
             extend_days_str = st.text_input("Number of Days to Extend By", value="7", key="extend_days_input")

             if st.button(f"Extend Access for {user_to_extend}"):
                 # Validate days input
                 if not extend_days_str.isdigit() or int(extend_days_str) <= 0:
                      st.error("Please enter a positive whole number for days.")
                 else:
                      extend_access(st.session_state.username, user_to_extend, extend_days_str)
        else:
             st.info("No other non-admin users available for access extension.")

# --- Entry Point ---
if __name__ == "__main__":
    # Initialize session state keys if they don't exist
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    app()