import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Use Optional for statuses that might be missing/invalid
from pydantic import BaseModel, Field, validator # Pydantic v1 style validator
# from pydantic import field_validator # Use for Pydantic v2
from typing import List, Literal, Optional
import logging
from dotenv import load_dotenv
import numpy as np # Import numpy for NaN checks

# --- Configuration ---
load_dotenv()
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/data.xlsx") # Default to data.xlsx in parent directory

# --- Updated Column Names to Match New Excel Structure ---
# Check exact spelling and casing in your Excel file
BUILDING_NUMBER_COLUMN = "BuildingNumber"
BUILDING_COLUMN = "BuildingName"
FLOOR_NUMBER_COLUMN = "FloorNumber"
FLOOR_DESC_COLUMN = "FloorDesc"
ROOM_NUMBER_COLUMN = "RoomNumber"
SPACE_USE_CODE_COLUMN = "SpaceUseCode"
SPACE_USE_COLUMN = "SpaceUse"
IS_RAIDER_ROOM_COLUMN = "IsRaiderRoom" # Make sure this is the exact name if cut off
CAMPUS_COLUMN = "Campus"
ADDRESS_COLUMN = "Address"
BUILDING_LOCATION_COLUMN = "BuildingLocation" # Now seems to be a text field like 'main'
BUILDING_LAT_COLUMN = "BuildingLat"
BUILDING_LON_COLUMN = "BuildingLon" # Check if it's 'BuildingLon' or 'BuildingLor'
ROOM_ID_COLUMN = "RoomID"
ROOM_LAT_COLUMN = "RoomLat"
ROOM_LON_COLUMN = "RoomLon"
WIFI_STATUS_COLUMN = "WifiStatus"
AUDIO_STATUS_COLUMN = "AudioStatus" # Make sure this is the exact name if cut off

# Columns that seem to be missing in the new format (keep for potential future use or remove)
ENTITY_ID_COLUMN = "EntityID" # Keep optional in model if column might reappear
BUILDING_LOCATION_ID_COLUMN = "BuildingLocationID" # Keep optional in model

# --- Values can be reused ---
ACTIVE_STATUS_VALUE = "Active"
INACTIVE_STATUS_VALUE = "Inactive"
# Define the valid status literals
StatusLiteral = Literal['active', 'inactive']

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class BuildingStatus(BaseModel):
    entity_id: Optional[str] = Field(None, alias=ENTITY_ID_COLUMN) # Keep if needed, but likely None
    building_number: Optional[str] = Field(None, alias=BUILDING_NUMBER_COLUMN)
    building_name: str = Field(..., alias=BUILDING_COLUMN)

    # Floor and room information
    floor_number: Optional[str] = Field(None, alias=FLOOR_NUMBER_COLUMN)
    floor_desc: Optional[str] = Field(None, alias=FLOOR_DESC_COLUMN)
    room_number: str = Field(None, alias=ROOM_NUMBER_COLUMN)
    room_id: str = Field(None, alias=ROOM_ID_COLUMN) # Added RoomID

    # Space usage information
    space_use_code: Optional[str] = Field(None, alias=SPACE_USE_CODE_COLUMN)
    space_use: Optional[str] = Field(None, alias=SPACE_USE_COLUMN)
    # IsRaiderRoom might be numeric (1/0) or text in Excel
    is_raider_room: Optional[bool] = Field(None, alias=IS_RAIDER_ROOM_COLUMN)

    # Location information
    campus: Optional[str] = Field(None, alias=CAMPUS_COLUMN) # Added Campus
    address: Optional[str] = Field(None, alias=ADDRESS_COLUMN)
    # BuildingLocationID might be missing now, keep Optional
    building_location_id: Optional[str] = Field(None, alias=BUILDING_LOCATION_ID_COLUMN) # Keep if needed, but likely None
    building_location: Optional[str] = Field(None, alias=BUILDING_LOCATION_COLUMN) # Now seems text based

    # Coordinates - building level takes precedence, room level as fallback
    # Using BuildingLat/Lon as primary coordinates
    latitude: float = Field(..., alias=BUILDING_LAT_COLUMN)
    longitude: float = Field(..., alias=BUILDING_LON_COLUMN)
    # Keep room coordinates optional
    room_latitude: float = Field(None, alias=ROOM_LAT_COLUMN)
    room_longitude: float = Field(None, alias=ROOM_LON_COLUMN)

    # Status fields
    wifi_status: StatusLiteral = Field(..., alias=WIFI_STATUS_COLUMN)
    audio_status: StatusLiteral = Field(..., alias=AUDIO_STATUS_COLUMN)

    class Config:
        allow_population_by_field_name = True # Important for aliases


    # Pydantic v1 style validators
    @validator('latitude')
    def latitude_must_be_valid(cls, v):
        if v is None or pd.isna(v): # Check for None or NaN explicitly
             raise ValueError('Latitude cannot be missing')
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def longitude_must_be_valid(cls, v):
        if v is None or pd.isna(v): # Check for None or NaN explicitly
             raise ValueError('Longitude cannot be missing')
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    @validator('room_latitude')
    def room_latitude_must_be_valid(cls, v):
         # Allow None/NaN for room coordinates
        if v is not None and not pd.isna(v):
            if not -90 <= v <= 90:
                 raise ValueError('Room latitude must be between -90 and 90')
            return v
        return None # Return None if input is None or NaN

    @validator('room_longitude')
    def room_longitude_must_be_valid(cls, v):
         # Allow None/NaN for room coordinates
        if v is not None and not pd.isna(v):
             if not -180 <= v <= 180:
                 raise ValueError('Room longitude must be between -180 and 180')
             return v
        return None # Return None if input is None or NaN

# --- FastAPI App ---
app = FastAPI(
    title="University Building Systems Status API",
    description="Provides WiFi and Audio status and geographic coordinates for university buildings/rooms.",
    version="1.3.0"
)

# --- CORS Middleware --- (Adjust origins for production)
# Ensure your frontend URL is listed here if deployed
origins = [
    "http://localhost",
    "http://localhost:8501", # For local Streamlit development
    "https://my-uni-map-app-frontend-49f50cf71ca2.herokuapp.com", # Example deployed frontend
    # Add any other origins needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Function ---
def process_status(raw_status: str, building_name: str, status_type: str, index: int) -> StatusLiteral:
    """Processes a raw status string into 'active' or 'inactive'."""
    # Handle potential numeric types coming from Excel before lower()
    status = str(raw_status).strip().lower()
    if status == ACTIVE_STATUS_VALUE.lower():
        return 'active'
    elif status == INACTIVE_STATUS_VALUE.lower():
        return 'inactive'
    else:
        # Log unexpected or blank statuses, default to inactive
        logger.warning(f"Row {index+2} (Building: {building_name}): Unexpected {status_type} status '{raw_status}'. Treating as inactive.")
        return 'inactive'

def load_and_process_data(file_path: str) -> List[BuildingStatus]:
    """Loads building data, validates, and returns status (WiFi & Audio) with coordinates."""
    logger.info(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        raise HTTPException(status_code=500, detail=f"Data file not found: {file_path}")

    try:
        # Define columns to read as string explicitly, using NEW column names
        # Add any other ID-like or text columns that might be misinterpreted as numbers
        string_columns = {
            BUILDING_NUMBER_COLUMN: str,
            BUILDING_COLUMN: str,
            FLOOR_NUMBER_COLUMN: str, # Floor might be '0', 'Basement', etc.
            FLOOR_DESC_COLUMN: str,
            ROOM_NUMBER_COLUMN: str,
            ROOM_ID_COLUMN: str, # Added
            SPACE_USE_CODE_COLUMN: str,
            SPACE_USE_COLUMN: str,
            ADDRESS_COLUMN: str,
            CAMPUS_COLUMN: str, # Added
            BUILDING_LOCATION_COLUMN: str, # Now seems text
            WIFI_STATUS_COLUMN: str,
            AUDIO_STATUS_COLUMN: str,
            # Keep potentially missing columns here if needed for dtype, pandas handles missing cols
            ENTITY_ID_COLUMN: str,
            BUILDING_LOCATION_ID_COLUMN: str,
        }
        # Note: IS_RAIDER_ROOM_COLUMN is handled separately below

        if file_path.lower().endswith(".xlsx"):
            df = pd.read_excel(file_path, engine='openpyxl', dtype=string_columns)
        elif file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path, dtype=string_columns)
        else:
             raise HTTPException(status_code=500, detail="Unsupported data file format. Use .csv or .xlsx.")

        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        logger.info(f"Columns found: {df.columns.tolist()}")

        # --- Data Validation ---
        # Update required columns list based on the *essential* fields from the new format
        required_columns = [
            BUILDING_COLUMN,        # Essential for identification
            BUILDING_LAT_COLUMN,    # Essential for mapping
            BUILDING_LON_COLUMN,    # Essential for mapping
            WIFI_STATUS_COLUMN,     # Core data point
            AUDIO_STATUS_COLUMN,    # Core data point
            ROOM_LAT_COLUMN,
            ROOM_LON_COLUMN
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns in the data file: {missing_cols}. Please ensure the file at '{file_path}' contains these columns."
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # Fill NaN in status columns with empty string BEFORE processing
        # Use the NEW status column names
        df[WIFI_STATUS_COLUMN] = df[WIFI_STATUS_COLUMN].fillna('')
        df[AUDIO_STATUS_COLUMN] = df[AUDIO_STATUS_COLUMN].fillna('')
        # Fill NaN for other potentially missing text columns that should be empty string not None
        optional_text_cols = [ADDRESS_COLUMN, FLOOR_DESC_COLUMN, SPACE_USE_COLUMN, CAMPUS_COLUMN, BUILDING_LOCATION_COLUMN]
        for col in optional_text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')


        # Convert numeric columns (Lat/Lon) using NEW column names
        # Using errors='coerce' will turn unparseable values into NaT/NaN
        df[BUILDING_LAT_COLUMN] = pd.to_numeric(df[BUILDING_LAT_COLUMN], errors='coerce')
        df[BUILDING_LON_COLUMN] = pd.to_numeric(df[BUILDING_LON_COLUMN], errors='coerce')
        # Only convert room coords if columns exist
        if ROOM_LAT_COLUMN in df.columns:
             df[ROOM_LAT_COLUMN] = pd.to_numeric(df[ROOM_LAT_COLUMN], errors='coerce')
        else:
             df[ROOM_LAT_COLUMN] = np.nan # Add column as NaN if it doesn't exist
        if ROOM_LON_COLUMN in df.columns:
             df[ROOM_LON_COLUMN] = pd.to_numeric(df[ROOM_LON_COLUMN], errors='coerce')
        else:
             df[ROOM_LON_COLUMN] = np.nan # Add column as NaN if it doesn't exist


        # Convert IsRaiderRoom to boolean if the column exists
        # Handle potential numbers (like 1/0) or text ('Yes'/'No', 'TRUE'/'FALSE')
        if IS_RAIDER_ROOM_COLUMN in df.columns:
             # Map various possible inputs to True/False, treat others/NaN as None (-> False later)
             def map_bool(val):
                 if pd.isna(val): return None
                 val_str = str(val).lower().strip()
                 if val_str in ['true', 'yes', '1', '1.0']: return True
                 if val_str in ['false', 'no', '0', '0.0']: return False
                 return None # Unrecognized values become None

             df[IS_RAIDER_ROOM_COLUMN] = df[IS_RAIDER_ROOM_COLUMN].apply(map_bool)
        else:
             # If column doesn't exist, add it with None values
             df[IS_RAIDER_ROOM_COLUMN] = None

        # --- Data Processing ---
        processed_data = []
        for index, row in df.iterrows():
            # Check essential fields first
            building_name = row.get(BUILDING_COLUMN)
            # Ensure building name is a non-empty string
            if pd.isna(building_name) or not str(building_name).strip():
                logger.warning(f"Skipping row {index+2} due to missing or empty building name.")
                continue

            building_name = str(building_name).strip() # Clean it up

            lat = row.get(BUILDING_LAT_COLUMN)
            lon = row.get(BUILDING_LON_COLUMN)

            # IMPORTANT: Check for NaN coordinates after conversion
            # Pydantic model requires valid lat/lon, so skip rows where these are missing/invalid
            if pd.isna(lat) or pd.isna(lon):
                logger.warning(f"Row {index+2} (Building: {building_name}): Invalid or missing required coordinates (Lat: {lat}, Lon: {lon}). Skipping this entry.")
                continue

            # Process statuses using NEW column names
            final_wifi_status = process_status(row.get(WIFI_STATUS_COLUMN, ''), building_name, "WiFi", index)
            final_audio_status = process_status(row.get(AUDIO_STATUS_COLUMN, ''), building_name, "Audio", index)

            # Prepare data dictionary for Pydantic model, using NEW column names
            # Use row.get() to gracefully handle columns that *might* be missing from the file
            # The Pydantic model defines which are truly optional
            data_dict = {
                 # Identifiers (using new names)
                 BUILDING_NUMBER_COLUMN: row.get(BUILDING_NUMBER_COLUMN),
                 BUILDING_COLUMN: building_name, # Already validated non-empty

                 # Floor/Room (using new names)
                 FLOOR_NUMBER_COLUMN: row.get(FLOOR_NUMBER_COLUMN),
                 FLOOR_DESC_COLUMN: row.get(FLOOR_DESC_COLUMN),
                 ROOM_NUMBER_COLUMN: row.get(ROOM_NUMBER_COLUMN),
                 ROOM_ID_COLUMN: row.get(ROOM_ID_COLUMN), # Added

                 # Space Use (using new names)
                 SPACE_USE_CODE_COLUMN: row.get(SPACE_USE_CODE_COLUMN),
                 SPACE_USE_COLUMN: row.get(SPACE_USE_COLUMN),
                 IS_RAIDER_ROOM_COLUMN: row.get(IS_RAIDER_ROOM_COLUMN), # Already processed to bool/None

                 # Location (using new names)
                 CAMPUS_COLUMN: row.get(CAMPUS_COLUMN), # Added
                 ADDRESS_COLUMN: row.get(ADDRESS_COLUMN),
                 BUILDING_LOCATION_COLUMN: row.get(BUILDING_LOCATION_COLUMN), # Now text

                 # Coordinates (already validated BuildingLat/Lon are not NaN)
                 BUILDING_LAT_COLUMN: lat,
                 BUILDING_LON_COLUMN: lon,
                 # Pass Room Lat/Lon, will become None in model if NaN
                 ROOM_LAT_COLUMN: row.get(ROOM_LAT_COLUMN),
                 ROOM_LON_COLUMN: row.get(ROOM_LON_COLUMN),

                 # Status fields (already processed)
                 WIFI_STATUS_COLUMN: final_wifi_status,
                 AUDIO_STATUS_COLUMN: final_audio_status,

                 # Optional fields that might be entirely missing from new format
                 ENTITY_ID_COLUMN: row.get(ENTITY_ID_COLUMN),
                 BUILDING_LOCATION_ID_COLUMN: row.get(BUILDING_LOCATION_ID_COLUMN),
            }

            # Create BuildingStatus object, handling potential validation errors
            try:
                 building_entry = BuildingStatus(**data_dict)
                 processed_data.append(building_entry)
            except ValueError as pydantic_error: # Catch Pydantic's validation error
                 logger.error(f"Row {index+2} (Building: {building_name}): Validation Error creating Pydantic model. Check data types/values. Error: {pydantic_error}. Data: {data_dict}")
                 # Skip this problematic row and continue processing others

        logger.info(f"Successfully processed {len(processed_data)} building/room entries.")
        return processed_data

    except FileNotFoundError:
         logger.error(f"Caught FileNotFoundError for: {file_path}")
         raise HTTPException(status_code=500, detail=f"Server configuration error: Data file path invalid.")
    except KeyError as e:
         logger.error(f"Missing expected column during processing: {e}. Check column names in code vs file '{file_path}'.")
         raise HTTPException(status_code=500, detail=f"Data processing error: Missing column '{e}'.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred processing data file: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Unexpected error processing data file: {str(e)}")


# Load data at startup and store in app state
@app.on_event("startup")
async def startup_event():
    """Load data when the application starts."""
    try:
        app.state.building_data = load_and_process_data(DATA_FILE_PATH)
        logger.info(f"Loaded {len(app.state.building_data)} building records at startup.")
    except HTTPException as http_exc:
        # Log the error but allow app to start if data loading failed
        logger.error(f"HTTPException during startup data loading: {http_exc.detail}")
        app.state.building_data = [] # Initialize with empty list on failure
    except Exception as e:
        # Log the error but allow the app to start, maybe with empty data
        logger.exception("Failed to load data during startup due to an unexpected error.")
        app.state.building_data = [] # Initialize with empty list on failure


# --- API Endpoints ---

@app.get("/api/building-status",
         response_model=List[BuildingStatus],
         summary="Get Detailed Building & Room Status Information",
         description="Retrieves detailed information including WiFi status, Audio status, and geographic coordinates for all university buildings and rooms based on the loaded data file.")
async def get_all_building_status():
    """
    Endpoint to fetch all available building and room information including WiFi and Audio status.
    Data is loaded from the file at startup.
    """
    if not hasattr(app.state, 'building_data') or app.state.building_data is None:
         # This might happen if startup failed catastrophically before setting the state
         logger.error("Building data not found or not initialized in app state. Startup might have failed.")
         # Attempt to reload data, or return error immediately
         # For simplicity, returning error here. Could add a reload attempt.
         raise HTTPException(status_code=503, detail="Service unavailable: Data not loaded. Check server logs.")
    logger.info(f"Returning data for {len(app.state.building_data)} building/room entries.")
    return app.state.building_data



@app.get("/api/room-status/{room_id}",
         response_model=BuildingStatus,
         summary="Get Specific Room Status Information",
         description="Retrieves detailed information including WiFi status, Audio status, and geographic coordinates for a specific room based on the provided room ID.")
async def get_room_status(room_id: str):
    """
    Endpoint to fetch status information for a specific room using its ID.
    Returns detailed information including WiFi and Audio status, and geographic coordinates.
    """
    if not hasattr(app.state, 'building_data') or app.state.building_data is None:
         logger.error("Building data not found or not initialized in app state. Startup might have failed.")
         raise HTTPException(status_code=503, detail="Service unavailable: Data not loaded. Check server logs.")

    # Search for the room by ID
    for building in app.state.building_data:
        if building.room_id == room_id:
            logger.info(f"Found room with ID {room_id}. Returning data.")
            return building

    logger.warning(f"Room with ID {room_id} not found.")
    raise HTTPException(status_code=404, detail=f"Room with ID '{room_id}' not found.")


@app.get("/api/get-building-list",
         response_model=List[str],
         summary="Get List of All Building Names",
         description="Retrieves a list of all building names available in the loaded data file.")
async def get_building_list():
    """
    Endpoint to fetch a list of all building names from the loaded data.
    Returns a list of unique building names.
    """
    if not hasattr(app.state, 'building_data') or app.state.building_data is None:
         logger.error("Building data not found or not initialized in app state. Startup might have failed.")
         raise HTTPException(status_code=503, detail="Service unavailable: Data not loaded. Check server logs.")

    # Extract unique building names
    building_names = {building.building_name for building in app.state.building_data}
    logger.info(f"Returning list of {len(building_names)} unique buildings.")
    return list(building_names)

# Health check endpoint
@app.get('/api/health', response_model=dict, summary="Health Check")
async def health_check():
    """
    Provides a basic health check of the API.
    Indicates if the service is running and if data appears to be loaded.
    """
    data_loaded = hasattr(app.state, 'building_data') and isinstance(app.state.building_data, list)
    data_count = len(app.state.building_data) if data_loaded else 0
    status = "healthy" if data_loaded and data_count > 0 else "degraded"
    if data_loaded and data_count == 0:
        logger.warning("Health check: Service is running but no building data was loaded/found.")
        status = "degraded_no_data"
    elif not data_loaded:
        logger.warning("Health check: Service is running but data state attribute is missing.")
        status = "degraded_data_uninitialized"

    logger.info(f"Health check status: {status}, Data loaded: {data_loaded}, Record count: {data_count}")
    return {"status": status, "data_loaded": data_loaded, "record_count": data_count}

# Optional: Add main block for running with uvicorn if needed for simple local testing
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server for local testing...")
#     # Ensure DATA_FILE_PATH is correct relative to where you run this script, or use absolute path
#     # Example: Set environment variable or place data.xlsx next to the script
#     # os.environ["DATA_FILE_PATH"] = "data.xlsx"
#     uvicorn.run(app, host="0.0.0.0", port=8000)