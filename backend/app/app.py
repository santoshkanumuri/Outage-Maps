import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Use Optional for statuses that might be missing/invalid
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional
import logging
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/data.xlsx")
BUILDING_COLUMN = "BuildingName"
LATITUDE_COLUMN = "Latitude"
LONGITUDE_COLUMN = "Longitude"
# --- NEW: Specific Status Columns ---
WIFI_STATUS_COLUMN = "WifiStatus"
AUDIO_STATUS_COLUMN = "AudioStatus"
# --- Values can be reused ---
ACTIVE_STATUS_VALUE = "Active"
INACTIVE_STATUS_VALUE = "Inactive"
# Define the valid status literals including a potential 'unknown' or 'inactive' default
StatusLiteral = Literal['active', 'inactive']

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class BuildingStatus(BaseModel):
    # Keep identifying fields
    building_name: str = Field(..., alias=BUILDING_COLUMN)
    latitude: float = Field(..., alias=LATITUDE_COLUMN)
    longitude: float = Field(..., alias=LONGITUDE_COLUMN)
    # --- NEW: Add separate status fields ---
    # Use Optional if data might be missing, or provide a default
    wifi_status: StatusLiteral = Field(..., alias=WIFI_STATUS_COLUMN)
    audio_status: StatusLiteral = Field(..., alias=AUDIO_STATUS_COLUMN)

    class Config:
        allow_population_by_field_name = True
        # Pydantic v2: use_enum_values = True (if using Enums)

    # Keep coordinate validators
    @validator('latitude')
    def latitude_must_be_valid(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def longitude_must_be_valid(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    # Optional: Add validators for status fields if needed,
    # but the Literal type provides good validation already.

# --- FastAPI App ---
app = FastAPI(
    title="University Building Systems Status API",
    description="Provides WiFi and Audio status and geographic coordinates for university buildings.",
    version="1.1.0" # Version bump
)

# --- CORS Middleware --- (Adjust origins for production)
origins = ["http://localhost", "http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Helper Function ---
def process_status(raw_status: str, building_name: str, status_type: str, index: int) -> StatusLiteral:
    """Processes a raw status string into 'active' or 'inactive'."""
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
        if file_path.lower().endswith(".xlsx"):
            # Explicitly set dtype to string for status columns to handle numbers/blanks better
            df = pd.read_excel(file_path, engine='openpyxl', dtype={WIFI_STATUS_COLUMN: str, AUDIO_STATUS_COLUMN: str})
        elif file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path, dtype={WIFI_STATUS_COLUMN: str, AUDIO_STATUS_COLUMN: str})
        else:
             raise HTTPException(status_code=500, detail="Unsupported data file format. Use .csv or .xlsx.")

        logger.info(f"Successfully loaded data. Shape: {df.shape}")

        # --- Data Validation ---
        # Update required columns
        required_columns = [BUILDING_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, WIFI_STATUS_COLUMN, AUDIO_STATUS_COLUMN]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}. Found: {df.columns.tolist()}"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # Convert types and handle potential errors robustly
        df[BUILDING_COLUMN] = df[BUILDING_COLUMN].astype(str)
        # Fill NaN in status columns with empty string BEFORE processing
        df[WIFI_STATUS_COLUMN] = df[WIFI_STATUS_COLUMN].fillna('')
        df[AUDIO_STATUS_COLUMN] = df[AUDIO_STATUS_COLUMN].fillna('')
        df[LATITUDE_COLUMN] = pd.to_numeric(df[LATITUDE_COLUMN], errors='coerce')
        df[LONGITUDE_COLUMN] = pd.to_numeric(df[LONGITUDE_COLUMN], errors='coerce')

        # --- Data Processing ---
        processed_data = []
        for index, row in df.iterrows():
            building_name = str(row[BUILDING_COLUMN]).strip()
            if not building_name:
                logger.warning(f"Skipping row {index+2} due to empty building name.")
                continue

            # Handle coordinates
            lat = row[LATITUDE_COLUMN]
            lon = row[LONGITUDE_COLUMN]
            if pd.isna(lat) or pd.isna(lon):
                logger.warning(f"Row {index+2} (Building: {building_name}): Invalid or missing coordinates (Lat: {lat}, Lon: {lon}). Skipping this building.")
                continue # Skip buildings without valid coordinates

            # --- NEW: Process both statuses ---
            final_wifi_status = process_status(row[WIFI_STATUS_COLUMN], building_name, "WiFi", index)
            final_audio_status = process_status(row[AUDIO_STATUS_COLUMN], building_name, "Audio", index)

            # Create BuildingStatus object using dictionary unpacking
            try:
                 data_dict = {
                     BUILDING_COLUMN: building_name,
                     LATITUDE_COLUMN: lat,
                     LONGITUDE_COLUMN: lon,
                     WIFI_STATUS_COLUMN: final_wifi_status, # Use constant as key
                     AUDIO_STATUS_COLUMN: final_audio_status # Use constant as key
                 }
                 building_entry = BuildingStatus(**data_dict)
                 processed_data.append(building_entry)
            except (ValueError, TypeError) as pydantic_error:
                 logger.error(f"Row {index+2} (Building: {building_name}): Validation Error creating Pydantic model from dict {data_dict}. Error: {pydantic_error}")
                 # Skip this problematic row

        logger.info(f"Processed {len(processed_data)} building entries with valid coordinates.")
        return processed_data

    except FileNotFoundError:
         logger.error(f"Caught FileNotFoundError for: {file_path}")
         raise HTTPException(status_code=500, detail=f"Server configuration error: Data file path invalid.")
    except Exception as e:
        logger.exception(f"Error processing data file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data file: {str(e)}")


# Load data at startup and store in app state
@app.on_event("startup")
async def startup_event():
    """Load data when the application starts."""
    try:
        app.state.building_data = load_and_process_data(DATA_FILE_PATH)
        logger.info(f"Loaded {len(app.state.building_data)} building records at startup.")
    except Exception as e:
        # Log the error but allow the app to start, maybe with empty data
        logger.exception("Failed to load data during startup.")
        app.state.building_data = [] # Initialize with empty list on failure


# --- API Endpoints ---

@app.get("/api/building-status",
         response_model=List[BuildingStatus],
         summary="Get Status (WiFi & Audio) & Geo Coords for Buildings",
         description="Retrieves WiFi status, Audio status, and geographic coordinates for all university buildings.")
async def get_all_building_status():
    """
    Endpoint to fetch combined WiFi and Audio status and geographic coordinates for all buildings.
    Data is loaded at startup.
    """
    if not hasattr(app.state, 'building_data'):
         # This might happen if startup failed catastrophically before setting the state
         logger.error("Building data not found in app state. Startup might have failed.")
         raise HTTPException(status_code=503, detail="Service unavailable: Data not loaded.")
    logger.info(f"Returning status for {len(app.state.building_data)} buildings.")
    return app.state.building_data

@app.get("/api/building-status/{building_name}",
         response_model=BuildingStatus, # Returns the full object with both statuses
         summary="Get Status (WiFi & Audio) for a Specific Building")
async def get_building_status_by_name(building_name: str):
    """Get the combined WiFi and Audio status for a specific building by name."""
    if not hasattr(app.state, 'building_data'):
         raise HTTPException(status_code=503, detail="Service unavailable: Data not loaded.")

    logger.info(f"Searching for building: {building_name}")
    search_name = building_name.strip().lower()
    for building in app.state.building_data:
        # Compare against the Pydantic model field name
        if building.building_name.lower() == search_name:
            logger.info(f"Found building: {building.building_name}")
            return building

    logger.warning(f"Building '{building_name}' not found.")
    raise HTTPException(status_code=404, detail=f"Building '{building_name}' not found")

# Remove or adapt old /active, /inactive endpoints as they are now ambiguous
# Keeping the health check endpoint
@app.get('/api/health', response_model=dict)
async def health_check():
    """Health check endpoint."""
    # Optionally add check for data loaded status
    data_loaded = hasattr(app.state, 'building_data') and isinstance(app.state.building_data, list)
    status = "healthy" if data_loaded else "degraded"
    logger.info(f"Health check: {status}")
    return {"status": status, "data_loaded": data_loaded}