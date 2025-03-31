import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Literal
import logging
from dotenv import load_dotenv


# --- Configuration ---
load_dotenv()
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/data.xlsx")
BUILDING_COLUMN = "BuildingName"
STATUS_COLUMN = "Status"
LATITUDE_COLUMN = "Latitude"
LONGITUDE_COLUMN = "Longitude"
ACTIVE_STATUS_VALUE = "Active"
INACTIVE_STATUS_VALUE = "Inactive"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class BuildingStatus(BaseModel):
    building_name: str = Field(..., alias=BUILDING_COLUMN)
    status: Literal['active', 'inactive'] = Field(..., alias=STATUS_COLUMN)
    latitude: float = Field(..., alias=LATITUDE_COLUMN)
    longitude: float = Field(..., alias=LONGITUDE_COLUMN)

    class Config:
        allow_population_by_field_name = True

    # Add validators for realistic Lat/Lon ranges
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

# --- FastAPI App ---
app = FastAPI(
    title = " University Building WI-Fi Status API",
    description = "This API provides the status of university buildings' Wi-Fi connectivity.",
    version = "1.0.0",
)

# CORS Middleware

origins = ['http://localhost', 'http://localhost:8501']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Load Data ---

def load_and_process_data(file_path:str) -> List[BuildingStatus]:
    "Loads the data from the specified file and processes it into a list of BuildingStatus objects."
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise HTTPException(status_code=500, detail="Data file not found")

    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.readcsv(file_path)
        else:
            raise HTTPException(status_code=500, detail="Unsupported file format")

        logger.info(f"Data loaded successfully, processing...")

        # Ensure the required columns are present
        required_columns = [BUILDING_COLUMN, STATUS_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}, found columns: {df.columns}")
            raise HTTPException(status_code=500, detail=f"Missing columns: {missing_columns}")

        # preprocess the data

        df[BUILDING_COLUMN] = df[BUILDING_COLUMN].astype(str)
        df[STATUS_COLUMN] = df[STATUS_COLUMN].str.strip().str.lower()
        df[LATITUDE_COLUMN] = pd.to_numeric(df[LATITUDE_COLUMN], errors='coerce')
        df[LONGITUDE_COLUMN] = pd.to_numeric(df[LONGITUDE_COLUMN], errors='coerce')

        # --- Data Processing ---
        processed_data = []
        for index, row in df.iterrows():
            building_name = str(row[BUILDING_COLUMN]).strip()
            if not building_name:
                logger.warning(f"Skipping row {index+2} due to empty building name.")
                continue

            # Handle status
            raw_status = str(row[STATUS_COLUMN]).strip().lower()
            final_status = 'inactive' # Default
            if raw_status == ACTIVE_STATUS_VALUE.lower():
                final_status = 'active'
            elif raw_status != INACTIVE_STATUS_VALUE.lower():
                 logger.warning(f"Row {index+2} (Building: {building_name}): Unexpected status '{row[STATUS_COLUMN]}'. Treating as inactive.")
            # Handle coordinates
            lat = row[LATITUDE_COLUMN]
            lon = row[LONGITUDE_COLUMN]
            if pd.isna(lat) or pd.isna(lon):
                logger.warning(f"Row {index+2} (Building: {building_name}): Invalid or missing coordinates (Lat: {lat}, Lon: {lon}). Skipping this building.")
                continue # Skip buildings without valid coordinates

            try:
                building_entry = BuildingStatus(
                BuildingName=building_name,
                Status=final_status,
                Latitude=lat,
                Longitude=lon
                )

                processed_data.append(building_entry)
            except ValueError as e:
                logger.error(f"Error creating BuildingStatus object for row {index+2}: {e}")
                continue

        logger.info(f"Processed {len(processed_data)} buildings successfully.")
        return processed_data
    except Exception as e:
        logger.error(f"Error loading or processing data: {e}")
        raise HTTPException(status_code=500, detail="Error loading or processing data")



# Load data at startup
@app.on_event("startup")
def startup_event():
    app.state.building_data = load_and_process_data(DATA_FILE_PATH)
    logger.info("Data loaded and processed successfully at startup.")

@app.get("/api/buildings-status", response_model=List[BuildingStatus])
def get_bulding_status():
    "Get the status of all buildings."
    logger.info("Fetching building status data.")
    return app.state.building_data

@app.get("/api/building-status/{building_name}", response_model=BuildingStatus)
def get_building_status_by_name(building_name:str):
    "Get the status of specific building by name."
    logger.info(f"Fetching status for building: {building_name}")
    building_name = building_name.strip()
    for building in app.state.building_data:
        if building.building_name.lower() == building_name.lower():
            return building
    logger.error(f"Building '{building_name}' not found.")
    raise HTTPException(status_code=404, detail=f"Building '{building_name}' not found")

@app.get("/api/active", response_model=List[BuildingStatus])
def get_active_buildings():
    "Get all active buildings."
    logger.info("Fetching active buildings.")
    active_buildings = [building for building in app.state.building_data if building.status == 'active']
    return active_buildings

@app.get("/api/inactive", response_model=List[BuildingStatus])
def get_inactive_buildings():
    "Get all inactive buildings."
    logger.info("Fetching inactive buildings.")
    inactive_buildings = [building for building in app.state.building_data if building.status == 'inactive']
    return inactive_buildings

@app.get('/api/health', response_model=dict)
def health_check():
    "Health check endpoint."
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}
