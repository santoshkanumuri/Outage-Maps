# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Geospatial Dependencies if needed (keep these from previous steps)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git libgdal-dev gdal-bin libgeos-dev proj-bin libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code
COPY ./app.py /app/app.py
# Copy assets if you have them (e.g., for the static map image version)
# COPY ./assets /app/assets

# Expose the port (Heroku ignores this, uses $PORT)
# EXPOSE 8501

# --- Modify CMD to use $PORT ---
# Heroku provides $PORT. Streamlit needs --server.port $PORT
# Also disable certain features often handled by Heroku's routing/proxy
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false