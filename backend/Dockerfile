# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# --- Tell uvicorn the path to the data file INSIDE the container ---
# Make sure this matches the path used in your main.py (e.g., /code/data/data.xlsx)
ENV DATA_FILE_PATH=/code/data/data.xlsx

# Install dependencies (no changes needed here usually)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy data file INTO the image ---
COPY ./data /code/data

# Copy application code
COPY ./app /code/app

# Expose the port (Heroku uses $PORT, but EXPOSE is good practice)
# EXPOSE 8000 # Heroku ignores this, uses $PORT

# --- Modify CMD to use $PORT ---
# Heroku provides the $PORT env var. Uvicorn needs --port $PORT
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "$PORT"]