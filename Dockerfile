FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch wheels, then the project's remaining dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py streamlit_app.py ./
COPY model ./model

EXPOSE 8000 8501

# Default service is FastAPI. Override the command to run Streamlit instead.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
