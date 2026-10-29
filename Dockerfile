FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INTERVIEWTRACK_DATA_DIR=/app/data \
    HF_HOME=/app/model-cache \
    EMBEDDINGS_ENABLED=false \
    LLM_REVIEW_ENABLED=false
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY requirements-core.txt requirements.txt ./
RUN python -m pip install -r requirements-core.txt
COPY --chown=app:app app.py pytest.ini .env.example ./
COPY --chown=app:app interviewtrack/ ./interviewtrack/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app demo/fictional-inbox.db ./demo/fictional-inbox.db
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml
RUN mkdir -p /app/data /app/model-cache && chown -R app:app /app/data /app/model-cache
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

FROM runtime AS neural
USER root
RUN python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt
USER app
ENV EMBEDDINGS_ENABLED=true

# Default Docker build is the lightweight reviewer demo.
FROM runtime AS demo
