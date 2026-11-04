FROM python:3.12.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.9.7
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" LAB_DATA=/app/data
CMD ["streamlit", "run", "src/inference_lab/dashboard.py", "--server.address=0.0.0.0", "--server.headless=true"]
