FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY assets ./assets
COPY apps ./apps
COPY scripts ./scripts
RUN pip install --no-cache-dir -e ".[dashboard]"
ENV MUJOCO_GL=disabled
EXPOSE 8000
CMD ["uvicorn", "apps.dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
