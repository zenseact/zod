FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

# Copy over the bare minimum to install the package
COPY pyproject.toml README.md /zod/
COPY zod/ /zod/zod

RUN cd /zod && pip install --no-cache-dir ".[all]"

ENTRYPOINT "zod"
