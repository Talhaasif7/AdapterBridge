# Multi-stage lightweight Docker container for AdapterBridge
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# Final minimal runner image
FROM python:3.11-slim AS runner

LABEL org.opencontainers.image.title="AdapterBridge" \
      org.opencontainers.image.description="LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference Runtimes" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"

# Non-root user for Kubernetes security compliance
RUN useradd -m -u 10001 adapterbridge

WORKDIR /workspace

COPY --from=builder /install /usr/local

USER adapterbridge

ENTRYPOINT ["adapterbridge"]
CMD ["--help"]
