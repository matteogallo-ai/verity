# syntax=docker/dockerfile:1
# Build stage: resolve the locked dependencies into a virtualenv, install Verity as a
# wheel (prompts/ and migrations/ ride inside it, see verity.resources) and download
# the embedding and reranker models, so the runtime image needs no network.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf

WORKDIR /app
# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY prompts ./prompts
COPY migrations ./migrations
RUN uv sync --frozen --no-dev --no-editable

# The model names come from Settings, so the image holds the models the app loads.
RUN /app/.venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from verity.config import get_settings; \
s = get_settings(); \
SentenceTransformer(s.embedding_model); CrossEncoder(s.reranker_model)"

# Runtime stage: the virtualenv, the models and the demo corpus; no compiler, no uv,
# no source tree. Runs as an unprivileged user.
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin verity
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/hf /opt/hf
COPY datasets/corpus ./datasets/corpus

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1

USER verity
EXPOSE 8000
ENTRYPOINT ["verity"]
# 0.0.0.0 inside the container only; docker-compose.yml publishes the port on the
# host's loopback interface.
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
