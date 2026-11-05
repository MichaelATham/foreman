# usd-brep ships Linux wheels (x86_64 + aarch64) for CPython 3.12, so this
# image runs natively on both Intel and Apple-silicon hosts.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY foreman ./foreman
COPY examples ./examples
COPY tests ./tests
RUN pip install --no-cache-dir . && pip check
ENV PYTHONUNBUFFERED=1 PORT=8000
EXPOSE 8000
CMD ["foreman-server"]
