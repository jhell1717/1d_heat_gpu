# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy package metadata first for caching
COPY pyproject.toml README.md ./

# Install build tools and your package
RUN pip install --upgrade pip setuptools wheel

# Copy the actual package source
COPY heat2d ./heat2d

# Install the package
RUN pip install .

# Set the entrypoint to your CLI
ENTRYPOINT ["heat2d"]
