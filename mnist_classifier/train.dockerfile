# Base Image
FROM python:3.10-slim

# Install and run
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# To use my local cache -> Need to mount it.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# The previous steps are very common.

# Copy the requirements
# COPY <from> <to> -> Trying to keep docker as small as possible

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Setting up the working directory. --no-cache-dir is important to keep the image size small.
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Entry point
ENTRYPOINT ["python", "-u", "src/final_exercise/model.py"]
# -u redirects stdout and stderr to the console, which is useful for debugging.
