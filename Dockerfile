# RunPod Serverless GPU Worker for Farmium
# Using smaller base image (no PyTorch needed for our tools)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcairo2-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Copy handler and tools
COPY handler.py /handler.py
COPY tools/ /workspace/tools/

# Copy logo assets for VideoReframe
COPY assets/ /workspace/assets/

# Copy TikTok fonts for Captioner
COPY fonts/ /app/fonts/

# Set environment variables
ENV WORKSPACE=/workspace
ENV PYTHONUNBUFFERED=1

# FFmpeg settings for NVIDIA GPU encoding
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Run handler
CMD ["python", "-u", "/handler.py"]
