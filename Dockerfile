#Dockerfile, Image, Container
FROM python:3.11-bullseye

WORKDIR /Speech_Classification_CNN

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libsndfile1 \
    libsm6 \
    libxext6

RUN pip install --upgrade pip

RUN pip install torch torchaudio torchcodec numpy matplotlib scikit-learn \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python", "main.py"]