#Dockerfile, Image, Container
FROM python:3.14.3

WORKDIR /Src

RUN pip install torch
RUN pip install torchaudio
RUN pip install torchcodec
RUN pip install numpy
RUN pip install matplotlib
RUN pip install scikit-learn

COPY . .

CMD ["python", ".main.py"]
