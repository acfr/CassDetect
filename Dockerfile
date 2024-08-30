FROM ultralytics/ultralytics:latest

RUN pip install transformers accelerate

WORKDIR /app
