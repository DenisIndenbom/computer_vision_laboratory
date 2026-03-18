FROM python:3.10-slim

RUN pip install --no-cache-dir setuptools wheel
RUN pip install --no-cache-dir tensorboard

WORKDIR /app
EXPOSE 6006

CMD ["tensorboard", "--logdir=/logs", "--host=0.0.0.0"]