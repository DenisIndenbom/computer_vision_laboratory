FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

WORKDIR /cvlab

ENV PYTHONUNBUFFERED=1

COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

COPY . .

CMD ["python", "train.py", "--help"]