FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /cvlab

RUN apt update && apt install -y python3-venv

RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER=true

COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

COPY . .

CMD ["python", "train.py", "--help"]