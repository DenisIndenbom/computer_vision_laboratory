FROM python:3.10-slim

RUN pip install --no-cache-dir setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir tensorboard -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /tensorboard
EXPOSE 6006

CMD ["tensorboard", "--logdir=/tensorboard/logs", "--host=0.0.0.0"]