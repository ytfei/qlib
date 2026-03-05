FROM python:3.13-slim

WORKDIR /qlib

COPY . .

RUN apt-get update && \
    apt-get install -y build-essential

# 安装 qlib 及其依赖 pyqlib
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYQLIB=v2026.02.07.dev01
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -e .
