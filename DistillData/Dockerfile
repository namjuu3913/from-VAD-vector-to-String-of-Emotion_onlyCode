FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 (deadsnakes PPA)
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    # Python 3.11, venv, dev
    apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && \

    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 3.11 pip 
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11


WORKDIR /app

COPY requirements.txt .

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["python3.11", "distillation.py"]