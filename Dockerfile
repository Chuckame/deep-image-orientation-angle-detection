FROM python:3.11.11

RUN pip install --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_VISIBLE_DEVICES=-1

CMD [ "python", "./infer-rotation.py", "--image-path", "./inputs/download.jpeg" ]
