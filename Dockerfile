FROM python:3.12

WORKDIR /app
ADD benchmark/ benchmark/
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

ENTRYPOINT [ "python", "-m", "benchmark.bench" ]
