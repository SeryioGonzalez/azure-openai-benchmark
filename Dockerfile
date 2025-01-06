FROM python:3.12

WORKDIR /app
ADD prompts.json .
ADD apify.py .
ADD benchmark/ benchmark/
ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore


EXPOSE 8000 8001


#ENTRYPOINT [ "python", "-m", "benchmark.bench" ]

ENTRYPOINT [ "python", "apify.py" ]
