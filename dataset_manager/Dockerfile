FROM python:3.9-slim
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install tree
RUN tree
COPY . .
EXPOSE 5000

USER root

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
CMD uvicorn dataset_manager.app.main:app --host 0.0.0.0 --port 5000