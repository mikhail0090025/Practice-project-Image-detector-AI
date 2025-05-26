FROM python:3.9-slim
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install tree
RUN tree
COPY dataset_manager ./
COPY datasets ./datasets/
EXPOSE 5000

# RUN useradd app
USER root

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
CMD tree -d && uvicorn app.main:app --host 0.0.0.0 --port 5000