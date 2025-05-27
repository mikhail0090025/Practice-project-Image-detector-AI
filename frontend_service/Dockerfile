FROM python:3.9-slim
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install tree

COPY frontend_service ./
EXPOSE 5002

RUN useradd app
USER app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
CMD tree && uvicorn app.main:app --host 0.0.0.0 --port 5002