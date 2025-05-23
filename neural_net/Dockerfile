FROM python:3.12
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools && apt-get update && apt-get install -y gcc python3-dev && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5001

RUN useradd app
USER app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5001"]
CMD python neural_net_manager.py