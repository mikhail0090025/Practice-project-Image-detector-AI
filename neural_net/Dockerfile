FROM python:3.9-slim
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install tree
RUN tree

COPY /neural_net .
EXPOSE 5001

# RUN useradd app
USER root

# CMD tree && sleep 5 && python neural_net_manager.py
CMD tree && sleep 5 && uvicorn endpoints:app --host 0.0.0.0 --port 5001