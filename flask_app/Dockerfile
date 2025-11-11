FROM python:3.11.9-slim
WORKDIR /app
COPY ../src /app/src
COPY app.py /app
COPY ../requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]