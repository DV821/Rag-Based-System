FROM python:3.11-slim
WORKDIR /app
COPY ../src /app/src
COPY streamlit_ui.py /app
COPY ../requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]