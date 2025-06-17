FROM python:3.9-slim

EXPOSE $PORT
CMD exec uvicorn my_application:app --port $PORT --workers 1 main:app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
