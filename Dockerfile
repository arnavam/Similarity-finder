FROM python:3.13-slim

COPY requirements.txt .

RUN apt-get update && apt-get install -y git

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "a_streamlit_app.py",  "--server.port=8501", "--server.address=0.0.0.0"]


