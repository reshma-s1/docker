FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/chatai

ENTRYPOINT ["streamlit", "run", "app.py"]