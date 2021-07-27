FROM python:3.9-slim

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:./src"
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5004
ENV FLASK_ENV=development
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./src /app

EXPOSE 5004

CMD ["flask","run"]
# CMD python ./main.py