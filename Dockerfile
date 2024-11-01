FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt --upgrade

EXPOSE 80

ENV NAME World

CMD ["python", "main.py"]
