FROM ubuntu:latest

RUN apt update && apt upgrade -y

RUN apt install -y -q build-essential python3-pip python3-dev

COPY . /usr/app

WORKDIR /usr/app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["app.py"]