FROM python:3.9-alpine

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirement.txt

COPY . .

CMD ["python3", "-m", "/code/src/network/app.py"]