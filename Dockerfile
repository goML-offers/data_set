FROM python:3.8

WORKDIR /app/

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

#RUN pip install transformers diffusers accelerate

#RUN pip install xformers

RUN apt-get update && apt-get install -y poppler-utils

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
