FROM python:3.9

RUN apt-get update && apt-get install -y python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app/

COPY requirements.txt /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]