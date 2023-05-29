#FROM python:3.10.11-slim
#FROM python:3.10.11-slim-bullseye
FROM python:3.10.11-bullseye

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

#RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bundle app source
COPY ./src/static/*.css ./src/static/
COPY ./src/templates/* ./src/templates/
COPY ./src/*.py ./src/
COPY ./chroma_db/ ./chroma_db/


EXPOSE 5000
CMD [ "python", "./src/server.py"]