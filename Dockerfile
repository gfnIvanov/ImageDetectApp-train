FROM python:3.10.13-slim
WORKDIR /trainapp
COPY pyproject.toml pyproject.toml
COPY .env.public .env.public
COPY .env.secret .env.secret
ADD app /trainapp/app
ADD logs /trainapp/logs
ADD model /trainapp/model
ADD params /trainapp/params
COPY poetry.lock poetry.lock
RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN poetry install
EXPOSE 5202
CMD poetry run start