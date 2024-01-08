# Микросервис для обучения модели

[Бэкенд](https://github.com/gfnIvanov/ImageDetectApp-serv)

[Фронтенд](https://github.com/gfnIvanov/ImageDetectApp-web/tree/master)

## Обзор

На стороне сервиса выполняется высоконагруженная операция обучения модели. Данные для обучения передаются с бэкенда в виде байтов, которые десериализуются в тензор. Промежуточные результаты обучения и параметры оптимизатора сохраняются в двоичные файлы на каждой итерации (на каждой последующей итерации загружаются в объекты модели и оптимизатора). Обученная модель сохраняется в s3-хранилище Yandex Object Storage.

Для тренировки модели использован фреймворк машинного обучения [Pytorch](https://pytorch.org/).

Развертывание приложения (фронт, бэк и сервис обучения) производится при помощи docker-compose (файл на стороне сервера).

## Технологии

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
