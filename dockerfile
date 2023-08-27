FROM python:3.9-alpine
# copy and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --user -r /requirements.txt

# install srai_telegram_frontend module
COPY srai_telegram_frontend /srai_telegram_frontend
COPY setup.py /setup.py
COPY setup.cfg /setup.cfg
COPY README.md /README.md
RUN pip install --user -e .

# contains config
COPY app /app
WORKDIR /app
CMD python main.py