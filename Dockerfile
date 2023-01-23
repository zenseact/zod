FROM python:3.10-slim

ADD . /zod
RUN cd /zod && pip install --no-cache-dir ".[all]"

ENTRYPOINT [ "zod" ]