FROM python:3.10-slim

ADD . /tmp/repo
RUN cd /tmp/repo && pip install --no-cache-dir ".[all]"

ENTRYPOINT [ "zod" ]