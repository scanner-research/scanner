# Scanner base CPU image for Ubuntu 16.04

ARG base_tag
FROM ${base_tag}
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1

RUN bash ./deps.sh --install-all --prefix /usr/local && \
    rm -rf /opt/scanner-base

ENV PYTHONPATH /usr/local/python:${PYTHONPATH}
ENV PYTHONPATH /usr/local/lib/python3.5/site-packages:${PYTHONPATH}
ENV PYTHONPATH /usr/local/lib/python3.5/dist-packages:${PYTHONPATH}

WORKDIR /
