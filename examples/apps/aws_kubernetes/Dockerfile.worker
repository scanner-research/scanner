FROM scannerresearch/scanner:cpu-latest
WORKDIR /app

COPY worker.py .

ENV LD_LIBRARY_PATH /usr/local/lib/python3.5/dist-packages/scannerpy:$LD_LIBRARY_PATH
CMD python3 worker.py
