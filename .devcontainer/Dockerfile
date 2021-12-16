FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

RUN pip install tensorflow-hub tensorflow-datasets

RUN adduser --gecos "" --disabled-password dev && \
    mkdir -p /home/dev/.vscode-server/extensions /home/dev/.keras /home/dev/.local/lib/python3.8/site-packages && \
    chown -R dev /home/dev/.vscode-server /home/dev/.keras /home/dev/.local/lib/python3.8/site-packages
USER dev
