FROM tensorflow/tensorflow:2.7.0-jupyter

RUN adduser dev && \
    mkdir -p /home/dev/.vscode-server/extensions && \
    chown -R dev /home/dev/.vscode-server
USER dev
