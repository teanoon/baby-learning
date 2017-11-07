FROM babylearning_app:latest

ARG proxy
ENV http_proxy=$proxy
ENV https_proxy=$proxy
USER root
RUN echo "deb http://packages.linuxmint.com debian import" | tee /etc/apt/sources.list.d/firefox.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    # common dev tools
    vim htop git openssh-client zsh tmux wget curl w3m less \
    # common x11 client dependencies
    firefox \
    # better fonts
    ttf-wqy-microhei \
    # certificate
    ca-certificates \
    # some ides need machine-id info like atom/eclipse
    dbus libxtst6 && dbus-uuidgen > /etc/machine-id && \
    rm -rf /var/lib/apt/lists/*

USER dev
