FROM tensorflow_app:latest

USER root
RUN echo "deb http://packages.linuxmint.com debian import" | tee /etc/apt/sources.list.d/firefox.list && \
    apt-get update && apt-get install -y --force-yes --no-install-recommends \
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
    apt-get clean

USER dev
