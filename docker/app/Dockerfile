FROM tensorflow/tensorflow:0.12.0-py3

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

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

RUN useradd -m dev && \
    chsh -s /bin/zsh dev

USER dev
RUN git clone -b personalize git://github.com/teanoon/oh-my-zsh.git ~/.oh-my-zsh && \
    ln -s ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

WORKDIR /code
