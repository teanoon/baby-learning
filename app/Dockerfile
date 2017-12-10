FROM tensorflow/tensorflow:1.4.0-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    # common dev tools
    vim htop git openssh-client zsh tmux wget curl w3m less && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    pandas Flask pillow hyperopt pyyaml \
    # fix hyperopt/hyperopt/issues/325
    networkx==1.11

RUN useradd -m dev && \
    chsh -s /bin/zsh dev && \
    mkdir /project && chown dev:dev /project

USER dev
RUN git clone -b personalize git://github.com/teanoon/oh-my-zsh.git ~/.oh-my-zsh && \
    ln -s ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc && \
    ln -s ~/.oh-my-zsh/templates/tmux.conf.template.2.1 ~/.tmux.conf

WORKDIR /project
