FROM tensorflow/tensorflow:1.0.0-alpha-py3

RUN apt-get update && apt-get install -y --force-yes --no-install-recommends \
    # common dev tools
    vim htop git openssh-client zsh tmux wget curl w3m less && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas Flask pillow hyperopt

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

RUN useradd -m dev && \
    chsh -s /bin/zsh dev

USER dev
RUN git clone -b personalize git://github.com/teanoon/oh-my-zsh.git ~/.oh-my-zsh && \
    ln -s ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

WORKDIR /code
