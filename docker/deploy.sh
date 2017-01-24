# git clone/pull
# copy initial secret files or mount these files instead?
docker-compose run app bash -c "\
  if [ ! -f ~/.ssh/known_hosts ];
  then
    mkdir ~/.ssh/
    touch ~/.ssh/known_hosts
  fi
  if ! grep -q 'github.com' ~/.ssh/known_hosts;
  then
    ssh-keyscan github.com >> ~/.ssh/known_hosts
  fi

  if [ ! -f .gitignore ];
  then
    echo 'repo initializing'
    git clone git@github.com:teanoon/baby-learning.git
  else
    echo 'repo updating'
    git checkout master
    git pull
  fi
"
docker-compose run app python baby-learning/server/api.py
