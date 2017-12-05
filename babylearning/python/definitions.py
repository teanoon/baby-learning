import os

from os.path import join
from os.path import normpath

ROOT_DIR = os.getcwd()
CKPT_DIR = normpath(join(ROOT_DIR, '../checkpoints'))
RES_DIR = normpath(join(ROOT_DIR, '../resources'))

MNIST_DIR = join(CKPT_DIR, 'mnist')
