# Baby is learning

## Setup
1. Install docker && docker-compose
2. `> docker-compose up`
3. POST normal images to http://0.0.0.0:5000/recognize with 'multipart/form-data' and 'file' field to get hand writing recognition.

## Theories
1. [Neural network and deep learning](http://neuralnetworksanddeeplearning.com/)

## Objectives:
0. ~~Mnist sample~~
1. Learn more modeling:
    * ~~CNN~~
    * RNN
2. Learn complex algorithm:
    * Tweak hyper-parameters(hyperopt automation)
    * ~~tf.Data~~
    * ~~tf.keras~~
    * Weight initialization(L2 regularization)
    * ~~Dropout~~
    * ~~Epochs and learning rate decay~~
    * ~~Data augmentation~~
3. Start a web server to communicate with the baby:
    * Flask api backend
    * Serving with Tensorflow Serving
    * Wechat app as the client
4. Production-ready:
    * Log analytics(learning rate, accuracy per epoch)
    * Tensorboard
    * Visualize
    * Embedding
    * GPU computing
    * XLA
    * Google cloud ML engine
