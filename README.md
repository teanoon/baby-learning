#Baby is learning

## Setup
1. Install docker && docker-compose
2. `> ./docker/deploy.sh`
3. Then you can access http://0.0.0.0:5000 
4. Or POST /recognize with 'multipart/form-data' and 'file' field to get hand writing recognition.

## Theories

## Steps:
0. ~~Mnist sample~~
1. Learn more modeling.
    * ~~Graphic recognition~~
    * ~~Data mining~~
2. Learn complex algorithm.
    * ~~improve recognition success rate of this "hello world" example from 91.% to 99.9% up.~~
    * Generic data input(may have a perfect solution in TensorflowServing)
    * Dropout
    * Epochs
    * Data augmentation
3. Start a web server to communicate with the baby.
    * ~~flask api backend~~
    * Serving with Tensorflow Serving
    * Wechat app as the client
4. Enable GPU computing.
5. ~~Log~~, ~~monitor~~, ~~visualize~~, Log analytics, embedding and distribute.
