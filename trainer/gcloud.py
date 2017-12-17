import os

import tensorflow as tf
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from tensorflow.examples.tutorials.mnist import input_data

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/code/.config/gcloud/application_default_credentials.json"


def main(_):
    data_sets = input_data.read_data_sets('/code/data', one_hot=True)
    test_data = data_sets.test.images[0]

    project_id = 'delta-suprstate-168507'
    model_name = 'mnist'
    version = 'v2'

    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build('ml', 'v1', credentials=credentials,)
    name = 'projects/{}/models/{}/versions/{}'.format(project_id, model_name, version)
    response = ml.projects().predict(
        name=name,
        body={'instances': [{
            'x_input': test_data.tolist()
        }]}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    print(response['predictions'])


if __name__ == '__main__':
    tf.app.run()
