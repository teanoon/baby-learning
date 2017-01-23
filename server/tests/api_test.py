import json
import unittest

import io

import numpy

from server import api


class ApiTest(unittest.TestCase):
    def setUp(self):
        api.app.config['TESTING'] = True
        self.app = api.app.test_client()

    def test_hello_world(self):
        response = self.app.get('/')
        print(response.data)
        assert b'world' in response.data

    def test_recognize(self):
        response = self.app.post(
            '/recognize',
            content_type='multipart/form-data',
            data={'file': io.FileIO('resources/8.jpg')})
        data = json.loads(response.get_data(as_text=True))

        assert 8 == int(data['hit'])

if __name__ == '__main__':
    unittest.main()
