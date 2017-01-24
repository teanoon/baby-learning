import io
import json
import os
import unittest

import main

NUMBER_EIGHT_PIC = os.path.join(os.path.dirname(__file__), '../resources/8.jpg')


class MainTest(unittest.TestCase):
    def setUp(self):
        main.app.config['TESTING'] = True
        self.app = main.app.test_client()

    def test_hello_world(self):
        response = self.app.get('/')
        print(response.data)
        assert b'world' in response.data

    def test_recognize(self):
        response = self.app.post(
            '/recognize',
            content_type='multipart/form-data',
            data={'file': io.FileIO(NUMBER_EIGHT_PIC)})
        data = json.loads(response.get_data(as_text=True))

        assert 8 == int(data['hit'])

if __name__ == '__main__':
    unittest.main()
