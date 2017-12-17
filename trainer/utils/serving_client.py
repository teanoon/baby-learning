from grpc.beta import implementations
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis.predict_pb2 import PredictRequest


class ServingClient(object):
    def __init__(self, host, port):
        # TODO use a secure channel
        channel = implementations.insecure_channel(host, int(port))
        self._stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def request(self, name, signature_name, data_sets):
        request = PredictRequest()
        request.model_spec.name = name
        request.model_spec.signature_name = signature_name
        # TODO for loop
        image, label = data_sets[0]
        proto = tensor_util.make_tensor_proto(image, shape=[1, image.size])
        request.inputs['x_input'].CopyFrom(proto)
        return self._stub.Predict(request, 50000.0)
