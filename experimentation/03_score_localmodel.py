import os
import glob
import json
import numpy
from keras.models import load_model
import numpy as np
from utils import load_data


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = 'experimentation/mnist.h5'
    model = load_model(model_path)
    # print(model.summary())


def run(raw_data, request_headers):
    data = np.array(json.loads(raw_data)['data'])
    # result = model.predict_classes(data)

    # pred_classes = np.argmax(result)
    result = np.argmax(model.predict(data), axis=1)

    # result = model.predict_classes(raw_data)

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}


if __name__ == "__main__":
    # Test scoring
    init()

    n = 2
    data_folder = "data"
    X_test_path = glob.glob(
        os.path.join(
            data_folder,
            '**/t10k-images-idx3-ubyte.gz'),
        recursive=True)[0]
    X_test = load_data(X_test_path, False) / 255.0
    sample_indices = np.random.permutation(X_test.shape[0])[0:n]

    print(X_test[sample_indices].shape)
    js_data = json.dumps({"data": X_test[sample_indices].tolist()})
    # js_data = json.dumps({"data": np.ones([2, 784]).tolist()})

    prediction = run(js_data, {})

    print("Test result: ", prediction)
