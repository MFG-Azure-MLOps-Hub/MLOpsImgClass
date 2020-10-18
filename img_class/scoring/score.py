import os
import json
import numpy as np
from azureml.core.model import Model
from keras.models import load_model


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])

    model = load_model(model_path)


def run(raw_data, request_headers):
    data = np.array(json.loads(raw_data)['data'])
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
    test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
    prediction = run(test_row, {})
    print("Test result: ", prediction)
