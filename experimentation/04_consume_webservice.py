import json
import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests


def normalize(arr):
    # arr = arr.astype('float32')
    for channel in range(3):
        maxval = arr[:, :, channel].max()
        minval = arr[:, :, channel].min()
        if maxval != minval:
            arr[:, :, channel] -= minval
            arr[:, :, channel] *= int(255.0 / (maxval - minval))
    return arr


def process(img, hold=0):
    '''
    img is the path of image file
    hold!=0 is for attaining no normalized array
    '''
    ori_img = np.array(plt.imread(img))
    h = ori_img.shape[0]
    w = ori_img.shape[1]
    # pad
    if w > h:
        new_image = np.pad(ori_img, ((w - h, 0), (0, 0), (0, 0)),
                           'constant', constant_values=255)
    elif w < h:
        new_image = np.pad(ori_img, ((0, 0), (h - w, 0), (0, 0)),
                           'constant', constant_values=255)
    else:
        new_image = ori_img
    # resize
    # print(type(new_image))
    # new_image = Image.fromarray(new_image)
    new_image = Image.fromarray((new_image * 255).astype(np.uint8))
    new_image = new_image.convert('RGB').resize((128, 128), Image.ANTIALIAS)
    new_image = np.array(new_image)
    if hold == 0:
        # contrast stretching
        new_image = normalize(new_image)
        return new_image
    else:
        return new_image


if __name__ == "__main__":
    # Test scoring
    # init()

    oneclass = []
    processed1 = process(
        'data\\gear_images\\boots\\41edw+BCUjL._AC_US436_QL65_.jpg')
    processed2 = process(
        'data\\gear_images\\gloves\\41bBWex0b7L._AC_US320_QL65_.jpg')
    processed3 = process(
        'data\\gear_images\\insulated_jackets\\103454.jpeg')
    oneclass.append(processed1)
    oneclass.append(processed2)
    oneclass.append(processed3)

    # prediction = run(np.array(oneclass), {})

    arr = np.array(oneclass)
    js_data = json.dumps({"data": arr.tolist()})
    # test = bytes(test, encoding='utf8')
    # y_hat = service.run(input_data=test)

    # prediction = run(js_data, {})

    # print("Test result: ", prediction)

    print(arr.shape)

    # headers = {'Content-Type': 'application/json'}

    # for AKS deployment you'd need to the service key in the header as well
    # api_key = service.get_key()
    headers = {'Content-Type': 'application/json',
               'Authorization':
               ('Bearer ' + '3shRck4Fhh4B1HuZLBBKhQQWcplFX4IB')}
    # js_data = json.dumps({"data": np.ones([2, 128, 128, 3]).tolist()})
    scoring_uri = "http://9ba5476a-c807-49b9-a2f5-3e56ac92e63e.\
centralus.azurecontainer.io/score"
    scoring_uri = 'http://168.61.146.240:80/api/v1/service/mlops-aks/score'
    resp = requests.post(scoring_uri, js_data, headers=headers)

    print("POST to url", scoring_uri)
    # print("input data:", input_data)
    # print("label:", y_test[random_index])
    # print("prediction:", resp.json())
    print("prediction:", resp.text)

    # print(json.loads(js_data))
