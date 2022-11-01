import numpy as np
import tensorflow as tf
import cv2
import sys


def testModel(modelpath, modelNo, imagepath):
    print(modelpath)
    model = tf.keras.models.load_model(modelpath)
    img = cv2.imread(imagepath)
    w = img.shape[0]
    h = img.shape[1]
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(imgLAB)
    L = L / 255
    L = L.reshape(1, 1, 128, 128)

    prediction = model.predict(L)
    a = prediction[0][0]
    b = prediction[0][1]
    L = L.reshape(128, 128)
    L = np.array(L * 255, dtype=np.uint8)
    if(modelNo == 0):
        a = np.array(a * 255, dtype=np.uint8)
        b = np.array(b * 255, dtype=np.uint8)
    elif(modelNo == 1):
        a = np.array(a * 51, dtype=np.uint8)
        b = np.array(b * 51, dtype=np.uint8)
        a = a + 144
        b = b + 144
    elif(modelNo == 2):
        a = np.array(a * 127, dtype=np.uint8)
        b = np.array(b * 127, dtype=np.uint8)
        a = a + 128
        b = b + 128
    print(b.shape, L.shape)
    imgLAB = cv2.merge([L, a, b])
    image = cv2.cvtColor(imgLAB, cv2.COLOR_LAB2BGR)
    image = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)
    cv2.imwrite('result.jpg', image)


testModel(sys.argv[1], int(sys.argv[3]), sys.argv[2])
