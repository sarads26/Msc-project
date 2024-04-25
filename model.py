from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMAGE_SIZE = (224, 224, 3)
CATEGORIES = ['Autistic', 'Normal']

model = load_model("Model\AutismDetection_resnet_101_v2_model.h5")



def detect_Autism(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_flip_ver = cv2.flip(img, 0)
    img_flip_hor = cv2.flip(img, 1)

    images = []
    images.append(img)
    images.append(img_rotated_90)
    images.append(img_rotated_180)
    images.append(img_rotated_270)
    images.append(img_flip_ver)
    images.append(img_flip_hor)

    images = np.array(images)
    images = images.astype(np.float32)
    images /= 255

    op = []
    # make predictions on the input image
    for im in images:
        image = np.array(im)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred = pred.argmax(axis=1)[0]
        op.append(pred)
        # print("Pred:", pred, CATEGORIES[pred])

    op = np.array(op)
    print("Final Output:", CATEGORIES[np.bincount(op).argmax()])
    return CATEGORIES[np.bincount(op).argmax()]



print(detect_Autism(r"Autism dataset\test data\0026.jpg"))