
# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

IMAGE_SIZE = (224, 224, 3)
CATEGORIES = ['Autistic', 'Normal']

model = load_model("Model/AutismDetection_resnet_50_model.h5")

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

    print("Final Output:", CATEGORIES[np.bincount(np.array(op)).argmax()])
    return  CATEGORIES[np.bincount(np.array(op)).argmax()]

def model_prediction(image,input):

    dl_output = detect_Autism(image)

    data = pd.read_csv("files/final-data.csv")
    X = data.drop(columns=["Qchat-10-Score"])
    y = data["Qchat-10-Score"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create a traditional machine learning model
    rf_model = DecisionTreeClassifier()
    rf_model.fit(X_train, y_train)
    
    prediction = rf_model.predict([input])
    
    Output = ""
    Therapy = ""

    if dl_output == "Autistic":
        if prediction >= 7:
            Output = "Level 3 - Requiring Very Substantial Support"
            Therapy = "Intensive interventions, Individualized education plans, and Support for daily living skills."
        elif 3 < prediction < 7:
            Output = "Level 2 - Requiring Substantial Support"
            Therapy = "Applied behavior analysis (ABA), Speech therapy, and Specialized education programs."
        elif prediction == 3:
            Output = "Level 1 - Requiring Support"
            Therapy = "Social skills training, Speech therapy, and Occupational therapy."
    else:
        Output = "Normal Child"
        Therapy = "NO ASD Traits"


    return dl_output,Output,Therapy

if __name__ =="__main__":

    print(model_prediction("files/test data/0065.jpg",(0,1,1,1,0,0,0,0,1,0)))