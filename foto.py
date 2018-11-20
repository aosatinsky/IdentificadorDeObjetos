import time
import cv2
import io
import os

apipath = os.path.abspath(r"PATH TO API KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = apipath

from google.cloud import vision
from google.cloud.vision import types

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # Escape para salir
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        #Espacio para sacar la foto
        img_name = "Foto_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} Se escribio!".format(img_name))
        img_counter += 1
        time.sleep(2)
        client = vision.ImageAnnotatorClient()
        file_name = os.path.join(
        os.path.dirname(__file__),
        img_name)
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)
        response = client.web_detection(image=image)
        annotations  = response.web_detection
        if annotations.web_entities:
            print('\n{} Web entities found: '.format(
                        len(annotations.web_entities)))
            for entity in annotations.web_entities:
                print('\n\tScore      : {}'.format(entity.score))
                print(u'\tDescription: {}'.format(entity.description))
                if format(entity.description).lower().find("crime") == 0 or format(entity.description).lower().find("robbery") == 0 or format(entity.description).lower().find("murder") == 0 or format(entity.description).lower().find("criminal") == 0:
                    print("Hello World")
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        print('\nLabels:')
        for label in labels:
            print(label.description)
        

cam.release()

cv2.destroyAllWindows()