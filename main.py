import cv2
import os
import numpy as np

def get_path_list(root_path):
    full_human_path = []
    
    for human_path in os.listdir(root_path):
        full_human_path.append('{}/{}'.format(root_path, human_path))
    
    return full_human_path


def get_class_names(root_path, train_names):
    classes = []
    images = []
    
    for label, human_name in enumerate(os.listdir(root_path)):
        full_human_path = train_names[label]
        
        for image_path in os.listdir(full_human_path):
            full_image_path = '{}/{}'.format(full_human_path, image_path)

            if not (full_image_path.endswith('.jpg') | full_image_path.endswith('.png') | full_image_path.endswith('.jpeg') | full_image_path.endswith('.JPG')):
                continue
                
            image = cv2.imread(full_image_path)

            images.append(image)
            classes.append(label)

    return images, classes
        


def detect_faces_and_filter(image_list, image_classes_list=None):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_list = []
    face_location = []
    face_class_list = []

    for label, image in enumerate(image_list):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = cascade.detectMultiScale(image_gray, 1.2, 7)

        if(len(detected_faces) == 1):
            for detected_face in detected_faces:
                x, y, width, height = detected_face
                face = image_gray[y:y+height, x:x+width]

                face_list.append(face)
                face_location.append(detected_face)
                if(image_classes_list):
                    face_class_list.append(image_classes_list[label])

    return face_list, face_location, face_class_list


def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer


def get_test_images_data(test_root_path):
    images = []

    for image_path in os.listdir(test_root_path):
        full_image_path = '{}/{}'.format(test_root_path, image_path)

        if not (full_image_path.endswith('.jpg') | full_image_path.endswith('.png') | full_image_path.endswith('.jpeg') | full_image_path.endswith('.JPG')):
            continue
                
        image = cv2.imread(full_image_path)
        images.append(image)
    
    return images


def predict(recognizer, test_faces_gray):
    predict_results = []

    for face in test_faces_gray:
        label, confidence = recognizer.predict(face)
        predict_results.append([label, confidence])

        

    return predict_results
    

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    root_path = 'dataset/train'
    human_name = os.listdir(root_path)

    result_images = []

    for i, test_face in enumerate(test_image_list):
        x = test_faces_rects[i][0]
        y = test_faces_rects[i][1]
        width = test_faces_rects[i][2]
        height = test_faces_rects[i][3]

        text = human_name[predict_results[i][0]]
        image = test_face

        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 10)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=10, color=(0, 0, 255))

        result_images.append(image)

    return result_images
    
    
def combine_and_show_result(image_list):
    image0 = cv2.resize(image_list[0], (200, 200))  
    image1 = cv2.resize(image_list[1], (200, 200))
    image2 = cv2.resize(image_list[2], (200, 200))
    image3 = cv2.resize(image_list[3], (200, 200))
    image4 = cv2.resize(image_list[4], (200, 200))

    image_stacked = np.hstack((image0, image1, image2, image3, image4))

    cv2.imshow('Result', image_stacked)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = 'dataset/train'
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)     

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = 'dataset/test'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    combine_and_show_result(predicted_test_image_list)
