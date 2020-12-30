import cv2
import os
import numpy as np

def get_path_list(root_path):
    full_human_path = []
    
    for label, human_path in enumerate(os.listdir(root_path)):
        if human_path != '.DS_Store':
            full_human_path.append('{}/{}'.format(root_path, human_path))
    
    return full_human_path


def get_class_names(root_path, train_names):
    labels = []
    images = []
    
    for label, full_human_path in enumerate(train_names):
        # print(label, full_human_path)
        for  image_path in os.listdir(full_human_path):
            full_image_path = '{}/{}'.format(full_human_path, image_path)
            # print(full_image_path)
            if not (full_image_path.endswith('.jpg') | full_image_path.endswith('.png') | full_image_path.endswith('.jpeg')):
                continue
                
            image = cv2.imread(full_image_path)

            # cv2.imshow('image', image)
            # cv2.waitKey(0)

            images.append(image)
            labels.append(label)

    # print(labels)
    # print(images)

    return images, labels
        


def detect_faces_and_filter(image_list, image_classes_list=None):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_list = []
    face_location = []
    face_class_list = []

    for label, image in enumerate(image_list):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = cascade.detectMultiScale(image_gray, 1.2, 3)

        for detected_face in detected_faces:
            x, y, width, height = detected_face
            face = image_gray[y:y+height, x:x+width]
            # cv2.imshow('face', face)
            # cv2.waitKey(0)
            face_list.append(face)
            face_location.append([x, y, width, height])
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

        if not (full_image_path.endswith('.jpg') | full_image_path.endswith('.png') | full_image_path.endswith('.jpeg')):
            continue
                
        image = cv2.imread(full_image_path)

        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        images.append(image)
    
    return images


def predict(recognizer, test_faces_gray):
    predict_results = []

    for face in test_faces_gray:
        label, confidence = recognizer.predict(face)

        predict_results.append([label, confidence])

    return predict_results
    

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''
    
def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image
        Before the image combined, it must be resize with
        width and height : 200px

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

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
    # predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    # combine_and_show_result(predicted_test_image_list)