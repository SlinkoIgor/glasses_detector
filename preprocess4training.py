from PIL import Image
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os


def preprocess():
    os.makedirs('glasses_gan/crops/glasses')
    os.makedirs('glasses_gan/crops/no_glasses')

    for image_path in tqdm(glob.glob('glasses_gan/Images/*/*.jpg')):
        #     print(image_path)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            face_locations = face_recognition.face_locations(image, model='cnn')

        face_landmarks_list = face_recognition.face_landmarks(image, face_locations=face_locations)[0]
        eyebrows = np.array(face_landmarks_list['left_eyebrow'] + face_landmarks_list['right_eyebrow'])
        nose_tip = np.array(face_landmarks_list['nose_tip'])

        #     plt.imshow(image)
        #     plt.scatter(eyebrows[:,0], eyebrows[:,1])
        #     plt.scatter(nose_tip[:,0], nose_tip[:,1])
        #     plt.show()

        all_keypoints = np.array(face_landmarks_list['left_eyebrow'] +
                                 face_landmarks_list['right_eyebrow'] +
                                 face_landmarks_list['nose_tip'])

        left, right, top, bottom = (all_keypoints[:, 0].min(), all_keypoints[:, 0].max(),
                                    all_keypoints[:, 1].min(), all_keypoints[:, 1].max())

        width = right - left
        left = max(0, left - width // 3)
        right = min(image.shape[1] - 1, right + width // 3)

        image_crop = image[top:bottom, left:right]
        Image.fromarray(image_crop).save(image_path.replace('Images', 'crops'))
    #     plt.imshow(image_crop)
    #     plt.show()


