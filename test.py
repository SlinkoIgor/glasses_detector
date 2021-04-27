import logging
logging.basicConfig(level=logging.ERROR)

import face_recognition
import glob
import argparse
import numpy as np
import tensorflow as tf


class GlassesPredictor:
    def __init__(self, model_path):
        super().__init__()
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    @staticmethod
    def get_eyes_crop(image_path):
        try:
            image = face_recognition.load_image_file(image_path)
        except:
            logging.warning('probably not an image file', image_path)
            return None

        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            face_locations = face_recognition.face_locations(image, model='cnn')

        if not face_locations:
            logging.warning('no face on image file', image_path)
            return None

        face_landmarks_list = face_recognition.face_landmarks(image, face_locations=face_locations)[0]
        all_keypoints = np.array(face_landmarks_list['left_eyebrow'] +
                                 face_landmarks_list['right_eyebrow'] +
                                 face_landmarks_list['nose_tip'])

        left, right, top, bottom = (all_keypoints[:, 0].min(), all_keypoints[:, 0].max(),
                                    all_keypoints[:, 1].min(), all_keypoints[:, 1].max())

        width = right - left
        left = max(0, left - width // 3)
        right = min(image.shape[1] - 1, right + width // 3)

        image_crop = image[top:bottom, left:right]
        return image_crop

    def predict_on_dir(self, data_path):
        for image_path in glob.glob(data_path + '/*'):
            if self.predict_on_image_path(image_path):
                print(image_path)

    def predict_on_image_path(self, image_path):
        img_size = (64, 128)
        image_crop = self.get_eyes_crop(image_path)
        if image_crop is None:
            return False

        crop_tensor = tf.image.convert_image_dtype(image_crop, tf.float32)
        crop_tensor = tf.image.resize(crop_tensor, size=img_size) * 255.

        return self.predict_on_image(crop_tensor)

    def predict_on_image(self, image):
        input_index = self.interpreter.get_input_details()[0]['index']
        output_index = self.interpreter.get_output_details()[0]['index']
        self.interpreter.set_tensor(input_index, np.expand_dims(image, axis=0))
        self.interpreter.invoke()
        has_glasses = self.interpreter.get_tensor(output_index).flatten()[0] < 0
        return has_glasses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='dir-path with images', required=True)
    parser.add_argument('--model', help='path to tf-lite fp16 mode', default='weights/fp16_model.tflite')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    GlassesPredictor(model_path=args.model).predict_on_dir(args.dir)
