# Glasses Detector
indicates whether the person is wearing glasses. Glasses that are not worn on the bridge of the nose are not classified as worn.

# Operating principle:
With the help of facial landmark detection, a rectangle with the eyes of a person is determined. Next, CNN is used to classify whether there are points in this rectangle.

Model for facial landmark detection: https://github.com/ageitgey/face_recognition

Classification model: Mobile Net V3 Small (minimalistic), trained on a dataset https://www.kaggle.com/jeffheaton/glasses-or-no-glasses , with preprocessing . Classification network was then quantized to fp16 resulting with ~2MB size.

Classification part works ~20ms on i5 CPU for 1 image.

# Installation:
git clone https://github.com/SlinkoIgor/glasses_detector.git
cd glasses_detector
pip install -r requirements.txt

# Example:
On the image folder:
```bash
python test.py --dir=test_data/example_data_glasses/with_glasses
```

On one image:
```python
from test import GlassesPredictor
predictor = GlassesPredictor(model_path='weights/fp16_model.tflite')
predictor.predict_on_image_path('test_data/example_data_glasses/with_glasses/0.jpg')
>>> True
```

# Known issues:
- Bad landmarks detection on rotated images (make sure images are not rotated before running)
- Bad landmarks detection on faces with masks
- Runs slowly on CPU when utilizing CNN for landmark detection. You can switch it off

Current landmark detector works really good on most of images and it's really fast on CPU in cases in cases when dlib recognizes face, otherwise it switches to cnn approach which is significantly slower on CPU

# Ways to improve
Maybe it'll be a good idea to try these landmark detectors:
- https://github.com/1adrianb/face-alignment
- https://github.com/HRNet/HRNet-Facial-Landmark-Detection

Glasses classifier works perfectly on my test examples (found in Google). But given more time I'd prefer to increase generalizability by adding a couple of datasets:
- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- https://www.kaggle.com/ashish2001/512x512-face-parsing-segmentation-tfrecords

Also, no augmentations were harmed during the train procedure – an unfortunate omission.

