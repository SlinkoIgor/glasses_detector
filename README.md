# Glasses Detector
indicates whether the person is wearing glasses. Glasses that are not worn on the bridge of the nose are not classified as worn.

# Operating principle:
With the help of facial landmark detection, a rectangle with the eyes of a person is determined. Next, CNN is used to classify whether there are points in this rectangle.

Model for facial landmark detection: https://github.com/ageitgey/face_recognition

Classification model: Mobile Net V3 Small (minimalistic), trained on a dataset https://www.kaggle.com/jeffheaton/glasses-or-no-glasses , with preprocessing

# Installation:
git clone https://github.com/SlinkoIgor/glasses_detector.git
cd glasses_detector
pip install -r requirements.txt

# Example:
On the image folder:
```bash
python test.py --dir=test_data/example_data_glasses/with_glasses
```

# Known issues and ways to improve:
- Bad landmarks detection on rotated images (make sure images are not rotated before running)
- Bad landmarks detection on faces with masks
- Runs slowly on CPU when utilizing cnn for landmarks detection. You can switch it off
