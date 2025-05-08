# Emotion_detection_with_CNN

![emotion_detection](https://github.com/datamagic2020/Emotion_detection_with_CNN/blob/main/emoition_detection.png)

### Packages need to be installed
- version 3.9.21 of python || env tensor || anaconda
- pip install numpy
- pip install matplotlib
- pip install scikit-learn
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow  ||| optional || pip install tensorflow-cpu
- pip install pillow
- pip3 install scipy   || opcional
- conda install -c anaconda scipy || Opcional

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TranEmotionDetector.py

It will take several hours depends on your processor. (On i7 processor with 16 GB RAM it took me around 4 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
python TestEmotionDetector.py

### download FER2013 dataset
- data from below link and put in data folder under your project directory
- https://www.kaggle.com/datasets/msambare/fer2013/



Fila 3, columna 3 = 445: El modelo clasificó correctamente 445 imágenes de Happy como Happy.

Fila 3, columna 5 = 376: El modelo confundió 376 imágenes de Happy como Surprise.

Fila 0, columna 3 = 247: El modelo confundió 247 imágenes de Angry como Happy.

❌ ¿Qué te dice esta matriz?
El modelo tiende a confundir emociones similares, como:

Angry ↔ Happy

Fear ↔ Sad / Surprise

Happy ↔ Surprise / Neutral

Disgust (clase 1) está muy mal representada → probablemente pocas imágenes o poca capacidad del modelo para distinguirla.

Happy (clase 3) parece tener más aciertos que otras, pero aún así presenta confusiones.