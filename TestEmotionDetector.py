import cv2
import numpy as np
from keras.models import model_from_json

# Diccionario para mapear etiquetas de emociones a nombres
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Cargar el modelo de la arquitectura desde un archivo JSON
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Cargar los pesos entrenados en el modelo
emotion_model.load_weights("model/emotion_model.h5")
print("Modelo cargado desde el disco")

# Iniciar la alimentación de la cámara web
cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("C:\\Users\\Milena\\Desktop\\Deteccion_emociones\\emotion.mp4")

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    # Utilizar un clasificador Haar Cascade para detectar caras en el fotograma
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el fotograma
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Procesar cada cara detectada
    for (x, y, w, h) in num_faces:
        # Dibujar un cuadro alrededor de la cara
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predecir la emoción en la cara recortada
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Mostrar la etiqueta de la emoción en el fotograma
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el fotograma con las detecciones de emociones
    cv2.imshow('Emotion Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
