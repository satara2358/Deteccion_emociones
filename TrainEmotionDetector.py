# Importa las bibliotecas necesarias
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Inicializa generadores de datos de imágenes y aplica reescalamiento
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocesa todas las imágenes de entrenamiento
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Preprocesa todas las imágenes de validación
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Crea la estructura del modelo de red neuronal
emotion_model = Sequential()

# Agrega capas convolucionales, de agrupación y de dropout
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# Compila el modelo con la función de pérdida y el optimizador  => 
# emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Entrena el modelo neuronal
# emotion_model_info = emotion_model.fit_generator(
#     train_generator,
#     steps_per_epoch=28709 // 64,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=7178 // 64)
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

# # Guarda la estructura del modelo en un archivo JSON
# model_json = emotion_model.to_json()
# with open("emotion_model.json", "w") as json_file:
#     json_file.write(model_json)

# # Guarda los pesos entrenados del modelo en un archivo .h5
# emotion_model.save_weights('emotion_model.h5')
# Guardar arquitectura en JSON
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Guardar pesos
emotion_model.save_weights("emotion_model.weights.h5")
print("Modelo guardado en disco.")