import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Define dataset path
train_dir = "CT_images/Train"
test_dir = "CT_images/Test"

#Img dimensions & Batch Size
img_size = (150, 150)
batch_size = 32

#Data Augmentation & Normalization
train_datagen = ImageDataGenerator(
    rescale = 1.0/255.0, 
    rotation_range = 20,
    zoom_range = 0.2,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator(
    rescale = 1.0/255.0
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode ='binary'
)

#CNN 
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid') # Binary classification normal & stone

])

# compile model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics =['accuracy']
)

#Training model
history = model.fit(
    train_generator,
    validation_data = test_generator,
    epochs = 10
)

#Accuracy Evaluation
test_loss ,test_acc = model.evaluate(test_generator)
print(f"Test_accuracy: {test_acc:.2f}")

#Solve Trainig model
model.save("best_kidney_stone_model.h5")
print("Model saved as best_kidney_stone_model.h5")