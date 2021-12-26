import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np


def ProcessImage(filepath, *, color = None):
    
    width = 150
    height = 150
    img = cv2.imread(filepath)
    if color != None:   
        img = cv2.cvtColor(img, color)
    imgRescale = cv2.resize(img, (width, height), cv2.INTER_AREA)
    imgRescale = imgRescale.astype(np.float64)
    imgRescale /= 255
    return imgRescale

def GetCategories():
    train_label_mapping = {}
    
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'DataSet\Training_Data'))):
        train_label_mapping[number] = folder
    return train_label_mapping

def GetFiles():
    imgTrainList = []
    imgName = []
    for number, folder in enumerate(os.listdir(os.path.join(os.getcwd(), 'DataSet\Training_Data'))):
        for image_name in os.listdir(os.path.join(os.getcwd(), 'DataSet\Training_Data', folder)):
            
            imgName.append(number)

            image = ProcessImage(os.path.join(os.getcwd(), 'DataSet\Training_Data', folder, image_name))
            imgTrainList.append(image)
    return (imgTrainList, imgName)

def CreateModel():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
    ])

    model.summary()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def Train(model, train_image, train_filename):
    checkpoint_path = "DataSet\checkpoint\cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(train_image, train_filename, epochs=15, callbacks=[checkpoint_callback])

def Predict(model, mapping):
    images = []
    results = []
    images_RGB = []
    for images_path in os.listdir(os.path.join(os.getcwd(), 'DataSet\Data_Test')):
        
        image = ProcessImage(os.path.join(os.getcwd(), 'DataSet\Data_Test', images_path))
        images.append(image)
        image = np.array(image)
        
        # predict the image
        result = model.predict(np.expand_dims(image, axis=0))
        result = np.argmax(result[0])
        results.append(mapping[result])
        
        # get rgb image to display for 
        imageRGB = ProcessImage(os.path.join(os.getcwd(), 'DataSet\Data_Test', images_path), color=cv2.COLOR_BGR2RGB)
        images_RGB.append(imageRGB)

    
    plt.figure(figsize=(12,12))
    for i in range(len(images_RGB)):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_RGB[i])
        plt.xlabel(results[i])
    plt.show()


train_image, train_filename = GetFiles()
train_image = np.array(train_image)
train_filename = np.array(train_filename)


model = CreateModel()


Train(model, train_image, train_filename)
model.save('TrainedModel')


# model.load_weights(os.path.join(os.getcwd(), 'TrainedModel', 'saved_model.pb'))
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'TrainedModel'))
Predict(model, GetCategories())