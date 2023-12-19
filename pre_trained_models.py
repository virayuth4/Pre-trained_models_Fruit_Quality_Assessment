import os
import image
import random
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt


from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

#import pre-trained models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0



fruits = [
    "FreshApple",
    "FreshBanana",
    "FreshGrape",
    "FreshGuava",
    "FreshJujube",
    "FreshOrange",
    "FreshPomegranate",
    "RottenApple",
    "RottenBanana",
    "RottenGrape",
    "RottenGuava",
    "RottenJujube",
    "RottenOrange",
    "RottenPomegranate",
]

#HYPER_PARAMTERS
TARGET_SIZE = (224,224)
INCEPTION_SIZE = (299,299)
BATCH_SIZE = 64
EPOCHS = 20
TESTING_SIZE = 10

data_dir = '/path_to_your_data/'
train_path = data_dir+'/train'
val_path = data_dir+'/val'
test_path = data_dir+'/test'
folders = glob(train_path+'/*')




mobilenet_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),include_top=False,weights='imagenet')
vgg16= tf.keras.applications.vgg16.VGG16(input_shape=(224, 224, 3),include_top=False,weights='imagenet')
vgg19 = tf.keras.applications.vgg19.VGG19(input_shape=(224, 224, 3),include_top=False, weights='imagenet')
inception = tf.keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3),include_top=False,weights='imagenet')
#the data is alreay pre-process so for MobiletNet-V3-S & MobiletNet-V3-L, include_preprocessing=False must be included
mobilev3_small = MobileNetV3Small(input_shape=(224,224,3),include_top=False, weights="imagenet", include_preprocessing=False)
mobilev3_large = MobileNetV3Large(input_shape=(224,224,3),include_top=False, weights="imagenet", include_preprocessing=False) 
models = [mobilenet_v2, vgg16, vgg19, inception, mobilev3_small, mobilev3_large]



  
def main(model=inception, trainable_layer=False,testing=True):
  '''
  Arguments:
    model: Variable, specify your model name.
    trainable_layer: Boolean, for fine-tuning set trainable_layer=False, for training the model from scratch set trainable_layer=True.
    testing: Boolen, to test the random images from the directory set testing=True.
  '''
  if trainable_layer==False:
    for model in models:
      for layer in model.layers:
        layer.trainable = False
  else:
    for model in models:
      for layer in model.layers:
        layer.trainable = True
  
  if model == inception:
    size = INCEPTION_SIZE
  else:
    size = TARGET_SIZE

  train_datagen = ImageDataGenerator(rescale=1./255)
  test_datagen = ImageDataGenerator(rescale=1./255)


  training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=size,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')
  val_set = train_datagen.flow_from_directory(val_path,
                                                target_size=size,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')
  test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=size,
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')
  
  x = Flatten()(model.output)
  prediction = Dense(len(folders), activation='softmax')(x)
  model_ = Model(inputs=model.input, outputs=prediction)

  #compile the model
  model_.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

  #train the model
  train_engine = model_.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
    )
 
  '''
  To save the trained model use, model_.save('/path/name_youe_model.keras'). For example:
    fine_tuned_vgg16 = model_.save('/path_to_save_the_model/fine_tuned_vgg16.keras')
  '''

  if testing == False:
    print('Training Complete')
  else:
    wrong_list = []
    classes_ = training_set.class_indices
    directory = test_path
    for fruit in fruits:
      # the directory path for the current fruit
      fruit_directory = os.path.join(directory, fruit)

      # Get a list of all image files in the directory
      all_images = [i for i in os.listdir(fruit_directory) if os.path.isfile(os.path.join(fruit_directory, i))]

      # Select 3 random images without overlapping
      random_images = random.sample(all_images, min(TESTING_SIZE, len(all_images)))

      # Process and display each selected image
      for image in random_images:
        image_path = os.path.join(fruit_directory, image)
        # Load the image 
        img = Image.open(image_path)
        img = img.resize(size)
        img = np.array(img)
        plt.imshow(img.astype('uint8'))
        plt.show()

        img = np.expand_dims(img, axis=0)
        img = img/255.0

        prediction = model_.predict(img)
        index = np.argmax(prediction)
        ACTUAL_FRUIT = fruit
      
        print('Image Name:', image)
        for key, value in classes_.items():
          if index == value:
            PREDICTED_FRUIT = key
            print('ACTUAL:', ACTUAL_FRUIT)
            print('PREDICTED:', PREDICTED_FRUIT)
            if PREDICTED_FRUIT == fruit:
              print('PREDICTION STATUS: CORRECT')
            else:
              print('PREDICTION STATUS: WRONG')
              wrong_list.append(image)


        print('WRONG_PREDICTIONS:',wrong_list)
  


if __name__ == '__main__':
  main(model=inception)
