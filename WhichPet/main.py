import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.45))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.35))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# For Image Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('./dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# End - Image Augmentation
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/25,
                         epochs = 100,
                         validation_data = test_set,
                         validation_steps = 2000/25)


def Predict(filename):
    load_test = image.load_img(filename ,target_size=(64 ,64))
    load_test = image.img_to_array(load_test)
    load_test = np.expand_dims(load_test,0)
    if classifier.predict(load_test) == 0:
        print("Cat")
    else:
        print("Dog")

Predict("cat.4944.jpg")
 
# Save the model for further use.
def Load_Model2JSON(model):
    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("model2.h5")
    print("Saved model to disk")

Load_Model2JSON(classifier)

# Loading model from json file.
def Load_ModelFJSON():
    from keras.models import model_from_json
    json_file = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model2.h5")
    print("Loaded model from disk")
    return loaded_model
