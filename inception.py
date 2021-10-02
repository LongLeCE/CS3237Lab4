import reproducability
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import os.path

MODEL_FILE = 'flowers.hd5'


# Create a model if none exists. Freezes all training except in
# newly attached output layers. We can specify the number of nodes
# in the hidden penultimate layer, and the number of output
# categories.
def create_model(num_hidden, num_classes):
    base_model = InceptionV3(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False
    
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


# Loads an existing model file, then sets only the last
# 3 layers (which we added) to trainable.
def load_existing(model_file):

    # Load the model
    model = load_model(model_file)

    # Set only last 3 layers as trainable
    numlayers = len(model.layers)

    for layer in model.layers[:numlayers - 3]:
        layer.trainable = False
    
    for layer in model.layers[numlayers - 3:]:
        layer.trainable = True
    
    return model


def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20):
    if os.path.exists(model_file):
        print(f'\n*** Existing model found at {model_file}. Loading. ***\n\n')
        model = load_existing(model_file)
    else:
        print('\n*** Creating new model ***\n\n')
        model = create_model(num_hidden, num_classes)
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Create a checkpoint
    checkpoint = ModelCheckpoint(model_file)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(249, 249),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(249, 249),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )

    # Train last two layers
    for layer in model.layers[:249]:
        layer.trainable = False
    
    for layer in model.layers[249:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(learning_rate=1.2e-4, momentum=0.9), loss='categorical_crossentropy')

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )


def main():
    train(MODEL_FILE, train_path='flower_photos', validation_path='flower_photos', steps=114, num_epochs=10)


if __name__ == '__main__':
    main()
