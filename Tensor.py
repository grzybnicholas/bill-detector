from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as opencv
import os
import imghdr
import time as wait

# Function to prepare images by filtering out unwanted file extensions
def ImagePrep(path):
    AllowedExtension = ['jpg', 'jpeg', 'bmp', 'png']
    ImagePath = path
    for dir in os.listdir(ImagePath):
        for images in os.listdir(os.path.join(ImagePath ,dir)):
            Image = os.path.join(ImagePath, dir, images)
            try: 
                img = opencv.imread(Image)
                cash = img
                tip = imghdr.what(Image)
                if tip not in AllowedExtension:
                    os.remove(Image)
                    print("We just deleted a image with extension: {}".format(Image))
            except Exception as e: 
                print(e)

# Function to prepare image data for model training
def ImageData(ImagePath):
    tf.data.Dataset
    data = tf.keras.utils.image_dataset_from_directory(    
        ImagePath,
        image_size=(256, 256),
        batch_size=32,
        shuffle=True
    )
    class_names = data.class_names 
    dataITR = data.as_numpy_iterator()
    data = data.map(lambda x, y: (x/ 255.0, y))
    trainSize = 0.7
    val = 0.2
    test = 0.1
    total_batches = data.cardinality().numpy()  
    print("total batches: ", total_batches)
    train = int(total_batches * trainSize)
    validation = int(total_batches * val)
    test = int(total_batches * test + 1)
    trainData = data.take(train)
    valData = data.skip(train).take(validation)
    testData = data.skip(train + validation).take(test)
    print(len(trainData), len(valData), len(testData))
    return class_names, trainData, valData, testData

# Function to create a CNN model and train it
def MakeModelAndTrain(class_names, trainData, valData, epoch=15):
    model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3), kernel_regularizer=l2(0.01)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(class_names), activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_lr=0.00001)

    # Train the model
    print(model.summary())

    # Setup callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.0001)


    log = "logs"
    tensorboard = keras.callbacks.TensorBoard(log_dir=log)
    hist = model.fit(trainData, validation_data=valData, epochs= epoch, callbacks=[tensorboard])
    plotGraph(hist)
    return model

# Function to evaluate the trained model
def EvaluateModel(model, testData):
    prec = Precision()
    rec = Recall()
    acc = BinaryAccuracy()
    for batch in testData.as_numpy_iterator():
        images, labels = batch
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=-1)
        
        print("Labels shape:", labels.shape)
        print("Predictions shape:", predictions.shape)
        labels_onehot = tf.one_hot(labels, depth=predictions.shape[-1])
        
        prec.update_state(labels_onehot, predictions)
        rec.update_state(labels_onehot, predictions)
        acc.update_state(labels, predicted_labels)

    print("Precision: ", prec.result().numpy(), "Recall: ", rec.result().numpy(), "Accuracy: ", acc.result().numpy()) 

# Function to save the trained model
def SaveModel(model):
    try:
        model.save("Model.keras")
        return True
    except Exception as e:
        print(e) 
    return False

# Function to load a saved model
def LoadModel(file):
    model = keras.models.load_model(file)
    return model

# Function to make predictions on a single image
def Predict(model, image, class_names):
    img = opencv.imread(image)
    img = opencv.resize(img, (256, 256))
    img = np.reshape(img, (1, 256, 256, 3))
    prediction = model.predict(img)
    return class_names[np.argmax(prediction)]

# Helper function for making predictions using camera feed
def camHelper(model, class_names, frame):
    img = opencv.resize(frame, (256, 256))
    img = np.reshape(img, (1, 256, 256, 3))
    prediction = model.predict(img)
    return class_names[np.argmax(prediction)]

# Function to make predictions using camera feed
def cameraPredict(model, class_names):
    camera = opencv.VideoCapture(0)
    last_prediction_time = wait.time()  
    pred = "None"
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
        current_time = wait.time()
        if current_time - last_prediction_time >= 2:  
            pred = camHelper(model, class_names, frame)
            print(pred) 
            last_prediction_time = current_time  
        label = f'Predicted: {pred}' 
        opencv.putText(frame, label, (1, 20), opencv.FONT_HERSHEY_DUPLEX,.9 , (255, 255, 255), 2)
        opencv.imshow("Camera", frame)
        if opencv.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    opencv.destroyAllWindows()

# Function to continue training an existing model
def trainExistingModel(model, trainData, valData, epoch = 15):
    log = "logs"
    tensorboard = keras.callbacks.TensorBoard(log_dir=log)
    hist = model.fit(trainData, validation_data=valData, epochs= epoch, callbacks=[tensorboard])
    plotGraph(hist)
    return model

# Function to plot training and validation loss
def plotGraph(hist):
    fig = plt.figure()
    plt.plot(hist.history['loss'], color="red",label='Training Loss')
    plt.plot(hist.history['val_loss'], color = "green", label='Validation Loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc='lower left')
    plt.show()

# Function to display four images with their predictions
def FourImagesWithPrediction(model, class_names, testData):
    # Create a subplot of 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Test Images and Predictions')

    data_iterator = testData.as_numpy_iterator()

    # Get a batch of data
    batch = next(data_iterator)
    images, labels = batch

    # Only display the first 4 images and predictions in a 2x2 grid
    count = 0
    for i in range(2):
        for j in range(2):
            if count >= images.shape[0]:
                break
            # Prepare image for prediction
            img = images[count]
            prediction = model.predict(np.expand_dims(img, axis=0))
            predicted_label = class_names[np.argmax(prediction)]
            
            # Plotting
            axs[i, j].imshow(img.astype('float32'))  # Assume images are normalized to [0, 1]
            axs[i, j].title.set_text(predicted_label)
            axs[i, j].axis('off')  # Hide axes
            count += 1

    plt.tight_layout()
    plt.show()
