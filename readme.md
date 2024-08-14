# **[*TensorFlow*]** 

This document describes how to use the tensor.py script with main.py for currency detection using a TensorFlow model named Model.keras. First, we import functions like LoadModel and cameraPredict from tensor.py into main.py. The LoadModel function loads a pre-trained model that recognizes different currencies. Once loaded, cameraPredict utilizes a webcam feed to perform real-time currency detection. Additionally, there is a Predict function that allows for currency detection using static image files. This setup requires TensorFlow, OpenCV, and NumPy libraries to function correctly, enabling the practical application of deep learning for currency recognition both in real-time and from images. 

To use you can just open **Main.py** and run it. It should load the pre-train model and trun on camera for prediction. To get out of the camera prediction, press q and it should close. There is also a **predict()** function that takes in a image and predicts the currency type. There are provided images that can be used for testing.

# **[*YOLO Detection*]**

Project used Yolov5. The model in which we trained our model with is the Yolov5s.pt. The two main components are detect.py and train.py. The outcome of our training is the module name 350epoch.pt. To train the module you must have NVIDIA CUDA to train efficiently. If you decide to run 350epoch.pt with detect.py we tested it with our webcam(source 0) and ran it with a 5 ms repeater and a threshold of the confidence was 0.7.

Some early struggles we had with Yolov5 was gathering a good dataset for all the currencies which was our original goal, but instead we opted to make a module in which detected the front and back of USD Federal Notes. The module covers the $1 , $5, $10, $20, $50, specifically it was trained to detect the front and back of the notes through the use of labels in the dataset. We see that in the module trained with 350 epoch we get very accurate results, but how does this compare to TensorFlow. We find that our results are more accurate, but are justify as our data set was more than just 2400 images solely for USD bills. The bounding rectangle around the object detection is part of the yolo model but is trained with the dataset and has a 91% accuracy at the moment. Further improvements we can pursue with finding more datasets greater than 2,000 to fuel our yolo model in which we can create a more accurate and expansive model. In order to train it, you have to use a command within the terminal like python train.py --img 640 --epochs 350 --data data.yaml --weights yolov5s.pt, in which if using python, you specify the batch size, the data set, and the weights to train the model with. For the actual detection part, once its trained, you can then use the terminal to exectue the detect.py file with the use of a command like python detect.py --source 0 (0 is default for webcam for yolov5) , if you want to add any weights you attach the --weights flair to the end of that command (for example one of our weights for detection is 350epoch.pt in the exp21 folder in the weights).


[Download files for YOLO Detection project from Google Drive](https://drive.google.com/drive/folders/1vgToRy-XqEsTDR8Q_SWc86Rz-3ybA5Os?usp=drive_link)

# [**ISSUE WITH SUBMITTING ON GRADESCOPE**]

**Model Download**

The `Model.keras` file is too large to be included directly. Please download it from the following link:

[Download Model.keras from Google Drive](https://drive.google.com/drive/folders/1BMxW4czJk5LuvFMDe0FAneMsytD7VvYu?usp=sharing)

The data set that we use is included and in the main is also a way to train the model using the data set. For this, all that needs to be done instead of loading the model is to use the **MakeModelAndTrain()** which is already configured to make and train it on the data set in **Main.py** file. MakeModelAndTrain function takes in 4 parameters: 
1. - Training Images
2. - Validation Images
3. - ClassNames (Provided by **ImageData()** function used in the  beginning)
4. - ***Optional*** - Epochs (Default is 15 if no input is given)

After it finishes training for 15 epochs, the **cameraPredict()** function will be called and the camera will be turned on for testing. If you want to use an image to get a prediction, use the **Predict()** function and pass in the image as an argument. The **Predict()** function takes in 3 parameters

1. - Model Variable
2. - Image Dir
3. - ClassNames (Provided by **ImageData()** function used in the  beginning)

With all this done, other functions can be used to with model created such as **FourImagesWithPrediction(model, class_names, testData)** which provides the prediction of each image with the image being below the prediction. Saving the model can also be done by using **SaveModel(model)** which saves the model as model.keras (Hardcoded to save it that way). **Loadmodel(file)** loads the model from either a *.keras* or a *.h5* file. **trainExistingModel(model, trainData, valData, epoch = 15)** takes in the model either from loading the model from a file (.h5 or .keras) and contines to train model from where it left off. 

**Slight Overhead** 

When model is finished training, a plot will apear of **Val_Loss* and **Train_Loss*. CLose window showing these losses for the program to continue further. 

Thank You **:)**

