from Tensor import *
def main():
    Path = "Data"
    ImagePrep(Path)
    class_names, trainData, valData, testData = ImageData(Path)
    print(class_names)
    model = MakeModelAndTrain(trainData, valData, class_names)
    cameraPredict(model, class_names)
    Option = input("Do you want to save the model? (Y/N): ")
    if Option.upper() == "Y":
        SaveModel(model)


if __name__ == "__main__":
    main()
