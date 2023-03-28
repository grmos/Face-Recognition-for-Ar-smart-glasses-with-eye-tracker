# Face Recognition algorithm for AR glasses with eye tracker

# Target behind the project
Macular degeneration (MD) is a condition that affects the macula, which is the part of the eye responsible for the central vision. MD leads to rapid central vision loss and can make everyday activities like recognizing faces very difficult. In this project we aim to create an algorithm for AR smart glasses that a person can wear and be able to recognize faces. One of the biggest problems in these kinds of face recognition systems is the low resolution and blurry images. To figure this out we used [Adaface](https://arxiv.org/abs/2204.00964) for feature extraction which is an algorithm that has been trained in both high and low resolution images. We also used a 4K camera to have better accuracy. This resulted in the occurrence of two problems that affect the real time performance.  Firstly, by using the 4K camera we have to process more pieces of information and secondly if the user is in a crowded place, the algorithm has to detect every single person and therefore takes a lot of time.  To deal with that we suggest the use of AR glasses, embedded with eye trackers, so that the algorithm will process a ROI in every frame. The desired window has as a center the spot where the user’s gaze is located.

# Demo 
Full video [here](https://drive.google.com/file/d/1x0mvlKvxBczQxlJIhxSdiSrEDS3CKZbG/view?usp=sharing)
![Demo](assets/demo.gif)
The demo shows the result of the face recognition system. The algorithm can be applied in the glasses with Eye Tracker. The blue circle follows the gaze of the person wearing them. In this project we didn’t have an eye tracker so the circle is moved by the mouse of the PC. After that step we included as an entrance to the system a ROI which core is the center of the circle in which we apply the face detection and recognition.

# Enviroment
- Python == 3.6.13
- Install libraries: `pip3 install -r requirements.txt`
- conda == 22.9.0

# Data Preparation
The data should have the following structure.
```
└──DATASET_NAME
    └── person_1
            └──1.jpg
                ⋮
            └──n.jpg
            ⋮
    └── person_N                                                                                  
            └──1.jpg
                ⋮
            └──n.jpg                                                                             
```
#### Steps
1. Place the dataset in the folder `Dataset/`
2. run
    `python createDataset.py -ntr <number of train images per person> -nte <number of test images per person> -ndf <DATASET_NAME> -non <number of persons>`
- Notes
1. if you want all data for training and no testset run
    `python createDataset.py -ntr  <large number> -ndf <DATASET_NAME>`
lareg number should be at least max(class images)
2. If you have a trainset and a testset folder run python `createDataset.py -ntr 5000 -ndf <TRAIN_DATASET_NAME>` and then `python createDataset.py -ntr 0 -nte 5000 -ndf <TEST_DATASET_NAME>`
  
# Load Pretrained Models
Pretrained model from [this](https://github.com/mk-minchul/AdaFace).

| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |

Download it and place it in `Feature_Extractor/pretrained/`

# Create Embeddings
run 
  `python train_eval.py -EMBEDDINGS 1 -EVAL 0 -ef "output/Embeddings.pickle" -vef "output/validation_Embeddings.pickle" -ff "output/features.pickle" -bval 1`
- Notes
1. The features are the embeddings in a different format to pass as input in K-NN.
2. if you want to apply **data augmentation** run
  `python train_eval.py -EMBEDDINGS 1 -EVAL 0 -ef "output/Embeddings.pickle" -vef "output/validation_Embeddings.pickle" -ff "output/features.pickle" -bval 1 -DA 1`
3. If you dont want validation data run
    `python train_eval.py -EMBEDDINGS 1 -EVAL 0 -ef "output/Embeddings.pickle" -ff "output/features.pickle" -bval 0`

# Train neural Network for classification
run
  `python myneuralnetwork.py -td "output/Embeddings.pickle" -vd "output/validation_Embeddings.pickle" -ep 5000 -ms 1 -mp "pretrained/classifier.hdf5"`
- Notes
1. The ANN has only one hidden layer with 64 neurons.You can change it with -nnh <Number of Neurons> argument.
2. If you want a different architecture you can modify the script 
3. If you havent create validation data for previous step run
    `python myneuralnetwork.py -td "output/Embeddings.pickle" -vd "output/validation_Embeddings.pickle" -bvd 0 -ep 5000 -ms 1 -mp "pretrained/classifier.hdf5"`

# Evaluation of the models
run
  `python train_eval.py -EVAL 1 -KNN 1 -ANN 1  -ef "output/Embeddings.pickle" -ff "output/features.pickle" -mp "pretrained/classifier.hdf5"`
- Notes
1. If you want to test the model in blur testset run
  `python train_eval.py -EVAL 1 -KNN 1 -ANN 1 -bt 1 -ef "output/Embeddings.pickle" -ff "output/features.pickle" -mp "pretrained/classifier.hdf5"`
2. For more see the arguments in the script `train_eval.py`

# Real time performance
run
  `python Realtime.py -v 0 -mp "pretrained/classifier.hdf5" -ef "output/Embeddings.pickle"`
- Notes
  if you want a video file as input 
1. Place your video file in `videos/<VIDEO_NAME>`
2. run
    `python Realtime.py -mp "pretrained/classifier.hdf5" -ef "output/Embeddings.pickle" -vf "videos/<VIDEO_NAME>"`


