# Face Recognition algorithm for AR glasses with eye tracker

#Target behind the project
Macular degeneration(MD) is a condition that affects the macula, which 
is the part of the eye responsible for central vision.MD leeds to rapid 
central vision loss and can make everyday activities like  recognising 
faces very difficult. In this project we aim to create an algorithm for 
AR smart glasses that a person can wear and be able to recognize faces.
One of the biggest problem in these kind face recogntion problems is the
low resolution and blury images. To figure this out we use [AdaFace](https://arxiv.org/abs/2204.00964)
for feature extraction that is an algorithm that has trained in both high 
and low resolution images. We also use 4K camera to have better accuracy.
Two problemes arise that affect real time performance.First by using 4K
camera we have to proccess more information and secondly if user in a 
crowded place ,the algorithm should detect every single person that takes
lot of time. To deal with we suggest the use of  AR smart glasses, embedded 
with eye trackers ,so that algorithm proccess a ROI in every frame. The 
desired window will created with center the spot whre the user;s gaze is.


![Demo](assets/demo.gif)
The demo shows the result of the face recogntion system. The algorithm can apllied
in glasses with Eye Tracker. The blue circle follows gaze. In this project we hadnt
an eye tracker so the circle moves with the mouse. After that step we pass a ROI with
cenet the circle's center in which we apply face detection and recogntion.

# Enviroment
Python == 3.6
Install libraries: `pip3 install -r requirements.txt`

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
####Steps
1. Place the dataset in the folder `Dataset/`
2. run
    `python createDataset.py -ntr <number of train images per person> -nte <number of test images per person> -ndf <DATASET_NAME> -non <number of persons>`
- Notes
   if you want all data for training and no testset run
    `python createDataset.py -ntr  float('inf') -ndf <DATASET_NAME>`

# Load Pretrained Models
Pretrained model from [this](https://github.com/mk-minchul/AdaFace).

| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |

Download it and place it in `Feature_Extractor/pretrained/`

# Create Embeddings
run 
  `python train_eval.py -EMBEDDINGS True -EVAL False -ef "output/Embeddings.pickle" -vef "output/validation_Embeddings.pickle" -ff "output/features.pickle"`
- Notes
  1.The features are the embeddings in a different format to pass as input in K-NN
  2.if you want to apply **data augmentation** run
  `python train_eval.py -EMBEDDINGS True -EVAL False -ef "output/Embeddings.pickle" -vef "output/validation_Embeddings.pickle" -ff "output/features.pickle" -DA True`

# Train neural Network for classification
run
  `python myneuralnetwork.py -td "output/Embeddings.pickle" -vd "output/validation_Embeddings.pickle" -ms True -mp "pretrained/classifier.hdf5"`
- Notes
  1.The ANN has only one hidden layer with 64 neurons.You can change it with -nnh <Number of Neurons> argument.
  2.If you want a different architecture you can modify the script 

# Evaluation of the models
run
  `python train_eval.py -KNN True -ANN True  -ef "output/Embeddings.pickle" -ff "output/features.pickle" -mp "pretrained/classifier.hdf5"`
- Notes
  1.If you want to test the model in blur testset run
  `python train_eval.py -KNN True -ANN True -bt True -ef "output/Embeddings.pickle" -ff "output/features.pickle" -mp "pretrained/classifier.hdf5"`
  2.For more see the arguments in the script `train_eval.py`

# Real time performance
run
  `python Realtime.py -v False -mp "pretrained/classifier.hdf5" -ef "output/Embeddings.pickle"`
- Notes
  if you want a video file as input 
1. Place your video file in `videos/<VIDEO_NAME>`
2. run
    `python Realtime.py -mp "pretrained/classifier.hdf5" -ef "output/Embeddings.pickle" -vf "videos/<VIDEO_NAME>"`


