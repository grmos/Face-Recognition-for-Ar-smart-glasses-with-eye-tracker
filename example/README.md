# Example of training and evaluation in PubFig Dataset (256x256 jpg)



# Data Preparation
1. Download the dataset from [here](https://www.kaggle.com/datasets/kaustubhchaudhari/pubfig-dataset-256x256-jpg)
unzip and place it in folder `Dataset/`
- The data  have the following structure.
```
CelebDataProcessed
    └── Abhishek Bachan
            └──1.jpg
                ⋮
            └──n.jpg
            ⋮
    └── William Macy                                                                                  
            └──1.jpg
                ⋮
            └──n.jpg                                                                             
```
2. run
    `python createDataset.py -ntr <number of train images per person> -nte <number of test images per person> -ndf "CelebDataProcessed" -non <number of persons>`

# Load Pretrained Models
Pretrained model from [this](https://github.com/mk-minchul/AdaFace).

| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R100 | MS1MV3     | [gdrive](https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing) |

Download it and place it in `Feature_Extractor/pretrained/`

# Create Embeddings
run 
  `python train_eval.py -EMBEDDINGS 1 -EVAL 0 -ef "output_celebA/celebA_Embeddings.pickle" -vef "output_celebA/celebA_val_Embeddings.pickle" -ff "output_celebA/celebA_features.pickle" -bval 1`
- Notes
1. The features are the embeddings in a different format to pass as input in K-NN
2. if you want to apply **data augmentation** run
  `python train_eval.py -EMBEDDINGS 1 -EVAL 0 -ef "output_celebA/celebA_Embeddings.pickle" -vef "output_celebA/celebA_val_Embeddings.pickle" -ff "output_celebA/celebA_features.pickle" -bval 1-DA 1`

# Train neural Network for classification
run
  `python myneuralnetwork.py -td "output_celebA/celebA_Embeddings.pickle" -vd "output_celebA/celebA_val_Embeddings.pickle" -ep 1000 -ms 1 -mp "pretrained/classifier.hdf5"`
- Notes
1. The ANN has only one hidden layer with 64 neurons.You can change it with -nnh <Number of Neurons> argument.
2. If you want a different architecture you can modify the script 

# Evaluation of the models
run
  `python train_eval.py -EVAL 1 -KNN 1 -ANN 1  -ef "output_celebA/celebA_Embeddings.pickle" -ff "output_celebA/celebA_features.pickle" -mp "pretrained/classifier.hdf5"`
- Notes
1. If you want to test the model in blur testset run
  `python train_eval.py -EVAL 1 -KNN 1 -ANN 1 -bt 1 -ef "output_celebA/celebA_Embeddings.pickle" -ff "output_celebA/celebA_features.pickle" -mp "pretrained/classifier.hdf5"`
2. The KNN score might be 192/192 ans ANN 72/72. That depends on the thressholds that can modify in the script/
3. For more see the arguments in the script `train_eval.py`



