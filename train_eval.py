from helper_functions import *
import time
import argparse

## 0=> False 1=> True


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-EMBEDDINGS", "--booltraining", type=int, default=0,
	help="create face embeddings 0=>No Training, 1=>Training")
ap.add_argument("-KNN", "--boolmyknn", type=int, default=0,
	help="Apply KNN as classifier with face embeddings as data ")
ap.add_argument("-PCA", "--boolpca", type=int, default=0,
	help="Apply PCA in 512-d face embeddings and then train with SVM ")
ap.add_argument("-ANN", "--boolnn", type=int, default=1,
	help="Apply pretrained ANN as classifier with face embeddings as data ")
ap.add_argument("-SVM", "--boolSVM", type=int, default=0,
	help="Apply pretrained SVM as classifier with face embeddings as data ")
ap.add_argument("-T_SVM", "--boolSVMtrain", type=int, default=0,
	help="Train SVM model in face embeddings  ")
ap.add_argument("-EVAL", "--booleval", type=int, default=0,
	help="Evaluate models 0=>No evaluation, 1=>evaluation ")
ap.add_argument("-th_SVM", "--thresshold_SVM", type=float, default=0.1,
	help="thresshold score for classification in SVM ")
ap.add_argument("-th_KNN", "--thresshold_KNN", type=float, default=0.35,
	help="thresshold score for classification in KNN  ")
ap.add_argument("-th_ANN", "--thresshold_ANN", type=float, default=0.95,
	help="thresshold score for classification in ANN  ")
ap.add_argument("-bt", "--blur_testset", type=bool, default=False,
	help="True if you want to evaluate your model in a blur testset ")
ap.add_argument("-ef", "--embeddingsfolfer", type=str, default="output_celebA/celebA_Embeddings.pickle",
	help="path to embeddings folder ")
ap.add_argument("-vef", "--val_embeddingsfolfer", type=str, default="output_celebA/celebA_val_Embeddings.pickle",
	help="path to validationembeddings folder ")
ap.add_argument("-ff", "--featuresfolder", type=str, default="output_celebA/celebA_features.pickle",
	help="path to fauteres folder  ")
ap.add_argument("-tdf", "--traindatafolder", type=str, default="Dataset/train",
	help="path to Training data folder  ")
ap.add_argument("-tedf", "--testdatafolder", type=str, default="Dataset/test",
	help="path to Testing data folder  ")
ap.add_argument("-mp", "--modelpath", type=str, default="pretrained/classifier.hdf5",
	help=" path for saving the model after training")
ap.add_argument("-bval", "--boolval", type=int, default=0,
	help=" True if you want to create validation data for ANN training")
ap.add_argument("-DA", "--DataAug", type=int, default=0,
	help=" Apply data Augmentation")
args = vars(ap.parse_args())

    
model = load_pretrained_model('ir_101')

if(args["booltraining"]  ):
    embeddings(model,
               datafolder=args["traindatafolder"],
               embeddings_folder=args["embeddingsfolfer"],
               features_folder=args["featuresfolder"],
               Validation=False,
               Data_augmentation=args["DataAug"],
               picperperson=3
               )
if(args["boolval"]):
    embeddings(model,
               datafolder=args["testdatafolder"],
               embeddings_folder=args["val_embeddingsfolfer"],
               features_folder=None,
               Validation=True,
               Data_augmentation=False,
               picperperson=3
               )

if(args["boolSVMtrain"] ):
    SVM_train(args["boolpca"],datafolder=args["embeddingsfolfer"])
if(args["booleval"] ):
    evaluate_models(model,args["thresshold_SVM"] ,args["thresshold_KNN"]   ,args["thresshold_ANN"]   ,args["boolmyknn"]  ,args["boolpca"],args["boolnn"],args["boolSVM"],testfolder=args["testdatafolder"],Embeddings_folder=args["embeddingsfolfer"],features_folder=args["featuresfolder"],model_path=args["modelpath"],boolblur_testset=args["blur_testset"])



