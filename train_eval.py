from helper_functions import *
import time
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-EMBEDDINGS", "--booltraining", type=bool, default=False,
	help="create face embeddings ")
ap.add_argument("-KNN", "--boolmyknn", type=bool, default=False,
	help="Apply KNN as classifier with face embeddings as data ")
ap.add_argument("-PCA", "--boolpca", type=bool, default=False,
	help="Apply PCA in 512-d face embeddings and then train with SVM ")
ap.add_argument("-ANN", "--boolnn", type=bool, default=True,
	help="Apply pretrained ANN as classifier with face embeddings as data ")
ap.add_argument("-SVM", "--boolSVM", type=bool, default=False,
	help="Apply pretrained SVM as classifier with face embeddings as data ")
ap.add_argument("-T_SVM", "--boolSVMtrain", type=bool, default=False,
	help="Train SVM model in face embeddings  ")
ap.add_argument("-EVAL", "--booleval", type=bool, default=True,
	help="Evaluate models  ")
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
ap.add_argument("-bval", "--boolval", type=bool, default=False,
	help=" True if you want to create validation data for ANN training")
ap.add_argument("-DA", "--DataAug", type=bool, default=False,
	help=" Apply data Augmentation")
args = vars(ap.parse_args())

##ARGUMENTS
booltraining=args["booltraining"]  
boolpca=args["boolpca"]      
boolmyknn=args["boolmyknn"]    
boolnn=args["boolnn"]        
boolSVM=args["boolSVM"]      
boolSVMtrain=args["boolSVMtrain"]  
booleval=args["booleval"]       
th_SVM=args["thresshold_SVM"]       
th_KNN=args["thresshold_KNN"]       
th_ANN=args["thresshold_ANN"]       

booltraining=True
booleval=True
boolmyknn=True

model = load_pretrained_model('ir_101')

if(booltraining):
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

if(boolSVMtrain):
    SVM_train(boolpca,datafolder=args["embeddingsfolfer"])
if(booleval):
    evaluate_models(model,th_SVM,th_KNN ,th_ANN ,boolmyknn,boolpca,boolnn,boolSVM,testfolder=args["testdatafolder"],Embeddings_folder=args["embeddingsfolfer"],features_folder=args["featuresfolder"],model_path=args["modelpath"],boolblur_testset=True)#args["blur_testset"]



