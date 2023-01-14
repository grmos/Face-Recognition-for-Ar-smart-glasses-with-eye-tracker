import os 
from imutils import paths
from PIL import Image  
import PIL 
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ntr", "--numoftrain", type=int, default=12,
	help="number of train images per person.If you  want all examples for training put a big number (5000) ")
ap.add_argument("-nte", "--numoftest", type=int, default=6,
	help="number of test images per person.If you dont want testset put 0 ")
ap.add_argument("-ndf", "--nameofdf", type=str, default="CelebDataProcessed",
	help="name of dataset folder ")
ap.add_argument("-non", "--numofnames", type=int, default=45,
	help="number of perons ")
args = vars(ap.parse_args())

trainfolder=os.path.abspath("Dataset/train")
testfolder=os.path.abspath("Dataset/test")
Datafolder=os.path.abspath("Dataset/"+args["nameofdf"])

if not os.path.exists(trainfolder):
    os.makedirs(trainfolder)
if not os.path.exists(testfolder):
    os.makedirs(testfolder)

imagePaths = list(paths.list_images(Datafolder))
p_name=imagePaths[0].split(os.path.sep)[-2].split('/')[-1]
counter=0
counter_names=1
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    name=name.split('/')[-1]
    nametrainsubfolder=trainfolder+"/"+name
    nametestsubfolder=testfolder+"/"+name
    if(name==p_name):
        counter+=1
    else:
        counter=1
        p_name=name
        counter_names+=1
    if(args["numofnames"]<counter_names):break
    if ( counter<=args["numoftrain"]):
        if not os.path.exists(nametrainsubfolder):
            os.makedirs(nametrainsubfolder)
        picture = Image.open(imagePath)  
        picture = picture.convert('RGB')
        picture.save(nametrainsubfolder+"/"+"{}.jpg".format(counter)) 
    elif(counter>args["numoftrain"] and counter<=args["numoftrain"]+args["numoftest"]):
        if not os.path.exists(nametestsubfolder):
            os.makedirs(nametestsubfolder)
        picture = Image.open(imagePath) 
        picture = picture.convert('RGB')
        picture.save(nametestsubfolder+"/"+"{}.jpg".format(counter-args["numoftrain"]))
    else:
        continue

    
