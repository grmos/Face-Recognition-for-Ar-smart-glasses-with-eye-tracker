import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from matplotlib import pyplot
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-td", "--TrainingData", type=str, default="output_celebA/celebA_Embeddings.pickle",
	help=" training data file")
ap.add_argument("-vd", "--ValidationData", type=str, default="output_celebA/celebA_val_Embeddings.pickle",
	help=" validation data file")
ap.add_argument("-e", "--EPOCHS", type=int, default=8000)
ap.add_argument("-lr", "--learning_rate", type=float, default=0.001)
ap.add_argument("-th_ANN", "--thresshold_ANN", type=float, default=0.95,
	help="thresshold score for classification in ANN  ")
ap.add_argument("-ms", "--modelsave", type=int, default=0,
	help=" True if you want to save the model after training")
ap.add_argument("-mp", "--modelpath", type=str, default="pretrained/classifier.hdf5",
	help=" path for saving the model after training")
ap.add_argument("-nnh", "--numofneurons", type=int, default=64,
	help=" number of  neurons in the hidden layer")
args = vars(ap.parse_args())

##Prepare DATA (Train , Validation)
train_data = pickle.loads(open(os.path.abspath(args["TrainingData"]), "rb").read())
val_data = pickle.loads(open(os.path.abspath(args["ValidationData"]), "rb").read())
labels=train_data["names"]+val_data["names"]
trainX=np.array(train_data["embeddings"])
testX=np.array(val_data["embeddings"])
labels=np.array(labels)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
trainY=labels[0:trainX.shape[0]]
testY=labels[trainX.shape[0]:trainX.shape[0]+testX.shape[0]]

##Construct Model archtecture
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(args["numofneurons"],input_shape=(512,),activation="relu"))
model.add((tf.keras.layers.Dense(trainY.shape[1],activation="softmax")))

##Configuration
EPOCHS=args["EPOCHS"]  
lr=args["learning_rate"]  
th_ANN=args["thresshold_ANN"]  
BS=len(trainX)

##Training
opt=tf.keras.optimizers.Adam(lr, decay=lr / EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=EPOCHS,batch_size=BS)
if(args["modelsave"] ):
    model.save(args["modelpath"] )

##Testing
predictions=model.predict(testX)
error,res=0,0
for i in range(len(testX)):
    if(predictions.max(axis=1)[i]>=th_ANN):
        res+=1
        if(testY.argmax(axis=1)[i]!=predictions.argmax(axis=1)[i]):
            error+=1
    elif(testY.argmax(axis=1)[i]==10 and predictions.max(axis=1)[i]<0.98):
        res+=1
print("RESULTS WITH THRESSHOLD:{}/{}".format(res-error,res))
if(res!=0):
    print((1-error/res)*100)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))
pyplot.title('Learning Curves')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.plot(H.history["loss"], label='train')
pyplot.plot(H.history["val_loss"], label='val')
pyplot.legend()
pyplot.show()


