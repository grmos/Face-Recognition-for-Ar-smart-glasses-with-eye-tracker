import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from Feature_Extractor.helper_functions import  *
from face_alignment import align
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from collections import Counter
from imutils import paths
import os
import cv2 
import pickle



def SVM_train(boolpca,datafolder,recognizer_folder="output/recognizer.pickle",labels_folder="output/le.pickle"):
    data = pickle.loads(open(os.path.abspath(datafolder), "rb").read())
    le = LabelEncoder()
    labels=le.fit_transform(data["names"])
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    if(boolpca):
        dataforpca=np.array(data["embeddings"])
        dataforpca,pca,scaler=preprocess_pca(dataforpca,Standardization=False)
        recognizer.fit(dataforpca.tolist(),labels)
    else:
        recognizer.fit(data["embeddings"],labels)
    f = open(os.path.abspath(recognizer_folder), "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    f = open(os.path.abspath(labels_folder), "wb")
    f.write(pickle.dumps(le))
    f.close()

def preprocess_pca(dataforpca,info=0.99,Standardization=True):
    if(Standardization):
        scaler = StandardScaler()
        scaler.fit(dataforpca)
        dataforpca = scaler.transform(dataforpca)
    else:
        scaler=None
    pca = PCA(info)
    pca.fit(dataforpca)
    dataforpca = pca.transform(dataforpca)
    print(dataforpca.shape)
    return dataforpca,pca,scaler

def test_pca(feature,pca,Standardization=True,scaler=None):
    feature=np.array(feature)
    if(Standardization):
        feature = scaler.transform(feature)
    feature = pca.transform(feature)
    feature=feature.tolist()
    return feature

def myknn(features,feature,labels,k=1):
    results=[]
    scores=[]
    similarity_scores = torch.cat(features)@(feature).T
    ss=similarity_scores[:,-1].detach().numpy()
    ss=ss.tolist()
    for i in range(k):
        result = np. where(ss == np.amax(ss))
        results.append(result[0][0])
        score=ss[result[0][0]]
        scores.append(score)
        ss[result[0][0]]=-1
    if(k==1):
        return labels[results[0]],scores[0]
    p_names=[]
    for i in range(k):
        p_name=labels[results[i]]
        p_names.append(p_name)
    p_names=Counter(p_names)
    p_name=p_names.most_common()[0][0]
    score=scores[0]
    return p_name,score

def embeddings(model,
               datafolder,
               embeddings_folder,
               features_folder,
               Validation=False,
               Data_augmentation=False,
               blur=0, # {0,1,2,3) 0 => no blur, 3 => max blur
               picperperson=2,# for Data_augmentation 
               blurscale=5,
               ):
    if(Data_augmentation):
        start=1
        end=1+blurscale*picperperson
        augments_1 = ['brightness',  'contrast', 'flip_h']
        augments_2 = ['brightness', 'flip_h']
        augments=[augments_1,augments_2]
        aug = Augment()
    if(blur):
        start=1+blurscale*blur
        end=start+1
    imagePaths = list(paths.list_images(os.path.abspath(datafolder)))
    knownEmbeddings = []
    knownNames = []
    for (i, imagePath) in enumerate(imagePaths):
        print("{}/{}".format(i,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        name=name.split('/')[-1]
        image = cv2.imread(imagePath)
        if(Data_augmentation or blur ):
            for j in range(start,end,blurscale):
                if(j>1):
                    b_image = cv2.blur(image,(j+1,j+1))
                    aligned_rgb_img,box = align.get_aligned_face(b_image)
                else:
                    aligned_rgb_img,box = align.get_aligned_face(image)
                if(aligned_rgb_img==None):
                    print(imagePath)
                    continue;
                bgr_tensor_input = to_input(aligned_rgb_img)
                feature, norms = model(bgr_tensor_input)
                if(len(feature)>0):
                    knownNames.append(name)
                    knownEmbeddings.append(feature.detach().numpy().flatten())
            for item in augments:
                augimage = aug.augment(images=image,
                    operations=item
                    )
                augimage=augimage.numpy()
                aligned_rgb_img,box = align.get_aligned_face(augimage)
                if(aligned_rgb_img==None):
                    print(imagePath)
                    continue;
                bgr_tensor_input = to_input(aligned_rgb_img)
                feature, norms = model(bgr_tensor_input)
                if(len(feature)>0):
                    knownNames.append(name)
                    knownEmbeddings.append(feature.detach().numpy().flatten())   
        else:
            aligned_rgb_img,box = align.get_aligned_face(image)
            if(aligned_rgb_img==None):
                print(imagePath)
                continue;
            bgr_tensor_input = to_input(aligned_rgb_img)
            feature, norms = model(bgr_tensor_input)
            if(len(feature)>0):
                knownNames.append(name)
                knownEmbeddings.append(feature.detach().numpy().flatten())

    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(os.path.abspath(embeddings_folder), "wb")
    f.write(pickle.dumps(data))
    f.close()
    features=[]
    if(Validation==False):
        for embedding in knownEmbeddings:
            features.append(torch.Tensor([embedding]))
        data2 = {"features": features, "names": knownNames}
        f = open(os.path.abspath(features_folder), "wb")
        f.write(pickle.dumps(data2))
        f.close()

    


def evaluate_models(model,
                    th_SVM,
                    th_KNN,
                    th_ANN,
                    boolmyknn,
                    boolpca,
                    boolnn,
                    boolSVM,
                    testfolder,
                    Embeddings_folder,
                    features_folder,
                    model_path,
                    boolblur_testset=False,
                    Recognizer_folder="output/recognizer.pickle",
                    labels_folder="output/le.pickle"):
    if(boolnn):
        data = pickle.loads(open(os.path.abspath(Embeddings_folder), "rb").read())
        lab = LabelEncoder()
        lab.fit_transform(data["names"])
        model_loaded = load_model(model_path)
    if(boolmyknn):
        datawithfeatures = pickle.loads(open(os.path.abspath(features_folder), "rb").read())
        labels=datawithfeatures["names"]
        features=datawithfeatures["features"]
    if(boolSVM):
        recognizer = pickle.loads(open(os.path.abspath(Recognizer_folder), "rb").read())
        le = pickle.loads(open(os.path.abspath(labels_folder), "rb").read())
    
    imagePaths = list(paths.list_images(os.path.abspath(testfolder)))
    knownEmbeddings = []
    knownNames = []
    res,res2,res3=0,0,0
    tot,tot2,tot3=0,0,0
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        name=name.split('/')[-1]
        imagePath = cv2.imread(imagePath)
        if(boolblur_testset):
            imagePath = cv2.blur(imagePath,(22,22))
        aligned_rgb_img,box = align.get_aligned_face(imagePath)
        if(aligned_rgb_img==None):
            continue;
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, norms = model(bgr_tensor_input)
        if(boolmyknn):
            result,score=myknn(features,feature,labels,k=1)  
        feature=[feature.detach().numpy().flatten()]
        if(boolpca):
            feature=test_pca(feature,pca,Standardization=False,scaler=scaler)
        if(boolnn):
            nninput=np.array(feature)
            predictions=model_loaded.predict(nninput)
            index=predictions.argmax(axis=1)
            nnscore=predictions.max(axis=1)
        if(boolSVM):
            preds = recognizer.predict_proba(feature)[0]
            j = np.argmax(preds)
            proba = preds[j]
            print("_________")
            print("SVM")
            print(proba)
            if(proba>th_SVM):
                p_name = le.classes_[j]
            else:
                p_name = "Unknown"
            if(p_name!="Unknown" or (p_name== "Unknown" and name=="random")):
                tot=tot+1
            else:
                print("Unknown")
            if(p_name==name or (p_name== "Unknown" and name=="random") ):
                res=res+1	
                print("true {}".format(p_name))
            else:
                print("false {}/{}".format(p_name,name))
            print("_________")
        if(boolmyknn):
            print("_________")
            print("K-NN")
            print(score)
            p_name=result
            if( name=="random" and score<th_KNN ):
                res2+=1
                tot2+=1
            if(score>=th_KNN):
                if(p_name==name):
                    res2+=1
                    print("True {}".format(p_name))
                else:
                    print("false {}/{}".format(p_name,name))
                tot2+=1
                print("_________")
        if(boolnn):
            p_name = lab.classes_[index[0]]
            score=nnscore[0]
            print("_________")
            print("ANN")
            print(score)
            if( name=="random" and score<th_ANN ):
                res3+=1
                tot3+=1
            if(score>= th_ANN ):
                if(p_name==name):
                    res3+=1
                    print("True {}".format(p_name))
                else:
                    print("false {}/{}".format(p_name,name))
                tot3+=1
                print("_________")
        
    if(boolSVM):
        print("_________")
        print("   SVM   ")
        print(100*res/tot)
        print("{}/{}".format(res,tot))
        print("_________")
    if(boolmyknn):
        print("_________")
        print("  K-NN   ")
        print(100*res2/tot2)
        print("{}/{}".format(res2,tot2))
        print("_________")
    if(boolnn):
        print("_________")
        print("   ANN   ")
        print(100*res3/tot3)
        print("{}/{}".format(res3,tot3))
        print("_________")

class Augment(object):

    def augment(self, images, operations):
        augmented = []
        for i in range(len(images)):
            if 'brightness' in operations:
                aug = self.random_brightness(images,
                                             max_delta=0.5)
            if 'contrast' in operations:
                aug = self.random_contrast(image=aug,
                                           lower=0.2,
                                           upper=1.8)
            if 'flip_h' in operations:
                aug = self.random_flip_h(image=aug)

        return aug

    def random_contrast(self, image, lower, upper):
        return tf.image.random_contrast(image=image,
                                        lower=lower,
                                        upper=upper)

    def random_brightness(self, image, max_delta=0.2):
        return tf.image.random_brightness(image=image,
                                          max_delta=max_delta)

    def random_flip_h(self, image):
        return tf.image.random_flip_left_right(image=image)
