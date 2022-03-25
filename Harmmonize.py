import pywt
import pywt.data
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.stats import linregress
from nilearn import image
import glob
import pandas
from  collections import Counter
import os
from Library.neuroCombat import neuroCombat
"""path = "data/*.gz"
for file in glob.glob(path):
    print (file)
    img = image.load_img(file)"""
atlas_path = "F:/Research/Thesis/download/Atlas_Extractor/difumo128/*.npy"
phenotype = pd.read_excel('F:/Research/Thesis/download/Labels.xls')
phenotype=phenotype.to_numpy()
site_names = np.unique(phenotype[:,10])

def CheckLabel(name):
    for item in phenotype:
        if (item[11]==name):
            classlabel = item[5]
            sex = item[6]
            age = item[7]
            #siteid = np.where(item[10]==site_names)[0][0]
            siteid = item[10]
            return age,sex,classlabel,siteid
    return None,None,None,None
data = np.genfromtxt('testdata.csv', delimiter=",", skip_header=1)
def Get_Upper_Triangle(data):
    x = []
    for i in range(data.shape[0]):
        for j in range(i-1):
            x.append(data[i,j])
    x= np.array(x)
    return x
def Make_Same_Size(X):
    data = []
    for item1 in X:
        for item2 in item1:
            data.append(item2)
    return np.array(data)
def Load_data(dirpath,fromage=1,toage=80,sitename='all',norm="roboust"):
    X=[]
    Y=[]
    Age =[]
    Sex = []
    Site = []
    for file in glob.glob(dirpath):
        basename = os.path.basename(file)
        base1= basename[0:len(basename)-17]

        age,sex,classlabel,siteid = CheckLabel(base1)
        if (age!=None):
            if (sitename=='all'):
                if ( age>=fromage and age<=toage):
                    X1 = np.load(file)
                    X1 = Get_Upper_Triangle(X1)
                    X.append(X1)
                    Age.append(age)
                    Sex.append(sex)
                    Site.append(siteid)
                    if (classlabel=="Autism"):
                        Y.append(1)
                    else:
                        Y.append(0)
            else:
                if ( age >= fromage and age<=toage and siteid==sitename):
                    X1 = np.load(file)
                    X1 = Get_Upper_Triangle(X1)
                    X.append(X1)

                    Age.append(age)
                    Sex.append(sex)
                    Site.append(siteid)
                    if (classlabel=="Autism"):
                        Y.append(1)
                    else:
                        Y.append(0)
        else:
            print(base1)
    X = np.array(X)
    Age = np.array(Age)
    Sex = np.array(Sex)
    Site = np.array(Site)
    print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))
#    X = Drop_Constant_Columns(X)
    print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))
    #X = Drop_Correlated(X)
    print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))
    Y = np.array(Y)

    #if (norm == "stand"):
    #    X = StandardScaler().fit_transform(X)
    #elif(norm == "roboust"):
    #    X = RobustScaler().fit_transform(X)
    #elif(norm == "minmax"):
    #    X = RobustScaler().fit_transform(X)
    batch_data= pd.DataFrame({"batch":(Site),"gender":(Sex),"age":(Age),"disease":(Y)})
    return X,Y,batch_data



data,Y,covars = Load_data(atlas_path)
covars = pd.DataFrame(covars)
categorical_cols = ['gender','age','disease']
batch_col = 'batch'
data_combat = neuroCombat(dat=data,covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
X = np.transpose(data_combat)

frequency = Counter(phenotype[:,10])
print(frequency)
