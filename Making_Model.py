from scipy.stats import norm
import glob
from sklearn.preprocessing import RobustScaler,StandardScaler,QuantileTransformer,MinMaxScaler
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import os
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Library.neuroCombat import neuroCombat
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
import random
from scipy.stats import loguniform


from collections import Counter
#************************Load initial Files**************************
phenotype = pd.read_excel('Labels.xls')
phenotype=phenotype.to_numpy()

#****************************Classifier******************************
class Pure_Classifier():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
        self.kfold = 10
        self.classifier = []
        pass
    def logit_p1value(self,model, x):
        p1 = model.predict_proba(x)
        n1 = len(p1)
        m1 = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
        answ = np.zeros((m1, m1))
        for i in range(n1):
            answ = answ + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p1[i, 1] * p1[i, 0]
        vcov = np.linalg.inv(np.matrix(answ))
        se = np.sqrt(np.diag(vcov))
        t1 = coefs / se
        p1 = (1 - norm.cdf(abs(t1))) * 2
        return p1

    def GenerateAllClassifiers(self):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=5, C=2)
        ridge = RidgeClassifier(alpha=0.5)
        sgdclassifier = SGDClassifier()
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        bagging = BaggingClassifier(dt,max_samples=0.5,max_features=0.5)
        voting = VotingClassifier(estimators=[('SVM', svclassifier), ('DT', dt), ('LR', lr)], voting='hard')
        #histogramgradient = HistGradientBoostingClassifier(min_samples_leaf=1,   max_depth = 2,  learning_rate = 1,max_iter = 1)
        myclassifier = [knn,nivebase,dt,lr,svclassifier,randomforest,voting,ridge,mlp,ada,bagging,sgdclassifier]
        return myclassifier

    def GenereateClassifier(self,outclassifier=None):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=5, C=2)
        ridge = RidgeClassifier(alpha=0.5)
        sgdclassifier = SGDClassifier()
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        bagging = BaggingClassifier(dt,max_samples=0.5,max_features=0.5)
        #voting = VotingClassifier(estimators=[('SVM', svclassifier), ('DT', dt), ('LR', lr)], voting='hard')
        #histogramgradient = HistGradientBoostingClassifier(min_samples_leaf=1,   max_depth = 2,  learning_rate = 1,max_iter = 1)
        self.classifier = [knn,nivebase,dt,lr,svclassifier,randomforest,ridge,mlp,ada,bagging,sgdclassifier]
        #self.classifier = [knn]
        if (outclassifier!=None):
            self.classifier = [outclassifier]

        #self.classifier = [knn,nivebase,dt,lr,svclassifier,randomforest,voting,ridge]

    def Get_Classifier_Names(self):
        classifier_names = []
        for classifier in self.classifier:
            classifier_names.append(classifier.__class__.__name__)
        return classifier_names
    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)
    def __Cross_Validation(self, X, y, classifier, kf):
        Data = X
        evlP = [[0 for x in range(7)] for YY in range(self.kfold)]
        k = 0
        misindex = []
        for train_index, test_index in kf.split(Data,y):
            classifier.fit(Data[train_index], y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)


            evlP[k][0] = (precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            evlP[k][6] = specificity

            index_counter = 0
            for index in test_index:
                if (y_pred[index_counter]!=y[index]):
                    misindex.append(index)
                index_counter+=1

            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return np.array(average[0]), misindex

    def DoCrossValidation(self, X,Y,resultfilename,savetofile=True):
        run_times = 1
        overall_performance = []
        misindex = []
        for runs in range(run_times):
            kf = StratifiedKFold(n_splits=self.kfold)
            classifier_names = []
            total_performance = []
            for classifier in self.classifier:
                classifier_names.append(classifier.__class__.__name__)
                performance,misindex = self.__Cross_Validation(X,Y,classifier,kf)
                total_performance.append(performance[0])

            total_performance = np.array(total_performance)
            overall_performance.append(total_performance)

        overall_performance = np.array(overall_performance)
        overal_mean = np.mean(overall_performance,axis=0)
        overal_variance = np.std(overall_performance,axis=0)
        if (savetofile==True):
            df1 = pd.DataFrame(overal_mean,index = classifier_names,columns=self.evaluationName)
            df2 = pd.DataFrame(overal_variance,index = classifier_names,columns=self.evaluationName)
            with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
                df1.to_excel(writer, sheet_name='Mean')
                df2.to_excel(writer, sheet_name='Variance')
        else:
            return overal_mean,overal_variance,misindex
#****************************PreProcess*****************************
class Preprocess():
    def __init__(self):
        pass
    def CheckLabel(self,name):
        for item in phenotype:
            if (item[11]==name):
                classlabel = item[5]
                sex = item[6]
                age = item[7]
                siteid = item[10]
                return age,sex,classlabel,siteid
        return None,None,None,None
    def Get_Upper_Triangle(self,data):
        x = []
        for i in range(data.shape[0]):
            for j in range(i):
                x.append(data[i,j])
        x= np.array(x)
        return x
    def Drop_Constant_Columns(self,X):
        dataframe = pd.DataFrame(X)
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == 1:
                dataframe.drop(column,inplace=True,axis=1)
        return dataframe
    def Vectorize_Matrix(self,X):
        data = []
        for item1 in X:
            for item2 in item1:
                data.append(item2)
        return np.array(data)
    def Drop_Correlated(self,X):
        df = pd.DataFrame(X)

        corr_matrix  = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

        df.drop(to_drop, axis=1, inplace=True)
        return df.to_numpy()

    def Read_1D(self,filename):

        file = open(filename, "r")
        file.readline()
        lines = file.readlines()

        roi_signal = []
        for item in lines:
            row = item.split("\t")

            row = np.asarray(row)

            row = row.astype(float)

            roi_signal.append(row)
        roi_signal = np.array(roi_signal)

        file.close()
        return roi_signal
    def Load_Atlas_Data_Abide(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):
        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            _index = basename.find("_rois")
            filesitename= basename[0:_index]
            #print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        time_series = self.Read_1D(file)
                        correlation_measure = ConnectivityMeasure(kind='correlation')
                        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)

                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        time_series = self.Read_1D(file)
                        correlation_measure = ConnectivityMeasure(kind='correlation')
                        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, batch_data
    def Load_Atlas_Data_NiLearn(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):
        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            _index = basename.find("_func_pre")
            filesitename= basename[0:_index]
            #print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        correlation_matrix =np.load(file)
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)

                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        correlation_matrix =np.load(file)
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, batch_data

    def Load_Data(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):

        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []

        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            filesitename= basename[0:len(basename)-17]
            print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        X1 = np.load(file)
                        X1 = self.Get_Upper_Triangle(X1)
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
                        X1 = self.Get_Upper_Triangle(X1)
                        X.append(X1)
                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age), "disease": (Y)})
        return X, Y, batch_data
    def BinAge(self,age):
        if (age >= 5 and age<=10):
            return "5-10"
        elif(age>10 and age<=20):
            return "10-20"
        elif(age>20 and age<=40):
            return "20-40"
        else:
            return ">40"
    def LoadSpecificAtlas(self,specific_atlas_dir,fromage=1,toage=70,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):
        X = []
        Y = []
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(specific_atlas_dir):
            print(file)
            basename = os.path.basename(file)
            _index = basename.find("_func_pre")
            filesitename = basename[0:_index]
            # print(filesitename)
            age, sex, classlabel, siteid = self.CheckLabel(filesitename)
            if (age != None):
                if (sitename == 'all'):
                    if (age >= fromage and age <= toage):
                        correlation_matrix = np.load(file)
                        X.append(correlation_matrix)
                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel == "Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if (age >= fromage and age <= toage and siteid == sitename):
                        correlation_matrix = np.load(file)
                        X.append(correlation_matrix)
                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel == "Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print("Main Data Size : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropconstant == True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated == True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould == True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)


        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, Sex

#****************************Make Models*****************************

class AutismClassifiers():
    def Classify_Atlas(self,resultfilename,fromage,toage):
        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier.kfold = 10
        classifier_names = classifier.Get_Classifier_Names()

        atlas_names = []
        data_dir = "AtlasExtracted/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        data_dir_nilearn = "AtlasExtracted/NilearnNew/*"
        for index,folder in enumerate(glob.glob(data_dir_nilearn)):
            atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))
        specificity =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0
        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"
            X, Y, df = preproc.Load_Atlas_Data_Abide(files, fromage, toage, applythresould=False)
            mean, varivance,misindx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            specificity[:,index] = mean[:,6]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        for index, folder in enumerate(glob.glob(data_dir_nilearn)):
            print(folder)
            files = folder + "/*.npy"
            X, Y, df = preproc.Load_Atlas_Data_NiLearn(files, fromage, toage,applythresould=False)
            mean, varivance,misidx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:, index+lastindex] = mean[:, 2]
            precision[:, index+lastindex] = mean[:, 0]
            recall[:, index+lastindex] = mean[:, 3]
            auc[:, index+lastindex] = mean[:, 5]
            specificity[:, index+lastindex] = mean[:, 6]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)
        df5 = pd.DataFrame(specificity, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Final_NonCombat{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
            df5.to_excel(writer, sheet_name='Specificity')
    def Classify_Atlas_With_Combat(self,resultfilename,fromage,toage):
        #This function classify abide autism fmri dataset with 13 classifier methods and save the perfomance of each classifier
        # in the excel file located in the result folder, so before running this code, it is need to create a folder with result name
        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier.kfold = 10
        classifier_names = classifier.Get_Classifier_Names()

        atlas_names = []
        data_dir = "AtlasExtracted/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        data_dir_nilearn = "AtlasExtracted/NilearnNew/*"
        for index,folder in enumerate(glob.glob(data_dir_nilearn)):
            atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))
        specificity =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0

        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"

            X, Y, covars = preproc.Load_Atlas_Data_Abide(files, fromage,toage,applythresould=False)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            #data_combat = neuroCombat(dat=X, covars=covars, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
            mean, varivance,misidx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            specificity[:,index] = mean[:,6]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        for index, folder in enumerate(glob.glob(data_dir_nilearn)):
            print(folder)
            files = folder + "/*.npy"
            X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, fromage,toage,applythresould=False)
            X = np.transpose(X)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            #data_combat = neuroCombat(dat=X, covars=covars, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
            mean, varivance,misidx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:, index+lastindex] = mean[:, 2]
            precision[:, index+lastindex] = mean[:, 0]
            recall[:, index+lastindex] = mean[:, 3]
            auc[:, index+lastindex] = mean[:, 5]
            specificity[:, index+lastindex] = mean[:, 6]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)
        df5 = pd.DataFrame(specificity, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Final_Combat{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
            df5.to_excel(writer, sheet_name='Specificity')

    def Classify_Atlas_Abide_For_Each_Site(self):
        preproc =Preprocess()
        site_names = np.unique(phenotype[:,10])
        #X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        data_dir = "Atlas_Extractor/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            accuracy = np.zeros((len(classifier_names), len(site_names)))
            precision = np.zeros((len(classifier_names), len(site_names)))
            recall = np.zeros((len(classifier_names), len(site_names)))
            auc = np.zeros((len(classifier_names), len(site_names)))
            specificity = np.zeros((len(classifier_names), len(site_names)))

            for index,site_name in enumerate(site_names):
                files = folder + "/*.1d"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_Abide(files, 1, 70,sitename=site_name, applythresould=False)
                    mean, varivance,misindex = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[:, index] = mean[:, 2]
                    precision[:, index] = mean[:, 0]
                    recall[:, index] = mean[:, 3]
                    auc[:, index] = mean[:, 5]
                    specificity[:, index] = mean[:, 6]
                except:
                    pass

                lastindex = index


            df1 = pd.DataFrame(accuracy, index=classifier_names, columns=site_names)
            df2 = pd.DataFrame(precision, index=classifier_names, columns=site_names)
            df3 = pd.DataFrame(recall, index=classifier_names, columns=site_names)
            df4 = pd.DataFrame(auc, index=classifier_names, columns=site_names)
            df5 = pd.DataFrame(specificity, index=classifier_names, columns=site_names)

            filename = os.path.split(folder)[1]
            with pd.ExcelWriter('result/Abide_Site{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')
                df5.to_excel(writer, sheet_name='Specificity')
    def Classify_Atlas_Nilearn_For_Each_Site(self):
        preproc =Preprocess()
        site_names = np.unique(phenotype[:,10])
        #X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        data_dir = "Atlas_Extractor/Nilearn/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            accuracy = np.zeros((len(classifier_names), len(site_names)))
            precision = np.zeros((len(classifier_names), len(site_names)))
            recall = np.zeros((len(classifier_names), len(site_names)))
            auc = np.zeros((len(classifier_names), len(site_names)))
            specificity = np.zeros((len(classifier_names), len(site_names)))

            for index,site_name in enumerate(site_names):
                files = folder + "/*.npy"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, 1, 70,sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[:, index] = mean[:, 2]
                    precision[:, index] = mean[:, 0]
                    recall[:, index] = mean[:, 3]
                    auc[:, index] = mean[:, 5]
                    specificity[:, index] = mean[:, 6]
                except:
                    pass

            df1 = pd.DataFrame(accuracy, index=classifier_names, columns=site_names)
            df2 = pd.DataFrame(precision, index=classifier_names, columns=site_names)
            df3 = pd.DataFrame(recall, index=classifier_names, columns=site_names)
            df4 = pd.DataFrame(auc, index=classifier_names, columns=site_names)
            df5 = pd.DataFrame(specificity, index=classifier_names, columns=site_names)

            filename = os.path.split(folder)[1]
            with pd.ExcelWriter('result/Nilearn_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')
                df5.to_excel(writer, sheet_name='Auc')
    def Classify_Atlas_For_Each_Site(self):
        preproc = Preprocess()
        site_names = np.unique(phenotype[:, 10])
        # X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifiers = classifier.classifier
        for cla in classifiers:
            classifier.classifier = [cla]
            classifier_names = classifier.Get_Classifier_Names()

            abide_atlas = ['rois_aal','rois_cc200','rois_cc400','rois_dosenbach160','rois_ez','rois_ho','rois_tt']
            nilearn_atlas = ['croddock','difumo128','difumo256','difumo512','hard','msdl','rsmith70']

            atlas_all = ['rois_aal','rois_cc200','rois_cc400','rois_dosenbach160','rois_ez','rois_ho','rois_tt','croddock','difumo128','difumo256','difumo512','hard','msdl','rsmith70']

            accuracy = np.zeros((len(atlas_all), len(site_names)))
            precision = np.zeros((len(atlas_all), len(site_names)))
            recall = np.zeros((len(atlas_all), len(site_names)))
            auc = np.zeros((len(atlas_all), len(site_names)))
            specificity = np.zeros((len(atlas_all), len(site_names)))

            data_dir = "AtlasExtracted/Abide/"
            lastindex = 0
            for i, filter_name in enumerate(abide_atlas):
                for j, site_name in enumerate(site_names):
                    files = data_dir + filter_name + "/*.1D"
                    try:
                        X, Y, covars = preproc.Load_Atlas_Data_Abide(files, 1, 70, sitename=site_name, applythresould=False)
                        mean, varivance,misidx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                        accuracy[i, j] = mean[:, 2]
                        precision[i, j] = mean[:, 0]
                        recall[i, j] = mean[:, 3]
                        auc[i, j] = mean[:, 5]
                        specificity[i, j] = mean[:, 6]

                    except:
                        pass

                df1 = pd.DataFrame(accuracy, index=atlas_all, columns=site_names)
                df2 = pd.DataFrame(precision, index=atlas_all, columns=site_names)
                df3 = pd.DataFrame(recall, index=atlas_all, columns=site_names)
                df4 = pd.DataFrame(auc, index=atlas_all, columns=site_names)
                df5 = pd.DataFrame(specificity, index=atlas_all, columns=site_names)

                filename = classifier_names[0]
                with pd.ExcelWriter('result/Cross_Atlas_{}.xls'.format(filename)) as writer:
                    df1.to_excel(writer, sheet_name='Accuracy')
                    df2.to_excel(writer, sheet_name='Precision')
                    df3.to_excel(writer, sheet_name='Recall')
                    df4.to_excel(writer, sheet_name='Auc')
                    df5.to_excel(writer, sheet_name='Specificity')

                lastindex=i

            data_dir = "AtlasExtracted/Nilearn/"
            for i, filter_name in enumerate(nilearn_atlas):
                for j, site_name in enumerate(site_names):
                    files = data_dir + filter_name + "/*.npy"
                    try:
                        X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, 1, 100, sitename=site_name, applythresould=False)
                        mean, varivance,misidx = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                        accuracy[i+lastindex, j] = mean[:, 2]
                        precision[i+lastindex, j] = mean[:, 0]
                        recall[i+lastindex, j] = mean[:, 3]
                        auc[i+lastindex, j] = mean[:, 5]
                        specificity[i+lastindex, j] = mean[:, 6]
                    except:
                        pass

                df1 = pd.DataFrame(accuracy, index=atlas_all, columns=site_names)
                df2 = pd.DataFrame(precision, index=atlas_all, columns=site_names)
                df3 = pd.DataFrame(recall, index=atlas_all, columns=site_names)
                df4 = pd.DataFrame(auc, index=atlas_all, columns=site_names)
                df5 = pd.DataFrame(specificity, index=atlas_all, columns=site_names)
                filename = classifier_names[0]

                with pd.ExcelWriter('result/Cross_Atlas_{}.xls'.format(filename)) as writer:
                    df1.to_excel(writer, sheet_name='Accuracy')
                    df2.to_excel(writer, sheet_name='Precision')
                    df3.to_excel(writer, sheet_name='Recall')
                    df4.to_excel(writer, sheet_name='Auc')
                    df5.to_excel(writer, sheet_name='Specificity')

class ConnectivityMap():
    def AppyThresould(self, coefs, thresould, size):
        from scipy import stats
        import math

        weights = np.zeros((size, size))
        pvalues = np.zeros((size, size, coefs.shape[0]))

        k = 0
        for i in range(size):
            for j in range(i):
                pvalues[i,j,:] = coefs[:,k]
                pvalues[j,i,:] = coefs[:,k]
                value = np.abs(np.mean(coefs[:,k]))

                weights[j, i] = value
                weights[i, j] = value
                #print("{},{}:{}".format(i, j, k))
                k += 1
        pv = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if (i!=j):
                    temp = stats.ttest_ind(pvalues[i,0,:] , pvalues[i,j,:])
                    pv[i,j] = temp[1]
                    if (pv[i,j]>0.05):
                        #print ("{},{}".format(i,j))
                        weights[i,j]=0

        return weights

    def Vectorize_Matrix(self,X):
        data = []
        for i in range(X.shape[0]):
            for j in range(i):
                data.append(X[i,j])
        return np.array(data,dtype=np.bool)

    def SaveWeights(self,heatmap):
        difumo = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=3, data_dir="Atlas/")

        labels = []
        coordinates = []
        for item in difumo.labels:
            coordinates.append([item[4], item[5], item[6]])

            labels.append(str(item[1][1:]).replace("'", ""))

        df = pd.DataFrame(heatmap, index=labels, columns=labels)
        df.to_excel("Result/difomo128_main.xls")

        df = pd.DataFrame(coordinates, index=labels)
        df.to_excel("Result/difumo128_coords.xls")

        # import statsmodels.api as sd
        # sd_model = sd.Logit(Y, sm.add_constant(x)).fit(disp=0)
        # print(sd_model.pvalues)
        # sd_model.summary()
    def Classify(self,rowVector,usingCombat=True):
        preproc = Preprocess()
        atlas_name = "AtlasExtracted/NilearnNew/difumo128/*"
        X, Y, covars = preproc.Load_Atlas_Data_NiLearn(atlas_name, 1, 70, applythresould=False)
        categorical_cols = ['gender', 'age']
        batch_col = 'batch'
        if (usingCombat==True):
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
        lr = LogisticRegression(C= 0.000339, penalty= 'l2', solver= 'lbfgs')
        rndlist  = [0,2,7,10,11,14,17,20,21,22,24,26,27,29,35,38,39,41,42,49,50,51,54,56,59,71,72,78,80,83,93,97,104,108,115,118,120,123,124,126,127,130,131,139,142,144,146,150,153,159]
        accuracys = []
        precisions = []
        f1Scores = []
        recalls = []
        specificitys = []

        for item in rndlist:
            X_train, X_test, y_train, y_test = train_test_split(X[:,rowVector], Y, test_size = 0.1, random_state = item)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)

            accuracys.append(accuracy_score(y_test,y_pred))
            precisions.append(precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            f1Scores.append (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            recalls.append (recall_score(y_test, y_pred, average="weighted"))
            specificitys.append(specificity)
        print ("Accuracy {}".format(np.mean(accuracys)))
        print ("precisions {}".format(np.mean(precisions)))
        print ("f1Scores {}".format(np.mean(f1Scores)))
        print ("recalls {}".format(np.mean(recalls)))
        print ("specificitys {}".format(np.mean(specificitys)))

    def Get_FeatureImportance_Difumo(self,usingCombat =False, fromage=1, toage=70):
        preproc = Preprocess()
        atlas_name = "AtlasExtracted/NilearnNew/difumo128/*"
        lr = LogisticRegression(random_state=0)
        X, Y, covars = preproc.Load_Atlas_Data_NiLearn(atlas_name, fromage, toage, applythresould=False)
        categorical_cols = ['gender', 'age']
        batch_col = 'batch'
        if (usingCombat==True):
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)


        coefs = []
        space = dict()
        space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
        space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        space['C'] = loguniform(1e-5, 100)
        lr = LogisticRegression(C= 0.000339, penalty= 'l2', solver= 'lbfgs')

        rndlist  = [0,2,7,10,11,14,17,20,21,22,24,26,27,29,35,38,39,41,42,49,50,51,54,56,59,71,72,78,80,83,93,97,104,108,115,118,120,123,124,126,127,130,131,139,142,144,146,150,153,159]
        accuracys = []
        precisions = []
        f1Scores = []
        recalls = []
        specificitys = []

        for item in rndlist:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = item)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)

            accuracys.append(accuracy_score(y_test,y_pred))
            precisions.append(precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            f1Scores.append (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            recalls.append (recall_score(y_test, y_pred, average="weighted"))
            specificitys.append(specificity)
            coefs.append(lr.coef_[0])

        coefs = np.array(coefs)
        weights = self.AppyThresould(coefs, 0, 128)

        print ("Accuracy {}".format(np.mean(accuracys)))
        print ("precisions {}".format(np.mean(precisions)))
        print ("f1Scores {}".format(np.mean(f1Scores)))
        print ("recalls {}".format(np.mean(recalls)))
        print ("specificitys {}".format(np.mean(specificitys)))

        return weights

    def ApplyWeightMask(self,weights , connectivity):
        result_index = np.where(weights>0.05)
        temp = connectivity
        temp[result_index]=0
        temp = weights*connectivity
        return temp
    def DropUnCorrelated(self,corr_matrix):
        import numpy as np
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features
    def KeepImportantFeature(self,correlation_Matrix ,thresould):
        correlation_Matrix_Copy = np.abs(correlation_Matrix)
        for i in range(correlation_Matrix.shape[0]):
            for j in range(correlation_Matrix.shape[1]):
                if (correlation_Matrix_Copy[i,j]<thresould):
                    correlation_Matrix_Copy[i,j]=0
                else:
                    correlation_Matrix_Copy[i, j] = 1
        rowVector = self.Vectorize_Matrix(correlation_Matrix_Copy)
        return rowVector
    def FilterHeatMap(self,correlation_Matrix ,thresould):
        correlation_Matrix_Copy = np.abs(correlation_Matrix)
        keeped_indexes = []
        for i in range(correlation_Matrix.shape[0]):
            keep_index = np.where(correlation_Matrix_Copy[i,:]>thresould)[0]
            if (len(keep_index)>0):
                keeped_indexes.append(i)
        filteredHeatMap = correlation_Matrix[keeped_indexes,:]
        filteredHeatMap = filteredHeatMap[:,keeped_indexes]
        index = np.where(np.abs(filteredHeatMap)<thresould)
        filteredHeatMap[index]=0
        return filteredHeatMap , keeped_indexes

    def Connectivity_All(self):
        preprocess = Preprocess()
        X,Y,Sex = preprocess.LoadSpecificAtlas("AtlasExtracted/NilearnNew/difumo128/*")
        difumo = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=3, data_dir="Atlas/")
        connectivity_all_mean = np.mean(X,0)
        usingCombat =True
        fileName = ""
        if (usingCombat==True):
            fileName = "Combat"

        weights = self.Get_FeatureImportance_Difumo(usingCombat)
        connectivity_all_mean = self.ApplyWeightMask(weights,connectivity_all_mean)

        labels = []
        yeo_labels =[]
        for item in difumo.labels:
            temp = item[1]
            labels.append( str(temp).replace("'","")[1:])
            temp = item[3]
            yeo_labels.append(str(temp).replace("'","")[1:])


        rowVector = self.KeepImportantFeature(connectivity_all_mean,0.001)
        count = np.where(rowVector==1)
        print("Count = {}" .format(len(count[0])))
        self.Classify(rowVector)
        filterHeatMap, keepIndexes = self.FilterHeatMap(connectivity_all_mean,0.001)

        newLabels = np.array(labels)[keepIndexes]
        newLabelsYeo = np.array(yeo_labels)[keepIndexes]

        mask = np.triu(np.ones_like(filterHeatMap))
        sns.set(font_scale=0.2)

        sns.heatmap(filterHeatMap,xticklabels=newLabels,yticklabels=newLabels,mask=mask)
        plt.savefig("result/heatmapAll_{}.svg".format(fileName),bbox_inches='tight')

        df = pd.DataFrame(filterHeatMap,index=newLabelsYeo,columns=newLabels)
        df.to_excel("result/heatmapData_{}.xls".format(fileName))

        coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=difumo.maps)
        plotting.plot_connectome(connectivity_all_mean, coordinates, edge_threshold="98%", title="",colorbar=True)
        plt.savefig("result/connectivity_{}.png".format(fileName))

        view = plotting.view_connectome(connectivity_all_mean, coordinates, edge_threshold='98%')
        view.open_in_browser()

    def Connectivity_Control(self):
        preprocess = Preprocess()
        X,Y,Age= preprocess.LoadSpecificAtlas("AtlasExtracted/NilearnNew/difumo128/*")
        difumo = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=3, data_dir="Atlas/")
        healthIndex = np.where(Y==0)[0]
        autismIndex = np.where(Y==1)[0]
        connectivity_Health =np.mean(X[healthIndex],0)
        connectivity_Autism =np.mean(X[autismIndex],0)


        usingCombat =True
        fileName = ""
        if (usingCombat==True):
            fileName = "Combat"

        weights = self.Get_FeatureImportance_Difumo(usingCombat)

        connectivity_Health = self.ApplyWeightMask(weights,connectivity_Health)
        connectivity_Autism = self.ApplyWeightMask(weights,connectivity_Autism)

        labels = []
        yeo_labels =[]
        for item in difumo.labels:
            temp = item[1]
            labels.append( str(temp).replace("'","")[1:])
            temp = item[3]
            yeo_labels.append(str(temp).replace("'","")[1:])


        keepHealth = self.KeepImportantFeature(connectivity_Health,0.001)
        count  = np.where(keepHealth==1)[0]
        print ("Health keep features = {}".format(len(count)))

        filterHealth, keepIndexesHealth = self.FilterHeatMap(connectivity_Health, 0.001)

        newLabelsHealth = np.array(labels)[keepIndexesHealth]
        newLabelsYeoHealth = np.array(yeo_labels)[keepIndexesHealth]

        df = pd.DataFrame(filterHealth,index=newLabelsYeoHealth,columns=newLabelsHealth)
        df.to_excel("result/heatmapDataHealth_{}.xls".format(fileName))


        keepAutism = self.KeepImportantFeature(connectivity_Autism,0.001)
        count  = np.where(keepAutism==1)[0]
        print ("Autism keep features = {}".format(len(count)))



        filterAutism, keepIndexesAutism = self.FilterHeatMap(connectivity_Autism,0.001)

        newLabelsAutism = np.array(labels)[keepIndexesAutism]
        newLabelsYeoAutism = np.array(yeo_labels)[keepIndexesAutism]

        df = pd.DataFrame(filterAutism,index=newLabelsYeoAutism,columns=newLabelsAutism)
        df.to_excel("result/heatmapDataAutism_{}.xls".format(fileName))


autism = AutismClassifiers()


autism.Classify_Atlas("Result",1,70)
autism.Classify_Atlas_With_Combat("Result",1,70)
autism.Classify_Atlas_For_Each_Site()

connectivity = ConnectivityMap()
connectivity.Connectivity_All()
connectivity.Connectivity_Control()
