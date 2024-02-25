import pandas as pd
import glob
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from scipy.stats import f_oneway
import pingouin as pg

def makep_values_for_combat_and_noncombat(filename1,filename2,Sheet):
    xls1 = pd.read_excel(filename1,sheet_name=Sheet)
    xls2 = pd.read_excel(filename2,sheet_name=Sheet)

    xls1 = xls1.to_numpy()[:,1:]
    xls2 = xls2.to_numpy()[:,1:]
    pvalues = []
    for i in range(xls1.shape[1]):
        data1 = xls1[:,i]
        data2 = xls2[:,i]
        pv = stats.ttest_ind(data1,data2).pvalue
        if (pv>0.1):
            pv=0.1
        pvalues.append(pv)
    return pvalues

def makedf_tovector(filename1,filename2,sheets):
    xls1 = pd.ExcelFile(filename1)
    xls2 = pd.ExcelFile(filename2)

    df_atlas = []
    df_combo = []
    pvalues = []
    for sheet in sheets:
        dftemp = pd.read_excel(xls1,sheet)
        df_atlas.append(dftemp)
        dftemp = pd.read_excel(xls2,sheet)
        df_combo.append(dftemp)
    sheet_index = 0
    for df1,df2 in zip(df_atlas,df_combo):
        data1 = np.array(df1.to_numpy()[:,1:].flatten(),dtype=float)
        data2 = np.array(df2.to_numpy()[:,1:].flatten(),dtype=float)
        pv = stats.ttest_ind(data1, data2).pvalue
        hist_data = [data1, data2]

        group_labels = ['Not harmmonized', 'Harmmonized']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels,bin_size=0.01)
        fig.update_layout(xaxis_title=sheets[sheet_index] ,title_x=0.5)
        fig.write_image("Result/atlas/{}.png".format(sheets[sheet_index]),scale = 2,width = 500,height=350)
        #fig.show()
        pvalues.append(pv)
        sheet_index+=1
    return pvalues

def makedf(filename1,filename2,metrics):
    xls1 = pd.ExcelFile(filename1)
    xls2 = pd.ExcelFile(filename2)
    sheets = [metrics]
    df_atlas = []
    df_combo = []
    for sheet in sheets:
        dftemp = pd.read_excel(xls1,sheet)
        df_atlas.append(dftemp)
        dftemp = pd.read_excel(xls2,sheet)
        df_combo.append(dftemp)
    atlas_names = df_atlas[0].columns[1:]
    classifiers = df_combo[0].to_numpy()[:, 0]
    index = 0
    dataframe = []
    for df1,df2 in zip(df_atlas,df_combo):
        data1 = df1.to_numpy()[:,1:]
        data2 = df2.to_numpy()[:,1:]
        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                print ("{},{}".format(i,j))
                temp1 = [data1[i,j],atlas_names[j],classifiers[i],"Not Harmmonized"]
                temp2 = [data2[i,j],atlas_names[j],classifiers[i],"Harmonized with Combo"]
                dataframe.append(temp1)
                dataframe.append(temp2)
    df = pd.DataFrame(dataframe,columns=["Performance","Atlas","Classifier","Group"])
    return df,atlas_names,classifiers

def heatmappvaluesatlas():
    data = pd.read_excel("Result/Atlas/Cross_NonCombat.xls",sheet_name="Accuracy")
    atlas_names = data.columns[1:]
    classifiers = data.to_numpy()[:,0]
    data =data.to_numpy() [:, 1:]
    pvalues = np.zeros((14, 14))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if (j<i):
                pv = stats.ttest_ind(data[:,i], data[:,j]).pvalue
                if (pv >= 0.05):
                    pvalues[i,j] =0.06
                else:
                    pvalues[i,j] = pv
            else:
                pvalues[i, j] = 0.06
    sns.heatmap(pvalues,xticklabels=atlas_names, yticklabels=atlas_names )
    plt.savefig("Result/Atlas/NonCross_Combat.jpg",bbox_inches='tight')

def plot_density_plot():
    plt.figure(figsize=(4, 5), dpi=250)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.5,1]},figsize = (4,5),dpi=250,sharex=True)
    path_result1 = "Result/Atlas/Cross_AllAge 1-10.xls"
    path_result2 = "Result/Atlas/Cross_Combat 1-10.xls"
    Sheets = ["Accuracy", "Precision", "Auc", "Recall", "Specificity"]
    sheet_name = Sheets[3]

    pvalues = makedf_tovector(path_result1, path_result2, Sheets)
    print (pvalues)

def drawboxplot():
    plt.figure(figsize=(4, 5), dpi=250)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.5,1]},figsize = (4,5),dpi=250,sharex=True)
    path_result1 = "Result/Atlas/Cross_NonCombat.xls"
    path_result2 = "Result/Atlas/Cross_Combat.xls"
    Sheets = ["Accuracy", "Precision", "Auc", "Recall", "Specificity"]
    sheet_name = Sheets[4]

    df,atlas_names,classifiers = makedf(path_result1, path_result2, sheet_name)
    pvalues = makep_values_for_combat_and_noncombat(path_result1, path_result2, sheet_name)

    #ax1=plt.subplot(211)
    ax1.bar(x=range(0, len(pvalues)), height=pvalues)
    ax1.set_ylabel('Pvalue')
    plt.setp(ax1.get_xticklabels(), visible=False)

    sns.boxplot(ax=ax2, data=df, x='Atlas', y='Performance', hue='Group', showfliers=True,width=.7,medianprops=dict(color="red", alpha=0.7))
    ax2.legend().set_visible(False)
    plt.xlabel('Atlas Name')
    plt.ylabel(sheet_name)
    plt.xticks(rotation=90, )
    plt.savefig("Result/Atlas/{}.jpg".format(sheet_name),bbox_inches='tight')

def anova():
    path_result1 = "Result/Atlas/Cross_AllAge 1-10.xls"
    path_result2 = "Result/Atlas/Cross_Combat 1-10.xls"
    Sheets = ["Accuracy", "Precision", "Auc", "Recall", "Specificity"]
    for sheet in Sheets:
        xls1 = pd.read_excel(path_result1,sheet_name=sheet)
        xls2 = pd.read_excel(path_result2,sheet_name=sheet)

        xls1 = xls1.to_numpy()[:,1:]
        xls2 = xls2.to_numpy()[:,1:]
        anova_statistic1 = f_oneway(xls1[:,0],xls1[:,1],xls1[:,2],xls1[:,3],xls1[:,4],xls1[:,5],xls1[:,6],xls1[:,7],xls1[:,8],xls1[:,9],xls1[:,10],xls1[:,11],xls1[:,12],xls1[:,13])
        anova_statistic2 = f_oneway(xls2[:,0],xls2[:,1],xls2[:,2],xls2[:,3],xls2[:,4],xls2[:,5],xls2[:,6],xls2[:,7],xls2[:,8],xls2[:,9],xls2[:,10],xls2[:,11],xls2[:,12],xls2[:,13])
        print (sheet)
        print (anova_statistic1)
        print (anova_statistic2)

heatmappvaluesatlas()
#plot_density_plot()
#anova()
#drawboxplot()
#phenotype = pd.read_excel('Labels.xls')
#phenotype = phenotype.to_numpy()
#site_names = np.unique(phenotype[:, 10])

#****************************************************************** pvalues among the sites ***************************************************

#path_result1 = "Result/Atlas_Site/Cross_*.*"


#index = 1
#rejectedList =[ 1,3,4,6,7,8]

#for file1 in glob.glob(path_result1):
#    basename = os.path.basename(file1)
#    accuracy1 = pd.read_excel(file1, sheet_name='Accuracy').to_numpy()

    #atlas_names = accuracy1[:,0]
    #accuracy1=accuracy1[:,1:]

    #mean = np.mean(accuracy1,axis=1)
    #pvalues = np.zeros((14, 14))

    #for i in range(accuracy1.shape[0]):
    #    for j in range(accuracy1.shape[0]):
    #        if (i!=j):
    #            pv = stats.ttest_ind(accuracy1[i,:], accuracy1[j,:]).pvalue
    #            pvalues[i,j] = pv
    #        else:
    #            pvalues[i, j] = 1
    #plt.figure(figsize=(20, 8), dpi=300)
    #plt.subplot(1,2,1)
    #plt.bar(x=atlas_names,height=mean)
    #plt.xticks(rotation=90)
    #plt.subplot(1, 2, 2)
    #sns.heatmap(pvalues, xticklabels=atlas_names, yticklabels=atlas_names, annot=False)
    #plt.title(basename[6:len(basename)-4])
    #plt.savefig("Result/Atlas_Site/pvalues_{}.jpg".format(basename[6:len(basename)-4]), bbox_inches='tight')
    #plt.subplot(2,1,index)
    #if (index == 9):
    #    sns.heatmap(pvalues, xticklabels=atlas_names, yticklabels=atlas_names, annot=False)
    #elif (index>8):
    #    sns.heatmap(pvalues, xticklabels=atlas_names ,annot=False)
    #elif (index==1 or index==5 ):
    #    ax= sns.heatmap(pvalues,  yticklabels=atlas_names, annot=False)
    #else:
    #    ax= sns.heatmap(pvalues,  annot=False)
    #    ax.tick_params(left=False, bottom=False)
    #plt.title(basename[6:len(basename)-4])
    #index+=1

#plt.show()
#plt.savefig("Result/Atlas_Site/pvalue_among_site.jpg",bbox_inches='tight')


#**************************************total atlas pvalues for all data
