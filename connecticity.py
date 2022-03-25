from mne.viz import plot_connectivity_circle
from mne.viz import plot_sensors_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def Yeo_Network(filename):
    df = pd.read_excel(filename)
    data = df.to_numpy()
    columns = data[0,:][1:]
    rows = data[:, 0][1:]
    nodes = np.unique(rows)

    graphCount = np.zeros((nodes.shape[0],nodes.shape[0]))
    graphWeight = np.zeros((nodes.shape[0],nodes.shape[0]))

    data = data [1:,1:]
    for i in range(data.shape[0]):
        for j in range(i):
            ni = np.where(nodes==rows[i])[0][0]
            nj = np.where(nodes==columns[j])[0][0]
            print ("{}_{}".format(ni,nj))
            graphCount[ni,nj] = graphCount[ni,nj]+1
            graphWeight[ni,nj] = graphWeight[ni,nj]+ data[i,j]
    result = np.divide(graphWeight,graphCount)
    return graphWeight,nodes
def DrawChordDiagram(data,node_names):
    colors_ref = ['orange', 'maroon', 'cornflowerblue', 'yellow', 'brown', 'darkblue', 'lightgreen', 'forestgreen',
                  'black', 'hotpink', 'lightpink', 'lightsteelblue', 'lightseagreen', 'steelblue', 'indigo', 'red']
    plot_connectivity_circle(data, node_names,
                             colormap='Blues',
                             facecolor='white',
                             node_linewidth=1,
                             textcolor='black',
                             colorbar=True, linewidth=1, node_colors=colors_ref)


filename = "result/heatmapDataAutism_Combat.xls"
resultAutism,nodes = Yeo_Network(filename)
mask = np.triu(np.ones_like(resultAutism))
sns.heatmap(resultAutism, xticklabels=nodes, yticklabels=nodes, mask=mask)
plt.title("Heatmap of 17 sub network")
plt.xlabel("Networks Node")
plt.ylabel("Networks Node")
plt.savefig("result/Heat_Autism.jpg", bbox_inches='tight')
DrawChordDiagram(resultAutism,nodes)

filename = "result/heatmapDataHealth_Combat.xls"
resultHealth,nodes = Yeo_Network(filename)
mask = np.triu(np.ones_like(resultHealth))
sns.heatmap(resultHealth, xticklabels=nodes, yticklabels=nodes, mask=mask )
plt.title("Heatmap of 17 sub network")
plt.xlabel("Networks Node")
plt.ylabel("Networks Node")
plt.savefig("result/Heat_Health.jpg", bbox_inches='tight')
DrawChordDiagram(resultHealth,nodes)

result =  resultHealth - resultAutism
mask = np.triu(np.ones_like(result))
sns.heatmap(result, xticklabels=nodes, yticklabels=nodes, mask=mask )
plt.savefig("result/Heat_Diff.jpg", bbox_inches='tight')
DrawChordDiagram(result,nodes)




df = pd.read_excel('result/heatmapData_Combat.xls')
node_names = df.columns[1:]
data =  df.to_numpy()
network_Node = data[1:,0]

nodes_ref = np.unique(network_Node)
colors_ref = ['orange','maroon','cornflowerblue','yellow','brown','darkblue','lightgreen','forestgreen',
          'black','hotpink','lightpink','lightsteelblue','lightseagreen','steelblue','indigo','red']

colors = []
for roi in network_Node:
    index = np.where(nodes_ref==roi)[0]
    colors.append(colors_ref[index[0]])


data = data[:,1:]
#plt.figure(num=None, figsize=(8, 8), facecolor='black')
plt.savefig("result/chord.pdf",bbox_inches='tight')
plot_connectivity_circle(data,node_names,
                         colormap='Blues',
                         facecolor='white',
                         node_linewidth=1,
                         textcolor='black',
                         colorbar=True,linewidth=1, node_colors=colors)
