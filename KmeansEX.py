import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means, KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.datasets import load_iris
from sklearn import datasets

iris = datasets.load_iris()
# print(dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df = pd.DataFrame(iris.data,columns=iris.feature_names)

df['flower'] = iris.target
df = df.drop(['sepal length (cm)', 'sepal width (cm)','flower'],axis=1)

km = KMeans(n_clusters=3)
yp = km.fit_predict(df)
# print(yp)

df['cluster'] = yp
print(df)

df0 = df[df.cluster==0]
df1 = df[df.cluster==1]
df2 = df[df.cluster==2]

plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],marker='*',color='red')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],marker='*',color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],marker='*',color='green')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.show()