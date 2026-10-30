import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_excel('C:/Users/geeta/Downloads/Iris_data.xlsx')
df.info()
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3,random_state = 42, n_init=10)
km.fit(X_scaled)
df['predicted_cluster'] = km.labels_

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]
plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
sns.scatterplot(x = 'pca1', y = 'pca2', hue = 'predicted_cluster', data = df,palette = 'Set1',s=70,style ='predicted_cluster',)
plt.title('K - Means Clustering(Predicted)')
plt.xlabel('pca1')
plt.ylabel('pca2')
plt.subplot(1,2,2)
df['target_name'] =df['target'].map({0:target_names[0],1:target_names[1],2:target_names[2]})
sns.scatterplot(x ='pca1',y ='pca2',hue ='target_name',data = df,palette = 'Set2',s=70)
plt.title('True Iris species')
plt.xlabel('pca1')
plt.ylabel('pca2')
plt.tight_layout()

plt.show()
print('\n-- Cluster vs True Label Cross-Tabulation ---')
print(pd.crosstab(df['predicted_cluster'],df['target_name']))