#importing different liabries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset into a variable 'data'
data=pd.read_csv('Wholesale customers data.csv')
print(data.head())

#describing the data set including mean, standard deviation, minimun value,maximum values and percentile values
print(data.describe())

#checking for data types and any null missing value in the data set
print(data.info())
print(data.isnull().sum())

#counting the individual number of regions
print(data.Region.value_counts())

#counting the individual number of channel
print(data.Channel.value_counts())

#creating histogram graph for each variable/features
def features(i):
    if data[i].dtype!='object':
        plt.figure(figsize=(8,5))
        plt.title("histogram of "+str(i))
        data[i].plot.hist(bins = 16)
        plt.show()
        plt.clf()
sns.set()
j=['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
for k in j:
    features(i=k)


#storing the dataset of some features into another variable 'data_set'
data_set = np.log(data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].copy())
print(data_set.head())

#heatmap-to show dependency of each feature on other features
data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True,cmap='Blues')
plt.show()

#plotting scatter plot
def scatterplot(i,j):
    sns.regplot(data=data,x=i,y=j)
    plt.show()
scatterplot(i='Milk',j='Grocery')
scatterplot(i='Milk',j='Detergents_Paper')
scatterplot(i='Detergents_Paper',j='Grocery')

def categorical(i,j):
    pd.crosstab(data[i],data[j]).plot(kind='bar',cmap='Paired')
    plt.show()
    print(pd.crosstab(data[i],data[j]))
categorical(i='Channel',j='Region')


# find outliers in the given dataset by using Tukey's Method
## replacing with median to treat the outliers
### IQR=InterQuartile Range
for k in list(data_set.columns):
    IQR = np.percentile(data_set[k], 75) - np.percentile(data_set[k], 25)
    Outlier_top = np.percentile(data_set[k], 75) + 1.5 * IQR
    Outlier_bottom = np.percentile(data_set[k], 25) - 1.5 * IQR
    data_set[k] = np.where(data_set[k] > Outlier_top, np.percentile(data_set[k], 50), data_set[k])
    data_set[k] = np.where(data_set[k] < Outlier_bottom, np.percentile(data_set[k], 50), data_set[k])

#plotting histogram after replacing outliers with respective medians
def df(i):
    if data_set[i].dtype!='object':
        plt.show()
        plt.title("histogram of "+str(i))
        data_set[i].plot.kde()
        plt.show()
for k in j:
    df(i=k)

#plotting pair plot
sns.pairplot(data_set,diag_kind = 'kde')

#copying dataset into another variable abd creating dummies
dataset = data.copy()
dataset['Channel'] = dataset['Channel'].map({1:'Horeca', 2:'Retail'})
dataset['Region'].replace([1,2,3],['Lisbon','Oporto','other'],inplace=True)
print(dataset.head())

df  =  pd.concat([dataset[['Channel','Region']],data_set],axis=1)
print(df.head())
df = pd.get_dummies(df,columns=['Channel','Region'],drop_first=True)
print(df.head())

##
########Model 1: Using MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_std = scaler.fit_transform(df)
df_std = pd.DataFrame(df_std,columns=df.columns)
print(df_std.head())

#######Model 2: Using Standard Scaling
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#df_std = scaler.fit_transform(df)
#df_std = pd.DataFrame(df_std,columns=df.columns)
#print(df_std.head())

######Model 3: Using Normalized Scaling
#from sklearn.preprocessing import Normalizer
#scaler = Normalizer()
#df_std = scaler.fit_transform(df)
#df_std = pd.DataFrame(df_std,columns=df.columns)
#print(df_std.head())


X = df_std.copy()
from sklearn.cluster import KMeans
cluster_range = range(1,20)
cluster_wss=[]
for i in cluster_range:
    model = KMeans(n_clusters=i, init='k-means++')
    model.fit(X)
    cluster_wss.append(model.inertia_)

plt.figure(figsize=[10,6])
plt.title('WSS curve for finding Optimul K value')
plt.xlabel('No. of clusters')
plt.ylabel('Inertia or WSS')
plt.plot(list(cluster_range),cluster_wss,marker='o')
plt.show()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=5,init='k-means++',random_state=0)
model.fit(X)
data_final = data.copy()
print(data_final.head())
data_final['clusters']=model.predict(X)
print(data_final.head())

#applying PCA to the model
from sklearn.decomposition import PCA
pca = PCA()
pc = pca.fit_transform(df_std)
pc_df = pd.DataFrame(pc)
print(pca.explained_variance_ratio_)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(df_std)
pc_df = pd.DataFrame(pc)
print(pc_df.head())

pca = pd.concat([pc_df,data_final['clusters']],axis=1)
pca.columns = ['pc1','pc2','clusters']
print(pca.shape)
print(pca.head())
print(pca.clusters.value_counts())

plt.figure(figsize=(12,6))
sns.scatterplot(x='pc1',y='pc2',hue= 'clusters', data=pca,palette='Set1')
plt.show()

data_final.groupby('clusters').Fresh.mean().plot(kind='bar')
plt.show()
data_final.groupby('clusters').Milk.mean().plot(kind='bar')
plt.show()
data_final.groupby('clusters').Grocery.mean().plot(kind='bar')
plt.show()
data_final.groupby('clusters').Frozen.mean().plot(kind='bar')
plt.show()
data_final.groupby('clusters').Detergents_Paper.mean().plot(kind='bar')
plt.show()
