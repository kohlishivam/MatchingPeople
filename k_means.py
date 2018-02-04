# %reset -f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator

# importing the mall data-set
dataset = pd.read_csv('output.csv')
from sklearn.preprocessing import LabelEncoder
X = dataset.iloc[:, 1:10].values
# print X
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
# print X
# importing the mall data-set
# Using the Elbow method to find the optimal number of clusters
from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i+1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
 

# Applying k-means to the mall dataset


from sklearn.decomposition import PCA
import pylab as pl
pca=PCA(n_components=2).fit(X_train)
pca_2d=pca.transform(X_train)
pl.figure('Trained Data predictions')
pl.scatter(pca_2d[:,0],pca_2d[:,1])
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_train)
pl.scatter(pca_2d[:,0],pca_2d[:,1],c=kmeans.labels_)
pl.show()
y_kmeans = kmeans.fit_predict(X_test)
pca=PCA(n_components=2).fit(X_test)
pca_2d=pca.transform(X_test)
pl.figure('Test Data predictions')
pl.scatter(pca_2d[:,0],pca_2d[:,1],c=kmeans.labels_)
pl.show()

# print y_kmeans

X_id = dataset.iloc[:,0].values

array_users = {}
for i in range(len(y_kmeans)):
	array_users[X_id[i]] = y_kmeans[i]

 

sorted_x = sorted(array_users.items(), key=operator.itemgetter(1))


# print sorted_x


#pl.scatter(pca_2d[y_kmeans==0,0],pca_2d[y_kmeans==0,1] , c = 'red')
#pl.scatter(pca_2d[y_kmeans==1,0],pca_2d[y_kmeans==1,1] , c = 'blue')
# Visulizing the clusters
#plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 3], s=0.5, c='red', label='Cluster 1')
#plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 3], s=0.5, c='blue', label='Cluster 2')

#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=0.5, c='yellow', label='Centroids')
#plt.title('Clusters of clients')
#plt.xlabel('User ID (k$)')
#plt.ylabel('age (1-100)')
#plt.ylim((0,100))
#plt.legend()
#plt.show()

def matching_algo(fbid , cluster , array_users):

	same_cluster_users = []
	for i,j in array_users.items():
		print i , j
		if j == cluster:
			same_cluster_users.append(i)

	print same_cluster_users		


	Y = dataset.iloc[:, 0].values

	users = Y.tolist()

	age = X[:,3].tolist()

	mother_tongue = X[:,0].tolist()

	school = X[:,7].tolist()

	college = X[:,8].tolist()


	index_fbid = users.index(fbid)
	
	score_dict = {}
	match_score = 0
	for i in same_cluster_users:
		match_score = 0
		if age[users.index(i)] == int(age[index_fbid]) +1 or int(age[index_fbid]) or int(age[index_fbid] -1) :
			match_score = match_score + 2

		if mother_tongue[users.index(i)]   ==  mother_tongue[index_fbid]:
			match_score	 = match_score + 3

		if school[users.index(i)]   ==  school[index_fbid]:
			match_score	 = match_score + 1
		
		if college[users.index(i)]   ==  college[index_fbid]:
			match_score	 = match_score + 1

		score_dict[i] = match_score
		

	sorted_dict=sorted(score_dict.items(), key=lambda x:x[1])
	print sorted_dict

	sorted_dict = dict(sorted_dict)

	xgreen = []
	ygreen = []
	xyellow = []
	yyellow=[]
	xblue = []
	yblue =[]
	for i,j in sorted_dict.items():

		if j<3:
			xblue.append(i)
			yblue.append(j)

		elif j>3 and j<5:
			xyellow.append(i)
			yyellow.append(j)

		elif j>5:
			xgreen.append(i)
			ygreen.append(j)

	#print xblue,yblue		


	plt.scatter(xgreen, ygreen, s=5, c='green', label='best match')
	plt.scatter(xblue, yblue, s=5, c='blue', label='nuetral')	
	plt.scatter(xyellow,yyellow , s=5 , c = 'orange' , label = 'medium match')	
	plt.ylim((0,10))
	plt.show()
	return sorted_dict		

	






matching_algo(100543,3,array_users)
matching_algo(100177,1,array_users)
matching_algo(100178,2,array_users)
matching_algo(100189,0,array_users)
