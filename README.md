# Feature Selection in GenomeData
Finding 100 prominent genes out of 4434 features (genes) using F-Test and then train the classifiers using those 100 genes.

There are 40 data instances and 4434 features (genes). First row is the class numbers second - end rows contain feature vectors. Each feature vector is a column. Adding the class number at the begining, each feature vector + label is a 4435 dimensional vector.

Using the 100 prominent genes (features) out of 4434, train the following classifiers:
1. SVM linear kernel
2. linear regression
3. KNN (k=3)
4. centroid method

Use the trained classifiers to predict the class labels of data instances provided in GenomeTestX.txt. There are total 10 data instances.
