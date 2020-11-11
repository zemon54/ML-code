# Importing libraries

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# dataset contains 2 columns with id and product description
product_descriptions = pd.read_csv('D:\ML Recommendar System\product_descriptions.csv')
print(product_descriptions.shape)

# Missing values
product_descriptions = product_descriptions.dropna()
product_descriptions.shape
print(product_descriptions.head())

# Only taking the product descriptions
product_descriptions1 = product_descriptions.head(50000)
# product_descriptions1.iloc[:,1]
product_descriptions1["product_description"].head(10)

# Converting the text in product description into numerical data for analysis
vectorizer = TfidfVectorizer(stop_words='english')
# X1 contains vector with 500 rows and columns with each different word in the document
# it contains tf-idf score for each word in the description of products
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
# print(X1)

# Fitting K-Means to the dataset
X = X1
kmeans = KMeans(n_clusters=10, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")


# plt.show()

# print clusters
def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# Top words in each cluster based on product description
# # Optimal clusters is
true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


# for i in range(true_k):
#    print_cluster(i)


# Predicting clusters based on key search words
def show_recommendations(product):
    print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    print(prediction)
    print_cluster(prediction[0])


show_recommendations("hammer")
