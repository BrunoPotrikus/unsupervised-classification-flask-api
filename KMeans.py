from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def classifierKMeans(xTrain, xTest):

    kmeans = KMeans(n_clusters=2, random_state=0).fit(xTrain)

    predicted = kmeans.predict(xTest)

    silhouette = silhouette_score(xTrain, kmeans.labels_, metric='euclidean')

    npXtest = np.array(xTest)
    yPred = predicted.reshape(-1)

    mean0 = npXtest[yPred==0].mean()
    mean1 = npXtest[yPred==1].mean()
    std0 =  npXtest[yPred==0].std(ddof=1)
    std1 =  npXtest[yPred==1].std(ddof=1)

    fisher = (abs(mean0 - mean1) ** 2) / ((std0 ** 2) + (std1 ** 2))

    arr = { 'classifier': 'KMeans', 'yPred': predicted.tolist(), 'silhouette': silhouette, 'fisher': fisher }

    return arr