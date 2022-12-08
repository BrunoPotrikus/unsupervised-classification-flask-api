from fcmeans import FCM
from sklearn.metrics import silhouette_score
import numpy as np

def classifierFuzzy(xTrain, xTest):

    npXtrain = np.array(xTrain)
    npXtest = np.array(xTest)

    fcm = FCM(n_clusters=2)
    fcm.fit(npXtrain)

    predicted = fcm.predict(npXtest)

    silhouette = silhouette_score(xTest, predicted, metric='euclidean')

    npXtest = np.array(xTest)
    yPred = predicted.reshape(-1)

    mean0 = npXtest[yPred==0].mean()
    mean1 = npXtest[yPred==1].mean()
    std0 =  npXtest[yPred==0].std(ddof=1)
    std1 =  npXtest[yPred==1].std(ddof=1)

    fisher = (abs(mean0 - mean1) ** 2) / ((std0 ** 2) + (std1 ** 2))

    arr = { 'classifier': 'KMeans', 'yPred': predicted.tolist(), 'silhouette': silhouette, 'fisher': fisher }

    return arr

