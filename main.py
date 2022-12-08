from flask import Flask, request
from KMeans import classifierKMeans
from FuzzyMeans import classifierFuzzy

app = Flask('Classificação Não Supervisionada')

@app.route('/kmeans', methods=["POST"])
def getKMeans():

    body = request.get_json()

    calcKMeans = classifierKMeans(body["xTrain"], body["xTest"])

    return calcKMeans

@app.route('/fuzzymeans', methods=["POST"])
def getFuzzy():

    body = request.get_json()

    calcFuzzy = classifierFuzzy(body["xTrain"], body["xTest"])

    return calcFuzzy

app.run(debug=True)