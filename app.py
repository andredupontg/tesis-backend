from flask import Flask, request
from flask_cors import CORS
from ml import covidDiagnosis
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route('/getResults', methods=['POST'])
def getResults():
    symptomsDataframe = pd.DataFrame.from_dict(
        request.get_json(), orient="index")
    covidResult = covidDiagnosis(symptomsDataframe)
    return covidResult


if __name__ == "__main__":
    app.run("localhost", 8080)
