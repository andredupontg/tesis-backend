from flask import Flask, request, Response
from flask_cors import CORS
from flask import jsonify
import requests
import json
from ml import covidDiagnosis

app = Flask(__name__)
CORS(app)


@app.route('/getResults', methods=['POST'])
def getResults(symptoms):
    return covidDiagnosis


if __name__ == "__main__":
    app.run("localhost", 8080)
