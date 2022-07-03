from flask import Flask, request, Response
from flask_cors import CORS
from flask import jsonify
import requests
import json

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run("localhost", 8080)
