from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rc("font", size=14)


def leerDataset():
    dataset = pd.read_csv(filepath_or_buffer='COVIDMachineLearning.csv')
    dataframe = pd.DataFrame(dataset)
    return dataframe


def limpiezaDataframe(dataframe):
    dataframe = dataframe[dataframe.corona_result != "other"]
    dataframe = dataframe.dropna(subset=['age_60_and_above'])
    dataframe = dataframe.dropna(subset=['gender'])
    dataframe = dataframe.replace("positive", 1)
    dataframe = dataframe.replace("negative", 0)
    dataframe = dataframe.replace("Abroad", "No Contact")
    dataframe = dataframe.replace("Other", "No Contact")
    return dataframe


def tecnicaOneHotEncoding(dataframe, feature, prefijo):
    variableDummy = pd.get_dummies(dataframe[feature], prefix=prefijo)
    dataframe = dataframe.drop(feature, axis=1)
    dataframe = pd.concat([dataframe, variableDummy], axis=1)
    return dataframe


def separarDataframe(dataframe):
    dataframe1 = dataframe["corona_result"]
    dataframe2 = dataframe.drop("corona_result", axis=1)
    arregloDataframes = [dataframe1, dataframe2]
    return arregloDataframes


def regresionLogistica(Y, X, symptomsDataframe):
    print("*********************************** FASE ENTRENAMIENTO ***********************************")
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    overSampler = RandomOverSampler(sampling_strategy=0.23)
    xOverSamp, yOverSamp = overSampler.fit_resample(
        xEntrenamiento, yEntrenamiento)
    regresionLog = LogisticRegression(random_state=42)
    regresionLog.fit(xOverSamp, yOverSamp)
    print("*********************************** FASE PREDICCION ***********************************")
    result = regresionLog.predict(symptomsDataframe)
    return str(result[0])


def covidDiagnosis(symptomsDataframe):
    dataframe = leerDataset()
    aux = limpiezaDataframe(dataframe)
    dataframe = aux
    aux = tecnicaOneHotEncoding(
        dataframe, "age_60_and_above", "Age 60 At Least")
    dataframe = aux
    aux = tecnicaOneHotEncoding(dataframe, "gender", "Gender")
    dataframe = aux
    aux = tecnicaOneHotEncoding(
        dataframe, "test_indication", "Test Indication")
    dataframe = aux
    aux = separarDataframe(dataframe)
    Y = aux[0]
    X = aux[1]
    return regresionLogistica(Y, X, symptomsDataframe)
