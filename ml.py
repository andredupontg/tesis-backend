from sklearn.svm import SVC
import itertools
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.rc("font", size=14)


def leerDataset():
    dataset = pd.read_csv(filepath_or_buffer='COVIDMachineLearning.csv')
    dataframe = pd.DataFrame(dataset)
    return dataframe


def resumenDataframe(dataframe):
    print(dataframe.head())
    print(dataframe.describe())


def resumenFeature(dataframe, feature):
    print(dataframe[feature].describe())


def identificarNulos(dataframe):
    print(dataframe.isnull().sum())


def identificarValoresFeature(dataframe, feature):
    print(dataframe[feature].value_counts())


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


def regresionLogistica(Y, X):
    print("*********************************** FASE ENTRENAMIENTO ***********************************")
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    overSampler = RandomOverSampler(sampling_strategy=0.23)
    xOverSamp, yOverSamp = overSampler.fit_resample(
        xEntrenamiento, yEntrenamiento)
    regresionLog = LogisticRegression(random_state=42)
    regresionLog.fit(xOverSamp, yOverSamp)
    print("*********************************** FASE PREDICCION ***********************************")
    yPrediccion = regresionLog.predict(xPrueba)
    yPrediccionEntrena = regresionLog.predict(xEntrenamiento)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        regresionLog.score(xPrueba, yPrueba)))

    print("*********************************** MATRIZ DE CONFUNSION ***********************************")
    matrizConfusion = confusion_matrix(yPrueba, yPrediccion)
    matrizConfusionEntrena = confusion_matrix(
        yEntrenamiento, yPrediccionEntrena)
    print(matrizConfusion)
    print(matrizConfusionEntrena)

    print("*********************************** METRICAS DE PRECISION ***********************************")
    print(classification_report(yPrueba, yPrediccion))

    print("*********************************** CURVA ROC ***********************************")
    areaBajoCurvaROC = roc_auc_score(yPrueba, regresionLog.predict(xPrueba))
    fpr, tpr, thresholds = roc_curve(
        yPrueba, regresionLog.predict_proba(xPrueba)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Regresión Logística (Área = %0.2f)' %
             areaBajoCurvaROC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de Falsos Positivos')
    plt.ylabel('Ratio de Verdaderos Positivos')
    plt.legend(loc="lower right")
    plt.savefig('AreaBajoLaCurvaROC')
    plt.show()

    plot_confusion_matrix(matrizConfusion, classes=['Prueba Negativa', 'Prueba Positivo'],
                          title='Matriz de confusión RL')
    plt.savefig('cm_RL.png')


def gradientBoosting(Y, X):
    print("*********************************** FASE ENTRENAMIENTO ***********************************")
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    overSampler = RandomOverSampler(sampling_strategy=0.23)
    xOverSamp, yOverSamp = overSampler.fit_resample(
        xEntrenamiento, yEntrenamiento)
    gradientBoost = GradientBoostingClassifier(learning_rate=0.05, verbose=1)
    gradientBoost.fit(xOverSamp, yOverSamp)
    print("*********************************** FASE PREDICCION ***********************************")
    yPrediccion = gradientBoost.predict(xPrueba)
    yPrediccionEntrena = gradientBoost.predict(xEntrenamiento)
    print('Accuracy of gradient boosting classifier on test set: {:.2f}'.format(
        gradientBoost.score(xPrueba, yPrueba)))

    print("*********************************** MATRIZ DE CONFUNSION ***********************************")
    matrizConfusion = confusion_matrix(yPrueba, yPrediccion)
    matrizConfusionEntrena = confusion_matrix(
        yEntrenamiento, yPrediccionEntrena)
    print(matrizConfusion)
    print(matrizConfusionEntrena)

    print("*********************************** METRICAS DE PRECISION ***********************************")
    print(classification_report(yPrueba, yPrediccion))

    print("*********************************** CURVA ROC ***********************************")
    areaBajoCurvaROC = roc_auc_score(yPrueba, gradientBoost.predict(xPrueba))
    fpr, tpr, thresholds = roc_curve(
        yPrueba, gradientBoost.predict_proba(xPrueba)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Gradient Boosting (Área = %0.2f)' %
             areaBajoCurvaROC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de Falsos Positivos')
    plt.ylabel('Ratio de Verdaderos Positivos')
    plt.legend(loc="lower right")
    plt.savefig('AreaBajoLaCurvaROC')
    plt.show()

    plot_confusion_matrix(matrizConfusion, classes=['Prueba Negativa', 'Prueba Positivo'],
                          title='Matriz de confusión GB')
    plt.savefig('cm_GB.png')


def supportVectorMachine(Y, X):
    print("*********************************** FASE ENTRENAMIENTO ***********************************")
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    overSampler = RandomOverSampler(sampling_strategy=0.23)
    xOverSamp, yOverSamp = overSampler.fit_resample(
        xEntrenamiento, yEntrenamiento)
    svc = SVC(probability=True)
    svc.fit(xOverSamp, yOverSamp)
    print("*********************************** FASE PREDICCION ***********************************")
    yPrediccion = svc.predict(xPrueba)
    yPrediccionEntrena = svc.predict(xEntrenamiento)
    print('Accuracy of support vector machine classifier on test set: {:.2f}'.format(
        svc.score(xPrueba, yPrueba)))

    print("*********************************** MATRIZ DE CONFUNSION ***********************************")
    matrizConfusion = confusion_matrix(yPrueba, yPrediccion)
    matrizConfusionEntrena = confusion_matrix(
        yEntrenamiento, yPrediccionEntrena)
    print(matrizConfusion)
    print(matrizConfusionEntrena)

    print("*********************************** METRICAS DE PRECISION ***********************************")
    print(classification_report(yPrueba, yPrediccion))

    print("*********************************** CURVA ROC ***********************************")
    areaBajoCurvaROC = roc_auc_score(yPrueba, svc.predict(xPrueba))
    fpr, tpr, thresholds = roc_curve(yPrueba, svc.predict_proba(xPrueba)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Support Vector Machine (Área = %0.2f)' %
             areaBajoCurvaROC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de Falsos Positivos')
    plt.ylabel('Ratio de Verdaderos Positivos')
    plt.legend(loc="lower right")
    plt.savefig('AreaBajoLaCurvaROC')
    plt.show()

    plot_confusion_matrix(matrizConfusion, classes=['Prueba Negativa', 'Prueba Positivo'],
                          title='Matriz de confusión SVC')
    plt.savefig('cm_SV.png')


def bagging(Y, X):
    print("*********************************** FASE ENTRENAMIENTO ***********************************")
    xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(
        X, Y, test_size=0.3, random_state=0, stratify=Y)
    overSampler = RandomOverSampler(sampling_strategy=0.23)
    xOverSamp, yOverSamp = overSampler.fit_resample(
        xEntrenamiento, yEntrenamiento)
    bg = BaggingClassifier()
    bg.fit(xOverSamp, yOverSamp)
    print("*********************************** FASE PREDICCION ***********************************")
    yPrediccion = bg.predict(xPrueba)
    yPrediccionEntrena = bg.predict(xEntrenamiento)
    print('Accuracy of bagging classifier on test set: {:.2f}'.format(
        bg.score(xPrueba, yPrueba)))

    print("*********************************** MATRIZ DE CONFUNSION ***********************************")
    matrizConfusion = confusion_matrix(yPrueba, yPrediccion)
    matrizConfusionEntrena = confusion_matrix(
        yEntrenamiento, yPrediccionEntrena)
    print(matrizConfusion)
    print(matrizConfusionEntrena)

    print("*********************************** METRICAS DE PRECISION ***********************************")
    print(classification_report(yPrueba, yPrediccion))

    print("*********************************** CURVA ROC ***********************************")
    areaBajoCurvaROC = roc_auc_score(yPrueba, bg.predict(xPrueba))
    fpr, tpr, thresholds = roc_curve(yPrueba, bg.predict_proba(xPrueba)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Bagging (Área = %0.2f)' % areaBajoCurvaROC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio de Falsos Positivos')
    plt.ylabel('Ratio de Verdaderos Positivos')
    plt.legend(loc="lower right")
    plt.savefig('AreaBajoLaCurvaROC')
    plt.show()

    plot_confusion_matrix(matrizConfusion, classes=['Prueba Negativa', 'Prueba Positivo'],
                          title='Matriz de confusión Bagging')
    plt.savefig('cm_B.png')


def covidDiagnosis():
    print("*********************************** LEER DATASET ***********************************")
    dataframe = leerDataset()

    print("*********************************** RESUMEN DATAFRAME ***********************************")
    resumenDataframe(dataframe)

    print("*********************************** IDENTIFICAR NULOS ***********************************")
    identificarNulos(dataframe)

    print("*********************************** REVISAR VALORES FEATURES ***********************************")
    identificarValoresFeature(dataframe, "corona_result")
    identificarValoresFeature(dataframe, "test_indication")

    print("*********************************** LIMPIAR DATAFRAME ***********************************")
    aux = limpiezaDataframe(dataframe)
    dataframe = aux

    print("*********************************** REVISAR VALORES FEATURES ***********************************")
    identificarValoresFeature(dataframe, "corona_result")
    identificarValoresFeature(dataframe, "test_indication")

    print("*********************************** APLICAR ONE HOT ENCODING ***********************************")
    aux = tecnicaOneHotEncoding(
        dataframe, "age_60_and_above", "Age 60 At Least")
    dataframe = aux
    aux = tecnicaOneHotEncoding(dataframe, "gender", "Gender")
    dataframe = aux
    aux = tecnicaOneHotEncoding(
        dataframe, "test_indication", "Test Indication")
    dataframe = aux

    print("*********************************** RESUMEN DATAFRAME ***********************************")
    resumenDataframe(dataframe)

    print("*********************************** IDENTIFICAR NULOS ***********************************")
    identificarNulos(dataframe)

    print("*********************************** SEPARAR DATAFRAMES Y, X ***********************************")
    aux = separarDataframe(dataframe)
    Y = aux[0]
    X = aux[1]

    print("*********************************** MOSTRAR DATAFRAMES Y, X ***********************************")
    resumenDataframe(Y)
    resumenDataframe(X)

    print("*********************************** APLICAR SUPPORT VECTOR MACHINE ***********************************")
    supportVectorMachine(Y, X)

    print("*********************************** APLICAR REGRESION LOGISTICA ***********************************")
    regresionLogistica(Y, X)

    print("*********************************** APLICAR GRADIENT BOOSTING ***********************************")
    gradientBoosting(Y, X)

    print("*********************************** APLICAR BAGGING ***********************************")
    bagging(Y, X)
