from flask import Flask , render_template 
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

import pyswarms as ps


app = Flask(__name__)

WINDOWS = 7 # jumlah hari sebelumnya untuk memprediksi kasus hari ini

# fungsi mengubah timeseries menjadi kolom regresi
def create_window(data: pd.DataFrame, window = 7):
    series = data['new_cases']
    
    X = []
    y = []
    for i in range(len(series)-window):
        X.append(series[i:i+window].tolist())
        y.append(series[i+window])
    
    X = pd.DataFrame(X,
                     columns = ['t{}'.format(i+1) for i in range(window)])
    
    return X, y

baru = pd.read_csv('newindodailycase.csv')
X, y = create_window(baru, window=WINDOWS)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    shuffle = False)

filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))
y_pred = np.round(model.predict(X_test),0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def datas():
    df = pd.read_csv('newindodailycase.csv', header=0)
    dt = list(df.values)
    return render_template("data.html", data=dt )

@app.route("/graphic")
def graphic():
    a = mean_absolute_percentage_error(y_test, y_pred)
    result = pd.DataFrame()
    result['predicted'] = y_pred
    result.index = list(baru.date[-len(result):])
    b = result.index
    # d = str(b)
    c = y_pred
    # print(d)
    return render_template("predik.html", data=a,x=c,y=b)

@app.route("/kasus")
def kasus():
    result = pd.DataFrame()
    result['predicted'] = y_pred
    result['actual'] = y_test
    result.index = baru.date[-len(result):]
    a = result.index
    result['data'] = a
    dt = list(result.values)
    return render_template("hasil.html", hasil = dt )

if __name__ == "__main__":
    app.run(debug=True)
