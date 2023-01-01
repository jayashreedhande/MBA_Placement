from flask import Flask , jsonify, redirect,render_template,request
import config2
import json
import sklearn
import numpy as np
import pandas as pd
from utils import MBAPlacement

app = Flask(__name__)

@app.route('/')
def  hello_flask():

    # print("Welcome to MBA Placement Prediction")
    return render_template('home.html')

@app.route('/predicted_status',methods = ['GET','POST'])
def prediction():
    if  request.method == 'POST':
        data  = request.form

        print('data :',data)

        mba_place = MBAPlacement(data)
        status = mba_place.get_placement_prediction()
        print('status :',status)
        # return str(status)
        return render_template('home2.html', data=status)


if __name__ =="__main__":
    app.run(host='0.0.0.0',port = 5050)