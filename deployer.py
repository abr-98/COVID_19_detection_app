from flask import Flask,request,render_template,redirect
import pickle
from werkzeug import secure_filename
import numpy as np
import flask
import tensorflow as tf
import cv2
import os
from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__, template_folder='templates')
#app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response


model_image=tf.keras.models.load_model('Covid_19_detection.h5')
with open('./model_risk.pkl', 'rb') as model_p:
    model_risk=pickle.load(model_p)

with open('./model_symptoms.pkl', 'rb') as model_p:
    model_symptoms=pickle.load(model_p)

@app.route('/upload')  
def upload():  
    return render_template('land_page.html') 

@app.route('/predicter', methods=['POST'])
def predicter():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            n=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            image=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images_final=np.array(image)/255
            pred_1=model_image.predict(np.array([images_final]))
            
            
            if (np.round(pred_1[0][0]))>0.5:
                msg="COVID-19 Detected. Proceed for Risk Prediction"
                return render_template("result_pos.html", message = msg)
            
            else:
                msg="No issues detected."
                return render_template("result_neg.html", message = msg)
                
                
        
            #return sendResponse(str(np.round(pred_1[0][0])))

            
            #return redirect('/predict')
@app.route('/re',methods=['POST'])
def re():
    if request.method == 'POST':
        return render_template('form.html')

@app.route('/getinfo',methods=['POST'])
def getinfo():
    
    fields=["name","age","gender","has_pneumonia","has_stiff","has_effusion","has_tired","has_thirst","has_appetite","has_pleural","has_symptom","has_weak","has_discomfort","has_cough","has_ache","has_rigor","has_cold","has_sore","has_pharyn","has_runny","has_respiratory","has_viral"]
    record=[]
    feat=[]
    if request.method=='POST':
        for i in range(len(fields)):
            if request.form.get(fields[i])=="":
                flash('fields missing')
                return redirect(request.url)
                
            record.append(request.form.get(fields[i]))
        feat.append(record[0])
        
        if record[2].upper()=='M' or record[i].upper()=='MALE':
            feat.append(1)
        else:
            feat.append(0)
        feat.append(record[1])
        i=3
        while (i<len(fields)):
            if record[i].upper()=='Y' or record[i].upper()=='YES':
                feat.append(1)
            else:
                feat.append(0)
            i+=1
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            image=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            images_final=np.array(image)/255
            pred_1=model_image.predict(np.array([images_final]))
            pred=pred_1[0][0]
        else:
            flash('image missing')
            return redirect(request.url)
        
        train_1=np.array([feat[1],float(feat[2]),pred])
        train_2=[]
        train_2.append(int(feat[1]))
        train_2.append(float(feat[2]))
        i=3
        while i<len(feat):
            train_2.append(int(feat[i]))
            i+=1
        train_2=np.array(train_2)
        print(train_1)
        print(train_2)
        test=[1,65.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        test=np.array(test)
        pred_risks=model_risk.predict(np.array([train_1]))
        pred_symptoms=model_symptoms.predict(np.array([test]))
        
        overall_pred=(float(pred_risks[0])+float(pred_symptoms[0]))/2.0
        if overall_pred>0.5:
            msg="Patient: "+record[0]+". Patient Status: Risky. Acute Symptoms detected. Advised to take immediate Medical Help."
        else:
            msg="Patient: "+record[0]+". Patient Status: Low Risk. Mild Symptoms. Nothing to worry about. "
        
        
        
        return render_template("Results_Final.html", message = msg)
        
if __name__ == "__main__":
    app.run()