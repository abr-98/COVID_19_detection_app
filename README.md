# COVID_19_detection_app


Docker and Flask based application to detect COVID-19 patient using X-Ray and symptoms

The project uses three models: 

1. A CNN model for studying the X-Ray to predict if the patient has COVID-19 or not

2. An XGBoost model which takes in the prediction probability of the X-ray predction by the CNN-model and the persons age and gender to predict the risk for the patient

3. A Gradient Boosting model which takes in the symptoms of the patient and predicts the risk of the patient

The final result is obtained by averaging the prediction of both the models.


Finally, Flask and Docker have been used to host the model. 

**Requirements**
numpy==1.18.5
matplotlib
opencv-python-headless
DateTime
sklearn
scikit-learn
pandas
seaborn
tensorflow
Werkzeug==0.14.1
Flask-WTF
WTForms
xgboost





