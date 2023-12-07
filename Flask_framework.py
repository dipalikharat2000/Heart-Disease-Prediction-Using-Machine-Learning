import pandas as pd
data = pd.read_csv("C:/Users/Admin/Downloads/Heart_disease _pred-1 (5)/Heart_disease _pred/heart (1) .csv")
data.isnull().sum()
data_dup = data.duplicated().any()
data_dup
data = data.drop_duplicates()
data_dup = data.duplicated().any()
data_dup

cate_val = []
cont_val = []
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
        
cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns = cate_val,drop_first=True)

from sklearn.preprocessing import StandardScaler

st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


X = data.drop('target',axis=1)

y = data['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)

X=data.drop('target',axis=1)
y=data['target']
from sklearn import svm
svm=svm.SVC()
svm.fit(X,y)


import pandas as pd

new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])


# p = svm.predict(new_data)
# if p[0]==0:
#     print("No Disease")
# else:
#     print("Disease")
   
  
  
import flask
from flask import *
import requests    
import joblib
from flask import Flask, render_template, redirect, url_for, request, session, flash, app, Blueprint, jsonify

#@app.route('/autocomplete',methods=['GET'])
app = Flask(__name__)

@app.route('/')
def input():
    return render_template('login page.html')

@app.route('/main', methods=['POST'])
def main():
    if request.method == 'POST':

        return render_template('HTML.html')
   

# Create other routes here.
# host/passing will be the website link
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = pd.read_csv("C:/Users/Admin/Downloads/Heart_disease _pred-1 (5)/Heart_disease _pred/heart (1) .csv")

        X=data.drop('target',axis=1)
        y=data['target']
        
        from sklearn import svm
        svm=svm.SVC()
        svm.fit(X,y)

        
        print("request get succesfully")
        age = request.form.get("age")
        
        
        print("--------------------------------")
        print(age)
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        print(trestbps)
        chol = request.form.get('chol')
        print(chol)
        fbs = request.form.get('fbs')
        print(fbs)
        restecg = request.form.get('restecg')
        print(restecg)
        thalach = request.form.get('thalach')
        print(thalach)
        exang = request.form.get('exang')
        print(exang)
        slope = request.form.get('slope')
        print(slope)
        ca = request.form.get('ca')
        print(ca)
        old = request.form.get('old')
        print(old)
        thal = request.form.get('thal')
        print(thal)
        
        print(sex)
        new_data = pd.DataFrame({
            'age':age,
            'sex':sex,
            'cp':cp,
            'trestbps':trestbps,
            'chol':chol,
            'fbs':fbs,
            'restecg':restecg,
            'thalach':thalach,
            'exang':exang,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal,    
        },index=[0])


      
        # print(new_data)
        # joblib.dump(svm,'model_joblib_heart')
        # model = joblib.load('model_joblib_heart')

        output=svm.predict(new_data)

        output=int(output)
        if output==0:
            result="No Disease"
        else:
            result="Disease"
   
        print(result)
        return render_template("HTML.html", result=result)
 
 
# main route to start with
if __name__ == '__main__':
    app.run(debug=True)

