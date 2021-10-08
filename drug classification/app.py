from flask import Flask,request,render_template
import pickle
import numpy as np
drug_model = pickle.load(open('Random_for_drug_classification.pkl','rb'))
app  = Flask(__name__)
@app.route('/')
def root():
    return render_template("base.html")
# @app.route('/index')
# def test():
#     return render_template("index.html")
@app.route('/drug_classification',methods=['POST','GET'])
def drug_classification():
    pr=""
    if request.method=='POST':
        values = [float(x) for x in request.form.values()]
        values= [np.array(values)]
        print(values)
        pr = drug_model.predict(values)
        print(pr)
    return render_template('predict.html',value=pr)
if __name__ == "__main__":
    app.run(debug=True)