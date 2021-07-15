from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
file= open('model.pkl','rb')
reg= pickle.load(file)
file.close()
@app.route('/',methods=["GET","POST"])
def Home():
    if request.method == 'POST':
        mydict= request.form
        hours= float(mydict['hours'])
        Hour = np.array([[hours]])
        Result = reg.predict(Hour)
        print(Result)
        return render_template('show.html',Result=Result)
    return render_template('index.html')
    # return 'Hello world' + str(Result)



if __name__=="__main__":
    app.run(debug=True)