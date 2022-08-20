from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():
    Nationality = request.form.get('Nationality')
    Age = request.form.get('Age')
    AverageLeadTime = request.form.get('AverageLeadTime')
    LodgingRevenue = request.form.get('LodgingRevenue')
    OtherRevenue = request.form.get('OtherRevenue')
    RoomNights = request.form.get('RoomNights')
    DaysSinceFirstStay = request.form.get('DaysSinceFirstStay')
    SRHighFloor = request.form.get('SRHighFloor')
    SRCrib = request.form.get('SRCrib')
    SRTwinBed = request.form.get('SRTwinBed')

    result = model.predict([[Nationality,Age,AverageLeadTime,LodgingRevenue,OtherRevenue,RoomNights,DaysSinceFirstStay,SRHighFloor,SRCrib,SRTwinBed]])[0]

    if result=='Yes':
        return render_template('index.html',label=1)
    else:
        return render_template('index.html',label=-1)

if __name__=='__main__':
    app.run(debug=False)
