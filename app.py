from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import pickle
#from flask_cors import CORS,cross_origin
model = pickle.load(open( "finalized_model.sav", "rb"))


model1 = pickle.load(open( "fl.sav", "rb"))
model2 = pickle.load(open( "f2.sav", "rb"))
model3 = pickle.load(open( "f3.sav", "rb"))
model4 = pickle.load(open( "f4.sav", "rb"))


app = Flask(__name__)
#app.config['CORS_HEADERS'] = 'Content-Type'
#cors = CORS(app)
@app.errorhandler(404)

def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/get_prediction", methods=['POST','OPTIONS'])
@cross_origin()
def get_prediction():
	if not request.json:
		abort(400)
	
	content = request.get_json()
		
	df_new = pd.DataFrame(data = request.get_json(), index=[0])
	
	#df_new = json.load(request.json)
	#cols=["CONSOLE","RATING","CRITICS_POINTS","CATEGORY","YEAR","PUBLISHER","USER_POINTS"]
	#df_new.columns=["PUBLISHER","CRITICS_POINTS","CATEGORY","YEAR","RATING","CONSOLE","USER_POINTS"]
	#df_new = df_new.astype({'YEAR':'int64','USER_POINTS':'float64','CRITICS_POINTS':'float64'},copy=True)
	#df_new = df_new[["RATING","PUBLISHER","CATEGORY","CONSOLE","USER_POINTS","CRITICS_POINTS","YEAR",]]
	
	#df_new = pd.DataFrame.from_dict(data = dict1.values())                               
	#df_new = df_new.T
	#df_new.columns=["ID","RATING","PUBLISHER","CATEGORY","CONSOLE","USER_POINTS","CRITICS_POINTS","YEAR"]
	
	#df_new = df_new.astype({'YEAR':'int64','USER_POINTS':'float64','CRITICS_POINTS':'float64'},copy=True)
	#df_new = df_new[["RATING","PUBLISHER","CATEGORY","CONSOLE","USER_POINTS","CRITICS_POINTS","YEAR",]]
	
	#df_new = pd.DataFrame.from_dict(data = dict1.values())
	#df_new = pd.DataFrame.from_dict(data = request.json)	
	#df_new = df_new.T
	#df_new.columns=["ID","RATING","PUBLISHER","CATEGORY","CONSOLE","USER_POINTS","CRITICS_POINTS","YEAR"]
    #
	df = df_new.copy()
	#
	categ1 = pd.DataFrame()
	categ1['CONSOLE_LB'] = model1.transform(df['CONSOLE'])
	categ1['CATEGORY_LB'] = model2.transform(df['CATEGORY'])
	categ1['PUBLISHER_LB'] = model3.transform(df['PUBLISHER'])
	categ1['RATING_LB'] = model4.transform(df['RATING'])
	#
	#
	new_df = pd.concat([categ1,df[['YEAR','CRITICS_POINTS','USER_POINTS']]],axis=1)
	result = model.predict(new_df)[0]
	
	print(result)
	
	#df = df[cols]
	#return jsonify({'result': model.predict(new_df)[0]}), 201
	#return print(result)
	
	a = str(result)[1:7]
	return a

if __name__ == "__main__":
    app.run()
