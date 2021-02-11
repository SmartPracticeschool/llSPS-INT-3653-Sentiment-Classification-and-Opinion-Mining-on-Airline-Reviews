from flask import render_template, Flask, request
from keras.models import load_model
import pickle 
import re
import tensorflow as tf
global graph
graph = tf.get_default_graph()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
with open('tweet.pkl','rb') as file:
    cv=pickle.load(file)
cla = load_model('airline_predictions.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods = ['GET','POST'])
def page2():
    
    if request.method == 'POST':
        topic = request.form['review']
        topic=re.sub('[^a-zA-Z]',' ', topic)
        topic=topic.lower()
        topic=topic.split()
        topic=[ps.stem(word) for word in topic if not word in set(stopwords.words('english'))]
        topic = ' '.join(topic)
    with graph.as_default():
        pred = cla.predict_classes(cv.transform([topic]))
    
    if(pred==0):
        topic = "Negative Tweet"
    elif(pred==1):            
        topic = "Neutral Tweet"
    else:
        topic = "Positive Tweet"
    return render_template('index.html',label = topic)
        
if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
