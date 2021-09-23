import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


datasetTrue = pd.read_csv("True.csv")
datasetFalse = pd.read_csv("Fake.csv")

datasetTrue["label"] = 1
datasetFalse["label"] = 0
dataSet = pd.concat([datasetTrue , datasetFalse] , ignore_index=True)

del dataSet["date"]
del dataSet["subject"]

titleAndTextColumn = dataSet.iloc[:,0:2]
labelColumn = dataSet.iloc[:,2]

labelDF = pd.DataFrame(data = labelColumn)


nltk.download('stopwords')
stopwords = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stopwords.update(punctuation)
porterStemmer = PorterStemmer()

LINECOUNT = len(dataSet)

def cleanText(text):
    newText = []
    text = text.lower()
    text = re.sub('[^a-zA-Z]' , " " , text)
    
    for i in text.split():
        if i.strip() not in stopwords:
            newText.append(i.strip())
    
    return " ".join(newText)

array = []


for i in range(0 , LINECOUNT):
    titleText = cleanText(dataSet["title"][i])
    textText = cleanText(dataSet["text"][i])
    text = titleText + " " + textText
    array.append(text)

countVectorizer = CountVectorizer(max_features=5000)

X = countVectorizer.fit_transform(array).toarray()
Y = labelDF.values
  
x_train, x_test , y_train , y_test = train_test_split(X , Y ,test_size=0.25 , random_state=0)

from keras.models import Sequential
from keras.layers import Dense

EPOCH = 5

classifier = Sequential()
classifier.add(Dense(1500 , activation = "relu" , input_dim = 5000 , name = "input-layer"))
classifier.add(Dense(1500 , activation = "relu"  , name = "hidder-layer" ))
classifier.add(Dense(1 , activation = "sigmoid" , name = "output-layer"))

classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics=['binary_accuracy'] )

classifier.fit(x_train , y_train , epochs=EPOCH)

predictions = classifier.predict(x_test)
predictions = (predictions > 0.5)


confusionMatrix = confusion_matrix(y_test , predictions)
print(confusionMatrix)

truePred  = confusionMatrix[0,0] + confusionMatrix[1,1]
falsePred = confusionMatrix[0,1] + confusionMatrix [1,0]

print("")
print("Başarı Yüzdesi : %" , (truePred / (truePred + falsePred)) * 100)

pieChart = np.array([truePred , falsePred])
label = ["True Predictions" , "False Predictions"]
explode = [0.2 , 0.0]


plt.pie(pieChart , labels= label , shadow=True , explode = explode)
plt.show()








               




    
    
    

    


                    
            
                   
    
    
