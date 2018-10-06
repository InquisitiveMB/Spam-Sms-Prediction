#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('TkAgg')
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.learning_curve import learning_curve
import warnings
import string
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

import tkinter as tk
root = tk.Tk()
root.geometry("1000x600")

w =tk.Canvas(root, width=5000, height=1000)
w.place(x=0, y=20)
w.create_line(0, 300, 5000, 300, fill="lightblue") 
#Preprocessing and Exploring the Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data.head()
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2" : "text", "v1":"label"})
data['label'].value_counts()

ham_words = ''
spam_words = ''
for val in data[data['label'] == 'spam'].text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        spam_words = spam_words + words + ' '
        
for val in data[data['label'] == 'ham'].text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '
        
spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)

#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#ham Word cloud
plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

data = data.replace(['ham','spam'],[0, 1]) 
data.head(10)

#Removing Stopwords from the messages
def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)


def find(p):
    if p == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")
        
def FIND(p):
    if p == 1:
        return "Message is SPAM"
    else:
        return "Message is NOT Spam"
        
        
def model_assessment(y_test,predicted_class):
    print('confusion matrix')
    print(confusion_matrix(y_test,predicted_class))
    #cmap = ListedColormap(['b', 'y', 'r', 'g'])  
    plt.matshow(confusion_matrix(y_test, predicted_class), cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    
    
        
data['text'] = data['text'].apply(text_process)
data.head()
text = pd.DataFrame(data['text'])
label = pd.DataFrame(data['label'])

#Converting words to vectors using TFIDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['text'])
vectors.shape
features = vectors

#Splitting into training and test set¶
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
dtcFit = dtc.fit(X_train, y_train)
predDTC=dtc.predict(X_test)
accDTC=accuracy_score(y_test , predDTC)
cmDTC = confusion_matrix(y_test, predDTC)
model_assessment(y_test, predDTC)

text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integers = vectorizer.transform(text)
p = dtc.predict(integers)[0]
find(p)

msgDTC = tk.Message(root, text = ' ', aspect=500)
msgDTC.place(x=630,y=250)
def retrieve_DTC():
    inputvalDTC = textBox.get("1.0","end-1c")
    integersDTC = vectorizer.transform([inputvalDTC])
    p = dtc.predict(integersDTC)[0] 
    messageDTC = FIND(p)
    msgDTC.configure(text=messageDTC)
    msgDTC.config(bg='lightgreen', font=('times', 15, 'italic'))
    print (messageDTC)
    
    
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=31, random_state=111)
rfcFit = rfc.fit(X_train, y_train)
predRFC=rfc.predict(X_test)
cmRFC = confusion_matrix(y_test, predRFC)
accRFC=accuracy_score(y_test , predRFC)
model_assessment(y_test, predRFC)

text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integers = vectorizer.transform(text)
p = rfc.predict(integers)[0]
find(p)

msgRFC = tk.Message(root, text = ' ', aspect=500)
msgRFC.place(x=900, y=250)
def retrieve_RFC():
    inputvalRFC = textBox.get("1.0","end-1c")
    integersRFC = vectorizer.transform([inputvalRFC])
    p = rfc.predict(integersRFC)[0] 
    messageRFC = FIND(p)
    msgRFC.configure(text=messageRFC)
    msgRFC.config(bg='lightgreen', font=('times', 15, 'italic'))
    print (messageRFC)
    
#naive_bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.2)
nbFit = mnb.fit(X_train, y_train)
predNB=mnb.predict(X_test)
accMNB=accuracy_score(y_test , predNB)
cmNB = confusion_matrix(y_test,predNB)
model_assessment(y_test, predNB)

text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integersNB = vectorizer.transform(text)
p = mnb.predict(integersNB)[0]
find(p)

"""  """

root.title("Spam SMS Prediction")
w = tk.Label(root, font = "Verdana 20 bold", text="Spam SMS Prediction Phase")
w.place(x=700, y=15, anchor="center")

image = Image.open("sms.png")
image = image.resize((150, 150), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
label =tk.Label(image=photo)
label.image = photo # keep a reference!
label.place(x=50,y=30)

msgNB = tk.Message(root, text = ' ', aspect=500)
msgNB.place(x=355,y=250)
def retrieve_NB():
    inputval = textBox.get("1.0","end-1c")
    integersNB = vectorizer.transform([inputval])
    p = mnb.predict(integersNB)[0]
    messageNB = FIND(p)
    msgNB.configure(text=messageNB)
    msgNB.config(bg='lightgreen', font=('times', 15, 'italic'))
    print (messageNB)
    
textBox = tk.Text(root, height=5, width=50)
textBox.place(x=500,y=60) 
w = tk.Label(root, font = "Verdana 10 bold", text="Predict Against Various Classification Algorithms")
w.place(x=700, y=170, anchor="center")

pred_scores_word_vectors = (('MNB', [accMNB]), ('DTC', [accDTC]), ('RFC', [accRFC]))
#predictions = pd.DataFrame({'col':pred_scores_word_vectors})
predictions = pd.DataFrame.from_items(pred_scores_word_vectors,orient='index', columns=['Score'])
predictions

predictions.plot(kind='bar', ylim=(0.85,1.0), figsize=(9,6), align='center', colormap="Accent")
plt.xticks(np.arange(6), predictions.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#predictions = predictions[['index', 'Score']].groupby('index').sum()
def retrieve_Accuracy():
    figure1 = plt.Figure(figsize=(5,4), dpi=80)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().place(x=950,y=390)
    predictions.plot(kind='bar', ylim=(0.85,1.0), legend=True, ax=ax1)
    ax1.set_title('Classification Vs. Accuracy')
    
def retrieve_ConfusionMatrix():
    image = Image.open("NB.png")
    image = image.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label =tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=140,y=390)
    
    image = Image.open("NBP.png")
    image = image.resize((200, 100), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=140,y=600)
     
    w = tk.Label(root, font = "Verdana 10 bold", text="Naive Bayes")
    w.place(x=250, y=720, anchor="center")
    
    image = Image.open("DTC.png")
    image = image.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label =tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=400,y=390)
    
    image = Image.open("DTCP.png")
    image = image.resize((200, 100), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=400,y=600)
    
    w = tk.Label(root, font = "Verdana 10 bold", text="Decision Tree")
    w.place(x=510, y=720, anchor="center")
    
    image = Image.open("RFC.png")
    image = image.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label =tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=700,y=390)
    
    image = Image.open("RFCP.png")
    image = image.resize((200, 100), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.place(x=700,y=600)
    
    w = tk.Label(root, font = "Verdana 10 bold", text="Random Forest")
    w.place(x=810, y=720, anchor="center")
    

w = tk.Label(root, font = "Verdana 20 bold", text="Training & Testing Phase")
#w.pack()
w.place(x=700, y=350, anchor="center")

buttonCommitNB = tk.Button(root, height=2, width=20, bg= 'lightblue', bd=5, text="Naive Bayes Classifier", 
                      command=retrieve_NB)
buttonCommitNB.place(x=355,y=190)

buttonCommitDTC = tk.Button(root, height=2, width=20, bg= 'lightblue', bd=5, text="Decision Tree Classifier", 
                      command=retrieve_DTC)
buttonCommitDTC.place(x=625,y=190)

buttonCommitRFC = tk.Button(root, height=2, width=20, bg= 'lightblue', bd=5, text="Random Forest Classifier", 
                      command=retrieve_RFC)
buttonCommitRFC.place(x=895,y=190)

buttonAccurcy = tk.Button(root, height=2, width=20, bg= 'lightblue', bd=5, text="See the Accuracy Score", 
                      command=retrieve_Accuracy)
buttonAccurcy.place(x=1030,y=330) 

buttonConfusion = tk.Button(root, height=2, width=15, bg= 'lightblue', bd=5, text="Confusion Matrix", 
                      command=retrieve_ConfusionMatrix)
buttonConfusion.place(x=10,y=500) 
 
tk.mainloop()

"""
#SVM
from sklearn.svm import SVC
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(X_train, y_train)
predSVM=svc.predict(X_test)
accSVM=accuracy_score(y_test , predSVM)
model_assessment(y_test, predSVM)
text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integers = vectorizer.transform(text)
p = svc.predict(integers)[0]
find(p)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(solver='liblinear', penalty='l1')
lrc.fit(X_train, y_train)
predLR=lrc.predict(X_test)
accLR=accuracy_score(y_test , predLR)
model_assessment(y_test, predLR)
text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integers = vectorizer.transform(text)
p = lrc.predict(integers)[0]
find(p)        


#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=49)
knc.fit(X_train, y_train)
predKNN=knc.predict(X_test)
accKNN=accuracy_score(y_test , predKNN)
model_assessment(y_test, predKNN)
text = ["Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app"]
integers = vectorizer.transform(text)
p = knc.predict(integers)[0]
find(p)

"""

