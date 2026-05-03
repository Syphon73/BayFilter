#from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class NaiveBayes:

    def fit(self,X,labels):
        # STEP 1
        labels = np.array(labels)

        X_spam  = X[labels == 0]
        X_nonspam = X[labels == 1]

        #sum WC per class
        spamClassWC = X_spam.sum(axis=0)
        nonspamClassWC = X_nonspam.sum(axis=0)

        #total words per class
        spamWords = spamClassWC.sum()
        nonspamWords = nonspamClassWC.sum()
        total = spamWords + nonspamWords

        #prior probabilities
        self.P_spam = spamWords/total
        self.P_nonspam = nonspamWords/total

        #STEP 2, 3
        V = X.shape[1] 
        # +1 is laplace soothing
        self.P_spamWClass = (spamClassWC + 1) / (spamWords + V)
        self.P_nonspamWClass = (nonspamClassWC + 1) / (nonspamWords + V)

    def predict(self,X):
        #STEP 4
        X = X.toarray()
        log_P_spamWClass = np.log(self.P_spamWClass)
        log_P_nonspamWClass = np.log(self.P_nonspamWClass)

        #STEP 5
        spam_score  = np.log(self.P_spam) + X @ log_P_spamWClass.T
        nonspam_score = np.log(self.P_nonspam) + X @ log_P_nonspamWClass.T

        return np.where(spam_score > nonspam_score, 0, 1)



#split the data into msg and category lists
df = pd.read_csv('spam.csv')
msgs = df['Message'].tolist()
labels = df['Category'].map({'ham':1, 'spam':0}).tolist()

#convert text -> numbers {vocab + wc}
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(msgs)

#split into training and testing data {matrix for P(wc|class) = count(w,class) + 1 / vocab + words in class}
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.2,
    random_state=42
) 


model = NaiveBayes()
model.fit(X_train, y_train)

#probablity = model.predict_proba(X_test)
#print(probablity)

new = ["at 9pm today", "Free rewards"]
newX = vectorizer.transform(new)

predictions = model.predict(newX)

#for msg, pred in zip(new_messages, predictions):
#   print(f"{msg!r}  →  {'Spam' if pred == 1 else 'Not Spam'}")

for msg, pred in zip(new, predictions):
    print(f"{msg!r} : {'Spam' if pred == 0 else 'Not Spam'}")
