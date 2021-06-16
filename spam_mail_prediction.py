

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm  import LinearSVC
from sklearn.metrics import accuracy_score

raw_mail_data=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
#replace the null valueswith null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.shape

mail_data.head()

#label spam mail as 0 ,non spam mail as 1
mail_data.loc[mail_data['v1']=='spam','v1',]=0
mail_data.loc[mail_data['v1']=='ham','v1',]=1

x=mail_data['v2']
y=mail_data['v1']

print(x)
print('...................')
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=3)

#transform the text data to feature vectors that can be used as  input to the svm model using TidVectorizer
#conert the text to lower case
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='true')
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.fit_transform(x_test)

#convert t_train nd y_test as integers values
y_train=y_train.astype('int')
y_test=y_train.astype('int')

#training the svm model with training mode
model=LinearSVC()
model.fit(x_train_features,y_train)

prediction_on_training_data=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)

print("accuracy on training data:",accuracy_on_training_data)

prediction_on_test_data=model.predict(x_test_features)
accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)

print("accuracy on test dta:",accuracy_on_test_data)

#prediction
input_mail=("")
#convert text to feature vectors
input_mail_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_mail_features)
print(prediction)
if(prediction[0]==1):
    print("ham mail")
else:
    print("spam mail")





