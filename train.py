from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import keras
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima_diabetis.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]

Y = dataset[:,8]
x_train,x_test= train_test_split(X,test_size=0.2)
y_train,y_test= train_test_split(Y,test_size=0.2)

#print len(test)
#12, input_dim=8, activation='relu
model = Sequential()
#model.add(Flatten())
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=150, batch_size=10)

classes =model.predict(x_test,batch_size=128)
print classes


