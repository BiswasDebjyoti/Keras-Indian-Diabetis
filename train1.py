# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima_diabetis.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
#print len(X)
x_train,x_test= train_test_split(X,test_size=0.1)
y_train,y_test= train_test_split(Y,test_size=0.1)

# create model
model = Sequential()
model.add(Dense(units=150, input_shape=(8,)))
model.add(Activation('relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model
model.fit(x_train, y_train, epochs=20, batch_size=10)#Batchsize changed from 10 to 100
# evaluate the model
scores = model.evaluate(x_test, y_test)
#print scores
print("\n\n\t%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#classes =model.predict(x_test)
#print classes

