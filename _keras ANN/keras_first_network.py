# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import keras 
start_time = time.time()
# %%

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# %%

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
print(type(dataset))
X = dataset[:,0:8]
y = dataset[:,8]


# %%

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# %%
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# %%
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)


# %%
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# %%
# make probability predictions with the model
predictions = model.predict(X)
# round predictions 
rounded = [round(x[0]) for x in predictions]

# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), rounded[i], y[i]))


# %%
print('duration to run the code is', time.time() - start_time)


# %%



