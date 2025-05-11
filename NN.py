# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Credit to: https://www.kaggle.com/code/anshchaurasiya/skin-cancer-detection
#OS libs

#Data handling tools
import seaborn as sns
sns.set_style('whitegrid')

#Deep learning libs
import tensorflow as tf
from tensorflow.keras.layers import Sequential
from tensorflow.keras.layers import Dense , Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers


img_shape = (-1, -1, -1)
num_class = -1

base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top = False, weights = 'imagenet', input_shape = img_shape, pooling= 'max')
model = Sequential([
    base_model,
    BatchNormalization(axis= -1 , momentum= 0.99 , epsilon= 0.001),
    Dense(256, kernel_regularizer = regularizers.l2(l= 0.016), activity_regularizer = regularizers.l1(0.006), bias_regularizer= regularizers.l1(0.006) , activation = 'relu'),
    Dropout(rate= 0.4 , seed = 75),
    Dense(num_class, activation = 'softmax')
])

model.compile(Adamax(learning_rate = 0.001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()