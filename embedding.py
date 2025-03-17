"""
Recurrent neural network
In recurrent neural network,multiple data can be sent at once ,because of feedback loop present in every neuron at 
hiddent layer which will send the data to itsef and other neurons in hidden layer and the data are sent based on timestamp

In RNNs, embedding layers are commonly used to convert sequences of tokens into vectors before feeding them 
into the recurrent layers for learning temporal or sequential patterns.

ANNs are suitable for problems where each input is independent and there are no temporal dependencies between inputs.
RNNs, with their feedback loops, are ideal for tasks where order and context matter, such as sequential data 
(time series, text, etc.), because they can "remember" information from earlier inputs.
"""
from tensorflow.keras.preprocessing.text import one_hot

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

## Define the vocublary size
voc_size = 10000

## one hot representation
one_hot_repr = [one_hot(words,voc_size) for words in sent]
# print(one_hot_repr)

## word embedding representation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences 
"""
Pad sequences are used to ensure that all sentences in a document list have the same length by 
padding or truncating them to a fixed number of words. This allows each sentence to be processed
 consistently across different timestamps.
"""
from tensorflow.keras.models import Sequential
import numpy as np

sent_length=8
embedded_docs=pad_sequences(one_hot_repr,padding="pre",maxlen=sent_length) #filling the documents with 0's pre/post
# print(embedded_docs)

## feature reprsentation

dim=10
model =Sequential()    
model.add(Embedding(voc_size,dim,input_length=sent_length)) #for creating embedding layer-required parameter is voc_size,dimension and input length
model.compile('adam','mse')#optimiser-adam and loss_function-mse
"""
Loss function ('mse'): The loss function measures the difference between the model's predictions and the actual values (ground truth). 
In this case, 'mse' stands for Mean Squared Error
"""
model.build(input_shape=(None, sent_length))  # Add this line to build the model

# print(model.summary()) 

print(model.predict(embedded_docs))



