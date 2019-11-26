from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence
import numpy as np

model = load_model('model.h5')

sent = "I have never seen this kind of film, very bad story. all time i got bored"

words = set(text_to_word_sequence(sent))
data = one_hot(sent, round(len(words) * 1.3))
data = np.array(data)
data  = np.expand_dims(data,axis=0)
data = sequence.pad_sequences(data,maxlen=2000)

print(model.predict([data]))

