import re
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Input, Lambda, Dense

model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def preprocess(text):
  text = text.lower()
  text = re.sub(r"[^а-я ]", "", text)
  return text

words = []
synonyms = []

with open("data.txt") as f:
  for line in f:
    word, synonym = line.strip().split(" - ")
    words.append(preprocess(word))
    synonyms.append(preprocess(synonym))

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(words + synonyms, target_vocab_size=2 ** 15)

encoded_words = tokenizer.encode(words)
encoded_synonyms = tokenizer.encode(synonyms)

model = tf.keras.Sequential([
  Input(shape=[], dtype=tf.string),
  Lambda(lambda x: tokenizer.encode(x)),
  model,
  Dense(tokenizer.vocab_size)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.001))

model.fit(words, encoded_synonyms, epochs=10)

word = "большой"
encoded = tokenizer.encode(word)
prediction = model.predict(encoded)
synonym = tokenizer.decode(prediction[0])
print(synonym)
