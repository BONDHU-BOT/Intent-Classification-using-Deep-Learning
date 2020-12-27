# Intent Classification using Deep Learning

## 1. Loading Data


```python
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
```

Dataset link: https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv


```python
Intent = "category"
Sentence = "text"
```


```python
def load_dataset(filename, Sentence, Intent):
  df = pd.read_csv(filename, names = [Sentence, Intent])
  intent = df[Intent]
  unique_intent = list(set(intent))
  sentences = list(df[Sentence])
  
  return (df, intent, unique_intent, sentences)
```


```python
df, intent, unique_intent, sentences = load_dataset("Dataset.csv", "text", "category")
```


```python
print(df.head(10))
```

                                                    text      category
    0                                               text      category
    1                     I am still waiting on my card?  card_arrival
    2  What can I do if my card still hasn't arrived ...  card_arrival
    3  I have been waiting over a week. Is the card s...  card_arrival
    4  Can I track my card while it is in the process...  card_arrival
    5  How do I know if I will get my card, or if it ...  card_arrival
    6                  When did you send me my new card?  card_arrival
    7       Do you have info about the card on delivery?  card_arrival
    8  What do I do if I still have not received my n...  card_arrival
    9       Does the package with my card have tracking?  card_arrival



```python
import seaborn as sns
import tkinter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sns.countplot(x=Intent, data=df)
```




    <AxesSubplot:xlabel='category', ylabel='count'>




```python
print(sentences[:5])
```

    ['text', 'I am still waiting on my card?', "What can I do if my card still hasn't arrived after 2 weeks?", 'I have been waiting over a week. Is the card still coming?', 'Can I track my card while it is in the process of delivery?']



```python
nltk.download("stopwords")
nltk.download("punkt")
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/shiningflash/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /home/shiningflash/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True



## 2. Data Cleaning


```python
#define stemmer
stemmer = LancasterStemmer()
```


```python
def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    words.append([i.lower() for i in w])
    
  return words  
```


```python
cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
```

    10004
    [['text'], ['i', 'am', 'still', 'waiting', 'on', 'my', 'card']]


## 3. Texts Tokenization


```python
def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token
```


```python
def max_length(words):
  return(len(max(words, key = len)))
```


```python
word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))
```

    Vocab Size = 2343 and Maximum length = 84



```python
def encoding_doc(token, words):
  return(token.texts_to_sequences(words))
```


```python
encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
```


```python
def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
```


```python
padded_doc = padding_doc(encoded_doc, max_length)
```


```python
padded_doc[:5]
```




    array([[1481,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0],
           [   1,   50,   64,  208,   30,    2,    6,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0],
           [  13,    8,    1,   10,   56,    2,    6,   64,  121,   11,  275,
             161,  453,  304,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0],
           [   1,   27,   52,  208,  305,    4,  240,    7,    5,    6,   64,
             454,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0],
           [   8,    1,  361,    2,    6,  183,    9,    7,   28,    5,  216,
              38,  362,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0]], dtype=int32)




```python
print("Shape of padded docs = ",padded_doc.shape)
```

    Shape of padded docs =  (10004, 84)



```python
#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
```


```python
output_tokenizer.word_index
```




    {'top_up_by_cash_or_cheque': 1,
     'card_delivery_estimate': 2,
     'visa_or_mastercard': 3,
     'wrong_amount_of_cash_received': 4,
     'beneficiary_not_allowed': 5,
     'card_payment_wrong_exchange_rate': 6,
     'verify_source_of_funds': 7,
     'top_up_by_card_charge': 8,
     'pin_blocked': 9,
     'automatic_top_up': 10,
     'change_pin': 11,
     'age_limit': 12,
     'edit_personal_details': 13,
     'declined_cash_withdrawal': 14,
     'card_linking': 15,
     'order_physical_card': 16,
     'fiat_currency_support': 17,
     'declined_transfer': 18,
     'topping_up_by_card': 19,
     'top_up_limits': 20,
     'why_verify_identity': 21,
     'declined_card_payment': 22,
     'cancel_transfer': 23,
     'transfer_into_account': 24,
     'wrong_exchange_rate_for_cash_withdrawal': 25,
     'top_up_failed': 26,
     'failed_transfer': 27,
     'card_about_to_expire': 28,
     'request_refund': 29,
     'lost_or_stolen_phone': 30,
     'reverted_card_payment': 31,
     'transaction_charged_twice': 32,
     'pending_top_up': 33,
     'balance_not_updated_after_cheque_or_cash_deposit': 34,
     'verify_top_up': 35,
     'passcode_forgotten': 36,
     'direct_debit_payment_not_recognised': 37,
     'exchange_rate': 38,
     'atm_support': 39,
     'transfer_fee_charged': 40,
     'verify_my_identity': 41,
     'virtual_card_not_working': 42,
     'card_not_working': 43,
     'top_up_reverted': 44,
     'card_acceptance': 45,
     'get_physical_card': 46,
     'pending_cash_withdrawal': 47,
     'lost_or_stolen_card': 48,
     'terminate_account': 49,
     'exchange_charge': 50,
     'activate_my_card': 51,
     'top_up_by_bank_transfer_charge': 52,
     'receiving_money': 53,
     'compromised_card': 54,
     'disposable_card_limits': 55,
     'getting_spare_card': 56,
     'get_disposable_virtual_card': 57,
     'cash_withdrawal_not_recognised': 58,
     'category': 59,
     'unable_to_verify_identity': 60,
     'supported_cards_and_currencies': 61,
     'card_payment_not_recognised': 62,
     'getting_virtual_card': 63,
     'pending_card_payment': 64,
     'pending_transfer': 65,
     'card_arrival': 66,
     'exchange_via_app': 67,
     'card_swallowed': 68,
     'balance_not_updated_after_bank_transfer': 69,
     'apple_pay_or_google_pay': 70,
     'country_support': 71,
     'transfer_timing': 72,
     'card_payment_fee_charged': 73,
     'cash_withdrawal_charge': 74,
     'extra_charge_on_statement': 75,
     'contactless_not_working': 76,
     'refund_not_showing_up': 77,
     'transfer_not_received_by_recipient': 78}




```python
encoded_output = encoding_doc(output_tokenizer, intent)
```


```python
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
```


```python
encoded_output.shape
```




    (10004, 1)




```python
def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))
```


```python
output_one_hot = one_hot(encoded_output)
```


```python
output_one_hot.shape
```




    (10004, 78)




```python
from sklearn.model_selection import train_test_split
```


```python
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)
```


```python
print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))
```

    Shape of train_X = (8003, 84) and train_Y = (8003, 78)
    Shape of val_X = (2001, 84) and val_Y = (2001, 78)


## 4. Bidirectional GRU 


```python
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(GRU(128)))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(78, activation = "softmax"))
  
  return model
```


```python
model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 84, 128)           299904    
    _________________________________________________________________
    bidirectional (Bidirectional (None, 256)               198144    
    _________________________________________________________________
    dense (Dense)                (None, 32)                8224      
    _________________________________________________________________
    dropout (Dropout)            (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 78)                2574      
    =================================================================
    Total params: 508,846
    Trainable params: 208,942
    Non-trainable params: 299,904
    _________________________________________________________________



```python
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
```


```python
hist = model.fit(train_X, train_Y,
                 epochs = 100,
                 batch_size = 32,
                 validation_data = (val_X, val_Y),
                 callbacks = [checkpoint])
```

    Epoch 1/100
    250/251 [============================>.] - ETA: 0s - loss: 4.3335 - accuracy: 0.0179
    Epoch 00001: val_loss improved from inf to 4.27058, saving model to model.h5
    251/251 [==============================] - 23s 93ms/step - loss: 4.3334 - accuracy: 0.0179 - val_loss: 4.2706 - val_accuracy: 0.0215
    Epoch 2/100
    251/251 [==============================] - ETA: 0s - loss: 4.1094 - accuracy: 0.0352
    Epoch 00002: val_loss improved from 4.27058 to 3.81736, saving model to model.h5
    251/251 [==============================] - 26s 103ms/step - loss: 4.1094 - accuracy: 0.0352 - val_loss: 3.8174 - val_accuracy: 0.0690
    Epoch 3/100
    251/251 [==============================] - ETA: 0s - loss: 3.7381 - accuracy: 0.0740
    Epoch 00003: val_loss improved from 3.81736 to 3.41435, saving model to model.h5
    251/251 [==============================] - 27s 106ms/step - loss: 3.7381 - accuracy: 0.0740 - val_loss: 3.4143 - val_accuracy: 0.1224
    Epoch 4/100
    250/251 [============================>.] - ETA: 0s - loss: 3.4254 - accuracy: 0.1086
    Epoch 00004: val_loss improved from 3.41435 to 3.13237, saving model to model.h5
    251/251 [==============================] - 22s 89ms/step - loss: 3.4254 - accuracy: 0.1086 - val_loss: 3.1324 - val_accuracy: 0.1759
    Epoch 5/100
    250/251 [============================>.] - ETA: 0s - loss: 3.2252 - accuracy: 0.1305
    Epoch 00005: val_loss improved from 3.13237 to 2.96009, saving model to model.h5
    251/251 [==============================] - 21s 83ms/step - loss: 3.2246 - accuracy: 0.1307 - val_loss: 2.9601 - val_accuracy: 0.1899
    Epoch 6/100
    250/251 [============================>.] - ETA: 0s - loss: 3.0463 - accuracy: 0.1566
    Epoch 00006: val_loss improved from 2.96009 to 2.76175, saving model to model.h5
    251/251 [==============================] - 24s 94ms/step - loss: 3.0459 - accuracy: 0.1567 - val_loss: 2.7618 - val_accuracy: 0.2284
    Epoch 7/100
    250/251 [============================>.] - ETA: 0s - loss: 2.9042 - accuracy: 0.1725
    Epoch 00007: val_loss improved from 2.76175 to 2.65965, saving model to model.h5
    251/251 [==============================] - 21s 83ms/step - loss: 2.9042 - accuracy: 0.1724 - val_loss: 2.6597 - val_accuracy: 0.2479
    Epoch 8/100
    250/251 [============================>.] - ETA: 0s - loss: 2.7758 - accuracy: 0.2026
    Epoch 00008: val_loss improved from 2.65965 to 2.58062, saving model to model.h5
    251/251 [==============================] - 21s 82ms/step - loss: 2.7761 - accuracy: 0.2025 - val_loss: 2.5806 - val_accuracy: 0.2614
    Epoch 9/100
    250/251 [============================>.] - ETA: 0s - loss: 2.6743 - accuracy: 0.2146
    Epoch 00009: val_loss improved from 2.58062 to 2.50623, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 2.6739 - accuracy: 0.2148 - val_loss: 2.5062 - val_accuracy: 0.2794
    Epoch 10/100
    250/251 [============================>.] - ETA: 0s - loss: 2.5910 - accuracy: 0.2379
    Epoch 00010: val_loss improved from 2.50623 to 2.35772, saving model to model.h5
    251/251 [==============================] - 20s 81ms/step - loss: 2.5912 - accuracy: 0.2378 - val_loss: 2.3577 - val_accuracy: 0.3178
    Epoch 11/100
    251/251 [==============================] - ETA: 0s - loss: 2.5195 - accuracy: 0.2524
    Epoch 00011: val_loss improved from 2.35772 to 2.31286, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 2.5195 - accuracy: 0.2524 - val_loss: 2.3129 - val_accuracy: 0.3203
    Epoch 12/100
    250/251 [============================>.] - ETA: 0s - loss: 2.4290 - accuracy: 0.2781
    Epoch 00012: val_loss improved from 2.31286 to 2.23922, saving model to model.h5
    251/251 [==============================] - 23s 90ms/step - loss: 2.4287 - accuracy: 0.2783 - val_loss: 2.2392 - val_accuracy: 0.3393
    Epoch 13/100
    250/251 [============================>.] - ETA: 0s - loss: 2.3697 - accuracy: 0.2887
    Epoch 00013: val_loss improved from 2.23922 to 2.13489, saving model to model.h5
    251/251 [==============================] - 21s 83ms/step - loss: 2.3696 - accuracy: 0.2886 - val_loss: 2.1349 - val_accuracy: 0.3653
    Epoch 14/100
    250/251 [============================>.] - ETA: 0s - loss: 2.2854 - accuracy: 0.3026
    Epoch 00014: val_loss improved from 2.13489 to 2.11181, saving model to model.h5
    251/251 [==============================] - 20s 81ms/step - loss: 2.2850 - accuracy: 0.3026 - val_loss: 2.1118 - val_accuracy: 0.3758
    Epoch 15/100
    250/251 [============================>.] - ETA: 0s - loss: 2.2246 - accuracy: 0.3231
    Epoch 00015: val_loss improved from 2.11181 to 2.08685, saving model to model.h5
    251/251 [==============================] - 21s 83ms/step - loss: 2.2244 - accuracy: 0.3231 - val_loss: 2.0869 - val_accuracy: 0.3958
    Epoch 16/100
    250/251 [============================>.] - ETA: 0s - loss: 2.1744 - accuracy: 0.3330
    Epoch 00016: val_loss improved from 2.08685 to 1.98987, saving model to model.h5
    251/251 [==============================] - 21s 83ms/step - loss: 2.1742 - accuracy: 0.3330 - val_loss: 1.9899 - val_accuracy: 0.4173
    Epoch 17/100
    250/251 [============================>.] - ETA: 0s - loss: 2.1058 - accuracy: 0.3459
    Epoch 00017: val_loss improved from 1.98987 to 1.94376, saving model to model.h5
    251/251 [==============================] - 20s 80ms/step - loss: 2.1057 - accuracy: 0.3459 - val_loss: 1.9438 - val_accuracy: 0.4218
    Epoch 18/100
    250/251 [============================>.] - ETA: 0s - loss: 2.0692 - accuracy: 0.3589
    Epoch 00018: val_loss improved from 1.94376 to 1.88552, saving model to model.h5
    251/251 [==============================] - 20s 79ms/step - loss: 2.0688 - accuracy: 0.3590 - val_loss: 1.8855 - val_accuracy: 0.4628
    Epoch 19/100
    251/251 [==============================] - ETA: 0s - loss: 2.0071 - accuracy: 0.3759
    Epoch 00019: val_loss improved from 1.88552 to 1.85381, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 2.0071 - accuracy: 0.3759 - val_loss: 1.8538 - val_accuracy: 0.4598
    Epoch 20/100
    250/251 [============================>.] - ETA: 0s - loss: 1.9553 - accuracy: 0.3950
    Epoch 00020: val_loss improved from 1.85381 to 1.84792, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 1.9551 - accuracy: 0.3950 - val_loss: 1.8479 - val_accuracy: 0.4693
    Epoch 21/100
    250/251 [============================>.] - ETA: 0s - loss: 1.9132 - accuracy: 0.4038
    Epoch 00021: val_loss improved from 1.84792 to 1.76239, saving model to model.h5
    251/251 [==============================] - 26s 102ms/step - loss: 1.9133 - accuracy: 0.4037 - val_loss: 1.7624 - val_accuracy: 0.4868
    Epoch 22/100
    251/251 [==============================] - ETA: 0s - loss: 1.8772 - accuracy: 0.4188
    Epoch 00022: val_loss improved from 1.76239 to 1.73721, saving model to model.h5
    251/251 [==============================] - 38s 151ms/step - loss: 1.8772 - accuracy: 0.4188 - val_loss: 1.7372 - val_accuracy: 0.4988
    Epoch 23/100
    251/251 [==============================] - ETA: 0s - loss: 1.8216 - accuracy: 0.4311
    Epoch 00023: val_loss did not improve from 1.73721
    251/251 [==============================] - 40s 159ms/step - loss: 1.8216 - accuracy: 0.4311 - val_loss: 1.7574 - val_accuracy: 0.4863
    Epoch 24/100
    251/251 [==============================] - ETA: 0s - loss: 1.7852 - accuracy: 0.4415
    Epoch 00024: val_loss improved from 1.73721 to 1.65693, saving model to model.h5
    251/251 [==============================] - 45s 179ms/step - loss: 1.7852 - accuracy: 0.4415 - val_loss: 1.6569 - val_accuracy: 0.5282
    Epoch 25/100
    251/251 [==============================] - ETA: 0s - loss: 1.7496 - accuracy: 0.4492
    Epoch 00025: val_loss improved from 1.65693 to 1.63070, saving model to model.h5
    251/251 [==============================] - 55s 218ms/step - loss: 1.7496 - accuracy: 0.4492 - val_loss: 1.6307 - val_accuracy: 0.5362
    Epoch 26/100
    251/251 [==============================] - ETA: 0s - loss: 1.7145 - accuracy: 0.4561
    Epoch 00026: val_loss improved from 1.63070 to 1.62319, saving model to model.h5
    251/251 [==============================] - 54s 215ms/step - loss: 1.7145 - accuracy: 0.4561 - val_loss: 1.6232 - val_accuracy: 0.5272
    Epoch 27/100
    251/251 [==============================] - ETA: 0s - loss: 1.6668 - accuracy: 0.4687
    Epoch 00027: val_loss improved from 1.62319 to 1.57346, saving model to model.h5
    251/251 [==============================] - 51s 204ms/step - loss: 1.6668 - accuracy: 0.4687 - val_loss: 1.5735 - val_accuracy: 0.5572
    Epoch 28/100
    251/251 [==============================] - ETA: 0s - loss: 1.6319 - accuracy: 0.4842
    Epoch 00028: val_loss did not improve from 1.57346
    251/251 [==============================] - 44s 174ms/step - loss: 1.6319 - accuracy: 0.4842 - val_loss: 1.5962 - val_accuracy: 0.5632
    Epoch 29/100
    251/251 [==============================] - ETA: 0s - loss: 1.6075 - accuracy: 0.4882
    Epoch 00029: val_loss did not improve from 1.57346
    251/251 [==============================] - 44s 175ms/step - loss: 1.6075 - accuracy: 0.4882 - val_loss: 1.5836 - val_accuracy: 0.5632
    Epoch 30/100
    251/251 [==============================] - ETA: 0s - loss: 1.5591 - accuracy: 0.4993
    Epoch 00030: val_loss improved from 1.57346 to 1.52777, saving model to model.h5
    251/251 [==============================] - 44s 174ms/step - loss: 1.5591 - accuracy: 0.4993 - val_loss: 1.5278 - val_accuracy: 0.5782
    Epoch 31/100
    251/251 [==============================] - ETA: 0s - loss: 1.5530 - accuracy: 0.5071
    Epoch 00031: val_loss improved from 1.52777 to 1.52052, saving model to model.h5
    251/251 [==============================] - 49s 194ms/step - loss: 1.5530 - accuracy: 0.5071 - val_loss: 1.5205 - val_accuracy: 0.5817
    Epoch 32/100
    251/251 [==============================] - ETA: 0s - loss: 1.4849 - accuracy: 0.5273
    Epoch 00032: val_loss did not improve from 1.52052
    251/251 [==============================] - 46s 184ms/step - loss: 1.4849 - accuracy: 0.5273 - val_loss: 1.5332 - val_accuracy: 0.5842
    Epoch 33/100
    251/251 [==============================] - ETA: 0s - loss: 1.4670 - accuracy: 0.5326
    Epoch 00033: val_loss did not improve from 1.52052
    251/251 [==============================] - 54s 217ms/step - loss: 1.4670 - accuracy: 0.5326 - val_loss: 1.5420 - val_accuracy: 0.5962
    Epoch 34/100
    251/251 [==============================] - ETA: 0s - loss: 1.4345 - accuracy: 0.5390
    Epoch 00034: val_loss improved from 1.52052 to 1.48726, saving model to model.h5
    251/251 [==============================] - 50s 199ms/step - loss: 1.4345 - accuracy: 0.5390 - val_loss: 1.4873 - val_accuracy: 0.5867
    Epoch 35/100
    251/251 [==============================] - ETA: 0s - loss: 1.4037 - accuracy: 0.5477
    Epoch 00035: val_loss did not improve from 1.48726
    251/251 [==============================] - 52s 207ms/step - loss: 1.4037 - accuracy: 0.5477 - val_loss: 1.5423 - val_accuracy: 0.5947
    Epoch 36/100
    251/251 [==============================] - ETA: 0s - loss: 1.4160 - accuracy: 0.5444
    Epoch 00036: val_loss did not improve from 1.48726
    251/251 [==============================] - 30s 120ms/step - loss: 1.4160 - accuracy: 0.5444 - val_loss: 1.5370 - val_accuracy: 0.5862
    Epoch 37/100
    251/251 [==============================] - ETA: 0s - loss: 1.3532 - accuracy: 0.5593
    Epoch 00037: val_loss improved from 1.48726 to 1.48493, saving model to model.h5
    251/251 [==============================] - 29s 115ms/step - loss: 1.3532 - accuracy: 0.5593 - val_loss: 1.4849 - val_accuracy: 0.6107
    Epoch 38/100
    251/251 [==============================] - ETA: 0s - loss: 1.3427 - accuracy: 0.5608
    Epoch 00038: val_loss did not improve from 1.48493
    251/251 [==============================] - 27s 109ms/step - loss: 1.3427 - accuracy: 0.5608 - val_loss: 1.4905 - val_accuracy: 0.6227
    Epoch 39/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2975 - accuracy: 0.5857
    Epoch 00039: val_loss improved from 1.48493 to 1.44457, saving model to model.h5
    251/251 [==============================] - 28s 111ms/step - loss: 1.2981 - accuracy: 0.5857 - val_loss: 1.4446 - val_accuracy: 0.6237
    Epoch 40/100
    251/251 [==============================] - ETA: 0s - loss: 1.2937 - accuracy: 0.5773
    Epoch 00040: val_loss did not improve from 1.44457
    251/251 [==============================] - 26s 105ms/step - loss: 1.2937 - accuracy: 0.5773 - val_loss: 1.4843 - val_accuracy: 0.6272
    Epoch 41/100
    251/251 [==============================] - ETA: 0s - loss: 1.2403 - accuracy: 0.5977
    Epoch 00041: val_loss did not improve from 1.44457
    251/251 [==============================] - 25s 100ms/step - loss: 1.2403 - accuracy: 0.5977 - val_loss: 1.5572 - val_accuracy: 0.6232
    Epoch 42/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2261 - accuracy: 0.5960
    Epoch 00042: val_loss did not improve from 1.44457
    251/251 [==============================] - 23s 93ms/step - loss: 1.2261 - accuracy: 0.5960 - val_loss: 1.5840 - val_accuracy: 0.6137
    Epoch 43/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2664 - accuracy: 0.5891
    Epoch 00043: val_loss did not improve from 1.44457
    251/251 [==============================] - 23s 92ms/step - loss: 1.2664 - accuracy: 0.5890 - val_loss: 1.5171 - val_accuracy: 0.6347
    Epoch 44/100
    251/251 [==============================] - ETA: 0s - loss: 1.1855 - accuracy: 0.6094
    Epoch 00044: val_loss did not improve from 1.44457
    251/251 [==============================] - 28s 110ms/step - loss: 1.1855 - accuracy: 0.6094 - val_loss: 1.4855 - val_accuracy: 0.6367
    Epoch 45/100
    251/251 [==============================] - ETA: 0s - loss: 1.1386 - accuracy: 0.6145
    Epoch 00045: val_loss did not improve from 1.44457
    251/251 [==============================] - 27s 109ms/step - loss: 1.1386 - accuracy: 0.6145 - val_loss: 1.5674 - val_accuracy: 0.6102
    Epoch 46/100
    251/251 [==============================] - ETA: 0s - loss: 1.1428 - accuracy: 0.6208
    Epoch 00046: val_loss did not improve from 1.44457
    251/251 [==============================] - 27s 107ms/step - loss: 1.1428 - accuracy: 0.6208 - val_loss: 1.4835 - val_accuracy: 0.6347
    Epoch 47/100
    251/251 [==============================] - ETA: 0s - loss: 1.1170 - accuracy: 0.6324
    Epoch 00047: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 115ms/step - loss: 1.1170 - accuracy: 0.6324 - val_loss: 1.5337 - val_accuracy: 0.6472
    Epoch 48/100
    251/251 [==============================] - ETA: 0s - loss: 1.0891 - accuracy: 0.6371
    Epoch 00048: val_loss did not improve from 1.44457
    251/251 [==============================] - 28s 113ms/step - loss: 1.0891 - accuracy: 0.6371 - val_loss: 1.4967 - val_accuracy: 0.6497
    Epoch 49/100
    251/251 [==============================] - ETA: 0s - loss: 1.0769 - accuracy: 0.6435
    Epoch 00049: val_loss did not improve from 1.44457
    251/251 [==============================] - 27s 109ms/step - loss: 1.0769 - accuracy: 0.6435 - val_loss: 1.5394 - val_accuracy: 0.6402
    Epoch 50/100
    251/251 [==============================] - ETA: 0s - loss: 1.0580 - accuracy: 0.6440
    Epoch 00050: val_loss did not improve from 1.44457
    251/251 [==============================] - 28s 113ms/step - loss: 1.0580 - accuracy: 0.6440 - val_loss: 1.5288 - val_accuracy: 0.6467
    Epoch 51/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0260 - accuracy: 0.6565
    Epoch 00051: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 116ms/step - loss: 1.0257 - accuracy: 0.6566 - val_loss: 1.5142 - val_accuracy: 0.6532
    Epoch 52/100
    251/251 [==============================] - ETA: 0s - loss: 1.0224 - accuracy: 0.6613
    Epoch 00052: val_loss did not improve from 1.44457
    251/251 [==============================] - 38s 150ms/step - loss: 1.0224 - accuracy: 0.6613 - val_loss: 1.4909 - val_accuracy: 0.6592
    Epoch 53/100
    251/251 [==============================] - ETA: 0s - loss: 1.0085 - accuracy: 0.6620
    Epoch 00053: val_loss did not improve from 1.44457
    251/251 [==============================] - 37s 149ms/step - loss: 1.0085 - accuracy: 0.6620 - val_loss: 1.5250 - val_accuracy: 0.6522
    Epoch 54/100
    251/251 [==============================] - ETA: 0s - loss: 0.9991 - accuracy: 0.6664
    Epoch 00054: val_loss did not improve from 1.44457
    251/251 [==============================] - 39s 157ms/step - loss: 0.9991 - accuracy: 0.6664 - val_loss: 1.5270 - val_accuracy: 0.6552
    Epoch 55/100
    251/251 [==============================] - ETA: 0s - loss: 0.9793 - accuracy: 0.6676
    Epoch 00055: val_loss did not improve from 1.44457
    251/251 [==============================] - 35s 138ms/step - loss: 0.9793 - accuracy: 0.6676 - val_loss: 1.5554 - val_accuracy: 0.6707
    Epoch 56/100
    251/251 [==============================] - ETA: 0s - loss: 0.9268 - accuracy: 0.6797
    Epoch 00056: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 114ms/step - loss: 0.9268 - accuracy: 0.6797 - val_loss: 1.5295 - val_accuracy: 0.6642
    Epoch 57/100
    251/251 [==============================] - ETA: 0s - loss: 0.9439 - accuracy: 0.6802
    Epoch 00057: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 116ms/step - loss: 0.9439 - accuracy: 0.6802 - val_loss: 1.5952 - val_accuracy: 0.6642
    Epoch 58/100
    251/251 [==============================] - ETA: 0s - loss: 0.9087 - accuracy: 0.6946
    Epoch 00058: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 117ms/step - loss: 0.9087 - accuracy: 0.6946 - val_loss: 1.5913 - val_accuracy: 0.6622
    Epoch 59/100
    250/251 [============================>.] - ETA: 0s - loss: 0.9004 - accuracy: 0.6955
    Epoch 00059: val_loss did not improve from 1.44457
    251/251 [==============================] - 29s 115ms/step - loss: 0.9004 - accuracy: 0.6955 - val_loss: 1.6798 - val_accuracy: 0.6532
    Epoch 60/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8781 - accuracy: 0.6970
    Epoch 00060: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.8780 - accuracy: 0.6971 - val_loss: 1.6234 - val_accuracy: 0.6727
    Epoch 61/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8709 - accuracy: 0.7005
    Epoch 00061: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.8711 - accuracy: 0.7005 - val_loss: 1.6448 - val_accuracy: 0.6727
    Epoch 62/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8663 - accuracy: 0.7039
    Epoch 00062: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.8664 - accuracy: 0.7039 - val_loss: 1.7071 - val_accuracy: 0.6687
    Epoch 63/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8500 - accuracy: 0.7113
    Epoch 00063: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.8500 - accuracy: 0.7112 - val_loss: 1.6326 - val_accuracy: 0.6732
    Epoch 64/100
    251/251 [==============================] - ETA: 0s - loss: 0.8341 - accuracy: 0.7102
    Epoch 00064: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.8341 - accuracy: 0.7102 - val_loss: 1.6507 - val_accuracy: 0.6782
    Epoch 65/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8156 - accuracy: 0.7172
    Epoch 00065: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.8154 - accuracy: 0.7174 - val_loss: 1.7312 - val_accuracy: 0.6652
    Epoch 66/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8050 - accuracy: 0.7207
    Epoch 00066: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.8047 - accuracy: 0.7209 - val_loss: 1.7547 - val_accuracy: 0.6607
    Epoch 67/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7792 - accuracy: 0.7297
    Epoch 00067: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.7792 - accuracy: 0.7299 - val_loss: 1.7610 - val_accuracy: 0.6757
    Epoch 68/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7789 - accuracy: 0.7333
    Epoch 00068: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.7786 - accuracy: 0.7333 - val_loss: 1.6850 - val_accuracy: 0.6717
    Epoch 69/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7749 - accuracy: 0.7364
    Epoch 00069: val_loss did not improve from 1.44457
    251/251 [==============================] - 38s 151ms/step - loss: 0.7748 - accuracy: 0.7363 - val_loss: 1.7332 - val_accuracy: 0.6737
    Epoch 70/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7432 - accuracy: 0.7372
    Epoch 00070: val_loss did not improve from 1.44457
    251/251 [==============================] - 25s 98ms/step - loss: 0.7429 - accuracy: 0.7373 - val_loss: 1.8610 - val_accuracy: 0.6822
    Epoch 71/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7326 - accuracy: 0.7445
    Epoch 00071: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.7325 - accuracy: 0.7446 - val_loss: 1.7883 - val_accuracy: 0.6872
    Epoch 72/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7453 - accuracy: 0.7404
    Epoch 00072: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.7458 - accuracy: 0.7402 - val_loss: 1.8509 - val_accuracy: 0.6832
    Epoch 73/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7157 - accuracy: 0.7517
    Epoch 00073: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 86ms/step - loss: 0.7163 - accuracy: 0.7516 - val_loss: 1.8285 - val_accuracy: 0.6847
    Epoch 74/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7224 - accuracy: 0.7500
    Epoch 00074: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.7222 - accuracy: 0.7501 - val_loss: 1.9006 - val_accuracy: 0.6782
    Epoch 75/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7251 - accuracy: 0.7450
    Epoch 00075: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.7251 - accuracy: 0.7450 - val_loss: 1.8626 - val_accuracy: 0.6692
    Epoch 76/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6800 - accuracy: 0.7628
    Epoch 00076: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6800 - accuracy: 0.7627 - val_loss: 1.9267 - val_accuracy: 0.6932
    Epoch 77/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6852 - accuracy: 0.7619
    Epoch 00077: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6853 - accuracy: 0.7617 - val_loss: 1.8616 - val_accuracy: 0.6872
    Epoch 78/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6749 - accuracy: 0.7615
    Epoch 00078: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6747 - accuracy: 0.7616 - val_loss: 2.0164 - val_accuracy: 0.6792
    Epoch 79/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6499 - accuracy: 0.7745
    Epoch 00079: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.6500 - accuracy: 0.7745 - val_loss: 1.8782 - val_accuracy: 0.6927
    Epoch 80/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6679 - accuracy: 0.7676
    Epoch 00080: val_loss did not improve from 1.44457
    251/251 [==============================] - 23s 90ms/step - loss: 0.6677 - accuracy: 0.7677 - val_loss: 1.9416 - val_accuracy: 0.6782
    Epoch 81/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6359 - accuracy: 0.7716
    Epoch 00081: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6357 - accuracy: 0.7717 - val_loss: 1.9453 - val_accuracy: 0.6962
    Epoch 82/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6212 - accuracy: 0.7830
    Epoch 00082: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6214 - accuracy: 0.7830 - val_loss: 1.8328 - val_accuracy: 0.6907
    Epoch 83/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6386 - accuracy: 0.7735
    Epoch 00083: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 86ms/step - loss: 0.6386 - accuracy: 0.7735 - val_loss: 2.0188 - val_accuracy: 0.6902
    Epoch 84/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6547 - accuracy: 0.7704
    Epoch 00084: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 86ms/step - loss: 0.6545 - accuracy: 0.7705 - val_loss: 2.0437 - val_accuracy: 0.6917
    Epoch 85/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6098 - accuracy: 0.7878
    Epoch 00085: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.6096 - accuracy: 0.7878 - val_loss: 2.0177 - val_accuracy: 0.6927
    Epoch 86/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6293 - accuracy: 0.7812
    Epoch 00086: val_loss did not improve from 1.44457
    251/251 [==============================] - 23s 90ms/step - loss: 0.6293 - accuracy: 0.7813 - val_loss: 2.0607 - val_accuracy: 0.7016
    Epoch 87/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6056 - accuracy: 0.7820
    Epoch 00087: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6054 - accuracy: 0.7821 - val_loss: 1.9644 - val_accuracy: 0.6902
    Epoch 88/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6456 - accuracy: 0.7799
    Epoch 00088: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6462 - accuracy: 0.7798 - val_loss: 2.0080 - val_accuracy: 0.6852
    Epoch 89/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6215 - accuracy: 0.7843
    Epoch 00089: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6213 - accuracy: 0.7843 - val_loss: 2.0475 - val_accuracy: 0.6947
    Epoch 90/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5761 - accuracy: 0.7954
    Epoch 00090: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 86ms/step - loss: 0.5762 - accuracy: 0.7952 - val_loss: 2.1777 - val_accuracy: 0.6877
    Epoch 91/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5653 - accuracy: 0.7959
    Epoch 00091: val_loss did not improve from 1.44457
    251/251 [==============================] - 27s 107ms/step - loss: 0.5654 - accuracy: 0.7958 - val_loss: 2.1215 - val_accuracy: 0.6937
    Epoch 92/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5704 - accuracy: 0.7987
    Epoch 00092: val_loss did not improve from 1.44457
    251/251 [==============================] - 30s 121ms/step - loss: 0.5705 - accuracy: 0.7987 - val_loss: 2.1993 - val_accuracy: 0.6777
    Epoch 93/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5979 - accuracy: 0.7901
    Epoch 00093: val_loss did not improve from 1.44457
    251/251 [==============================] - 30s 118ms/step - loss: 0.5984 - accuracy: 0.7901 - val_loss: 2.2655 - val_accuracy: 0.6817
    Epoch 94/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5628 - accuracy: 0.7987
    Epoch 00094: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.5627 - accuracy: 0.7988 - val_loss: 2.1356 - val_accuracy: 0.6947
    Epoch 95/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5243 - accuracy: 0.8164
    Epoch 00095: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 88ms/step - loss: 0.5244 - accuracy: 0.8163 - val_loss: 2.1908 - val_accuracy: 0.6962
    Epoch 96/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5378 - accuracy: 0.8055
    Epoch 00096: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.5377 - accuracy: 0.8056 - val_loss: 2.3204 - val_accuracy: 0.6867
    Epoch 97/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5264 - accuracy: 0.8106
    Epoch 00097: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.5262 - accuracy: 0.8107 - val_loss: 2.3225 - val_accuracy: 0.6867
    Epoch 98/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5299 - accuracy: 0.8131
    Epoch 00098: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.5298 - accuracy: 0.8132 - val_loss: 2.3495 - val_accuracy: 0.6972
    Epoch 99/100
    251/251 [==============================] - ETA: 0s - loss: 0.6215 - accuracy: 0.7827
    Epoch 00099: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.6215 - accuracy: 0.7827 - val_loss: 2.2520 - val_accuracy: 0.6887
    Epoch 100/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5402 - accuracy: 0.8124
    Epoch 00100: val_loss did not improve from 1.44457
    251/251 [==============================] - 22s 87ms/step - loss: 0.5402 - accuracy: 0.8123 - val_loss: 2.3806 - val_accuracy: 0.6922


## 5. Bidirectional LSTM 


```python
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(GRU(128)))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(78, activation = "softmax"))
  
  return model

model_lstm = create_model(vocab_size, max_length)

model_lstm.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model_lstm.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 84, 128)           299904    
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 256)               198144    
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                8224      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 78)                2574      
    =================================================================
    Total params: 508,846
    Trainable params: 208,942
    Non-trainable params: 299,904
    _________________________________________________________________



```python
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

hist = model_lstm.fit(train_X, train_Y,
                 epochs = 100,
                 batch_size = 32,
                 validation_data = (val_X, val_Y),
                 callbacks = [checkpoint])
```

    Epoch 1/100
    250/251 [============================>.] - ETA: 0s - loss: 4.3337 - accuracy: 0.0188
    Epoch 00001: val_loss improved from inf to 4.27265, saving model to model.h5
    251/251 [==============================] - 22s 86ms/step - loss: 4.3337 - accuracy: 0.0187 - val_loss: 4.2727 - val_accuracy: 0.0240
    Epoch 2/100
    251/251 [==============================] - ETA: 0s - loss: 4.1330 - accuracy: 0.0307
    Epoch 00002: val_loss improved from 4.27265 to 3.94408, saving model to model.h5
    251/251 [==============================] - 30s 118ms/step - loss: 4.1330 - accuracy: 0.0307 - val_loss: 3.9441 - val_accuracy: 0.0460
    Epoch 3/100
    250/251 [============================>.] - ETA: 0s - loss: 3.8935 - accuracy: 0.0562
    Epoch 00003: val_loss improved from 3.94408 to 3.64672, saving model to model.h5
    251/251 [==============================] - 30s 120ms/step - loss: 3.8934 - accuracy: 0.0564 - val_loss: 3.6467 - val_accuracy: 0.0820
    Epoch 4/100
    250/251 [============================>.] - ETA: 0s - loss: 3.6066 - accuracy: 0.0824
    Epoch 00004: val_loss improved from 3.64672 to 3.23732, saving model to model.h5
    251/251 [==============================] - 24s 97ms/step - loss: 3.6066 - accuracy: 0.0823 - val_loss: 3.2373 - val_accuracy: 0.1374
    Epoch 5/100
    251/251 [==============================] - ETA: 0s - loss: 3.3376 - accuracy: 0.1140
    Epoch 00005: val_loss improved from 3.23732 to 3.05696, saving model to model.h5
    251/251 [==============================] - 24s 95ms/step - loss: 3.3376 - accuracy: 0.1140 - val_loss: 3.0570 - val_accuracy: 0.1744
    Epoch 6/100
    251/251 [==============================] - ETA: 0s - loss: 3.1163 - accuracy: 0.1398
    Epoch 00006: val_loss improved from 3.05696 to 2.79989, saving model to model.h5
    251/251 [==============================] - 28s 112ms/step - loss: 3.1163 - accuracy: 0.1398 - val_loss: 2.7999 - val_accuracy: 0.2254
    Epoch 7/100
    250/251 [============================>.] - ETA: 0s - loss: 2.9559 - accuracy: 0.1663
    Epoch 00007: val_loss improved from 2.79989 to 2.64152, saving model to model.h5
    251/251 [==============================] - 20s 78ms/step - loss: 2.9562 - accuracy: 0.1662 - val_loss: 2.6415 - val_accuracy: 0.2534
    Epoch 8/100
    250/251 [============================>.] - ETA: 0s - loss: 2.7911 - accuracy: 0.2014
    Epoch 00008: val_loss improved from 2.64152 to 2.48571, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 2.7906 - accuracy: 0.2014 - val_loss: 2.4857 - val_accuracy: 0.2719
    Epoch 9/100
    250/251 [============================>.] - ETA: 0s - loss: 2.6792 - accuracy: 0.2166
    Epoch 00009: val_loss improved from 2.48571 to 2.36167, saving model to model.h5
    251/251 [==============================] - 20s 79ms/step - loss: 2.6792 - accuracy: 0.2167 - val_loss: 2.3617 - val_accuracy: 0.3063
    Epoch 10/100
    250/251 [============================>.] - ETA: 0s - loss: 2.5443 - accuracy: 0.2549
    Epoch 00010: val_loss improved from 2.36167 to 2.26401, saving model to model.h5
    251/251 [==============================] - 19s 77ms/step - loss: 2.5442 - accuracy: 0.2548 - val_loss: 2.2640 - val_accuracy: 0.3293
    Epoch 11/100
    250/251 [============================>.] - ETA: 0s - loss: 2.4382 - accuracy: 0.2663
    Epoch 00011: val_loss improved from 2.26401 to 2.18261, saving model to model.h5
    251/251 [==============================] - 19s 77ms/step - loss: 2.4387 - accuracy: 0.2662 - val_loss: 2.1826 - val_accuracy: 0.3633
    Epoch 12/100
    250/251 [============================>.] - ETA: 0s - loss: 2.3773 - accuracy: 0.2824
    Epoch 00012: val_loss improved from 2.18261 to 2.13561, saving model to model.h5
    251/251 [==============================] - 21s 84ms/step - loss: 2.3774 - accuracy: 0.2824 - val_loss: 2.1356 - val_accuracy: 0.3538
    Epoch 13/100
    250/251 [============================>.] - ETA: 0s - loss: 2.2824 - accuracy: 0.3079
    Epoch 00013: val_loss improved from 2.13561 to 2.03981, saving model to model.h5
    251/251 [==============================] - 23s 92ms/step - loss: 2.2826 - accuracy: 0.3079 - val_loss: 2.0398 - val_accuracy: 0.3913
    Epoch 14/100
    250/251 [============================>.] - ETA: 0s - loss: 2.2253 - accuracy: 0.3203
    Epoch 00014: val_loss improved from 2.03981 to 1.96118, saving model to model.h5
    251/251 [==============================] - 19s 75ms/step - loss: 2.2252 - accuracy: 0.3203 - val_loss: 1.9612 - val_accuracy: 0.4208
    Epoch 15/100
    250/251 [============================>.] - ETA: 0s - loss: 2.1420 - accuracy: 0.3351
    Epoch 00015: val_loss improved from 1.96118 to 1.92070, saving model to model.h5
    251/251 [==============================] - 19s 76ms/step - loss: 2.1416 - accuracy: 0.3354 - val_loss: 1.9207 - val_accuracy: 0.4133
    Epoch 16/100
    250/251 [============================>.] - ETA: 0s - loss: 2.0963 - accuracy: 0.3614
    Epoch 00016: val_loss improved from 1.92070 to 1.90697, saving model to model.h5
    251/251 [==============================] - 20s 80ms/step - loss: 2.0961 - accuracy: 0.3614 - val_loss: 1.9070 - val_accuracy: 0.4288
    Epoch 17/100
    250/251 [============================>.] - ETA: 0s - loss: 2.0274 - accuracy: 0.3668
    Epoch 00017: val_loss improved from 1.90697 to 1.83711, saving model to model.h5
    251/251 [==============================] - 21s 85ms/step - loss: 2.0275 - accuracy: 0.3667 - val_loss: 1.8371 - val_accuracy: 0.4608
    Epoch 18/100
    250/251 [============================>.] - ETA: 0s - loss: 1.9945 - accuracy: 0.3812
    Epoch 00018: val_loss improved from 1.83711 to 1.78613, saving model to model.h5
    251/251 [==============================] - 19s 74ms/step - loss: 1.9944 - accuracy: 0.3812 - val_loss: 1.7861 - val_accuracy: 0.4773
    Epoch 19/100
    250/251 [============================>.] - ETA: 0s - loss: 1.9376 - accuracy: 0.3919
    Epoch 00019: val_loss improved from 1.78613 to 1.75560, saving model to model.h5
    251/251 [==============================] - 19s 76ms/step - loss: 1.9374 - accuracy: 0.3920 - val_loss: 1.7556 - val_accuracy: 0.4883
    Epoch 20/100
    250/251 [============================>.] - ETA: 0s - loss: 1.9146 - accuracy: 0.3960
    Epoch 00020: val_loss did not improve from 1.75560
    251/251 [==============================] - 19s 74ms/step - loss: 1.9149 - accuracy: 0.3959 - val_loss: 1.7710 - val_accuracy: 0.4743
    Epoch 21/100
    250/251 [============================>.] - ETA: 0s - loss: 1.8692 - accuracy: 0.4126
    Epoch 00021: val_loss improved from 1.75560 to 1.68554, saving model to model.h5
    251/251 [==============================] - 19s 75ms/step - loss: 1.8692 - accuracy: 0.4125 - val_loss: 1.6855 - val_accuracy: 0.4963
    Epoch 22/100
    250/251 [============================>.] - ETA: 0s - loss: 1.8215 - accuracy: 0.4251
    Epoch 00022: val_loss improved from 1.68554 to 1.67111, saving model to model.h5
    251/251 [==============================] - 21s 85ms/step - loss: 1.8216 - accuracy: 0.4250 - val_loss: 1.6711 - val_accuracy: 0.4828
    Epoch 23/100
    250/251 [============================>.] - ETA: 0s - loss: 1.7742 - accuracy: 0.4336
    Epoch 00023: val_loss improved from 1.67111 to 1.62294, saving model to model.h5
    251/251 [==============================] - 23s 92ms/step - loss: 1.7746 - accuracy: 0.4335 - val_loss: 1.6229 - val_accuracy: 0.5212
    Epoch 24/100
    250/251 [============================>.] - ETA: 0s - loss: 1.7232 - accuracy: 0.4431
    Epoch 00024: val_loss improved from 1.62294 to 1.61406, saving model to model.h5
    251/251 [==============================] - 23s 92ms/step - loss: 1.7232 - accuracy: 0.4432 - val_loss: 1.6141 - val_accuracy: 0.5337
    Epoch 25/100
    250/251 [============================>.] - ETA: 0s - loss: 1.6907 - accuracy: 0.4577
    Epoch 00025: val_loss improved from 1.61406 to 1.56826, saving model to model.h5
    251/251 [==============================] - 27s 106ms/step - loss: 1.6904 - accuracy: 0.4578 - val_loss: 1.5683 - val_accuracy: 0.5397
    Epoch 26/100
    251/251 [==============================] - ETA: 0s - loss: 1.6465 - accuracy: 0.4723
    Epoch 00026: val_loss did not improve from 1.56826
    251/251 [==============================] - 29s 115ms/step - loss: 1.6465 - accuracy: 0.4723 - val_loss: 1.5739 - val_accuracy: 0.5442
    Epoch 27/100
    251/251 [==============================] - ETA: 0s - loss: 1.6174 - accuracy: 0.4819
    Epoch 00027: val_loss improved from 1.56826 to 1.53518, saving model to model.h5
    251/251 [==============================] - 24s 96ms/step - loss: 1.6174 - accuracy: 0.4819 - val_loss: 1.5352 - val_accuracy: 0.5517
    Epoch 28/100
    251/251 [==============================] - ETA: 0s - loss: 1.5961 - accuracy: 0.4863
    Epoch 00028: val_loss improved from 1.53518 to 1.52079, saving model to model.h5
    251/251 [==============================] - 24s 96ms/step - loss: 1.5961 - accuracy: 0.4863 - val_loss: 1.5208 - val_accuracy: 0.5637
    Epoch 29/100
    250/251 [============================>.] - ETA: 0s - loss: 1.5584 - accuracy: 0.4946
    Epoch 00029: val_loss improved from 1.52079 to 1.51989, saving model to model.h5
    251/251 [==============================] - 21s 84ms/step - loss: 1.5580 - accuracy: 0.4947 - val_loss: 1.5199 - val_accuracy: 0.5597
    Epoch 30/100
    250/251 [============================>.] - ETA: 0s - loss: 1.5395 - accuracy: 0.5059
    Epoch 00030: val_loss improved from 1.51989 to 1.47681, saving model to model.h5
    251/251 [==============================] - 22s 88ms/step - loss: 1.5393 - accuracy: 0.5059 - val_loss: 1.4768 - val_accuracy: 0.5772
    Epoch 31/100
    250/251 [============================>.] - ETA: 0s - loss: 1.4978 - accuracy: 0.5113
    Epoch 00031: val_loss did not improve from 1.47681
    251/251 [==============================] - 20s 81ms/step - loss: 1.4978 - accuracy: 0.5112 - val_loss: 1.4939 - val_accuracy: 0.5682
    Epoch 32/100
    250/251 [============================>.] - ETA: 0s - loss: 1.4708 - accuracy: 0.5206
    Epoch 00032: val_loss did not improve from 1.47681
    251/251 [==============================] - 21s 82ms/step - loss: 1.4711 - accuracy: 0.5206 - val_loss: 1.4886 - val_accuracy: 0.5887
    Epoch 33/100
    250/251 [============================>.] - ETA: 0s - loss: 1.4506 - accuracy: 0.5157
    Epoch 00033: val_loss improved from 1.47681 to 1.43358, saving model to model.h5
    251/251 [==============================] - 24s 95ms/step - loss: 1.4506 - accuracy: 0.5157 - val_loss: 1.4336 - val_accuracy: 0.5892
    Epoch 34/100
    250/251 [============================>.] - ETA: 0s - loss: 1.4201 - accuracy: 0.5386
    Epoch 00034: val_loss did not improve from 1.43358
    251/251 [==============================] - 28s 114ms/step - loss: 1.4208 - accuracy: 0.5384 - val_loss: 1.4936 - val_accuracy: 0.5722
    Epoch 35/100
    251/251 [==============================] - ETA: 0s - loss: 1.3793 - accuracy: 0.5415
    Epoch 00035: val_loss improved from 1.43358 to 1.43167, saving model to model.h5
    251/251 [==============================] - 25s 99ms/step - loss: 1.3793 - accuracy: 0.5415 - val_loss: 1.4317 - val_accuracy: 0.6012
    Epoch 36/100
    250/251 [============================>.] - ETA: 0s - loss: 1.3316 - accuracy: 0.5584
    Epoch 00036: val_loss did not improve from 1.43167
    251/251 [==============================] - 22s 86ms/step - loss: 1.3313 - accuracy: 0.5584 - val_loss: 1.4490 - val_accuracy: 0.6097
    Epoch 37/100
    251/251 [==============================] - ETA: 0s - loss: 1.3410 - accuracy: 0.5569
    Epoch 00037: val_loss improved from 1.43167 to 1.40131, saving model to model.h5
    251/251 [==============================] - 35s 139ms/step - loss: 1.3410 - accuracy: 0.5569 - val_loss: 1.4013 - val_accuracy: 0.6092
    Epoch 38/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2971 - accuracy: 0.5719
    Epoch 00038: val_loss did not improve from 1.40131
    251/251 [==============================] - 23s 91ms/step - loss: 1.2971 - accuracy: 0.5718 - val_loss: 1.4842 - val_accuracy: 0.5947
    Epoch 39/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2807 - accuracy: 0.5761
    Epoch 00039: val_loss did not improve from 1.40131
    251/251 [==============================] - 20s 79ms/step - loss: 1.2806 - accuracy: 0.5760 - val_loss: 1.4172 - val_accuracy: 0.6107
    Epoch 40/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2446 - accuracy: 0.5811
    Epoch 00040: val_loss improved from 1.40131 to 1.39536, saving model to model.h5
    251/251 [==============================] - 21s 82ms/step - loss: 1.2442 - accuracy: 0.5813 - val_loss: 1.3954 - val_accuracy: 0.6127
    Epoch 41/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2128 - accuracy: 0.5901
    Epoch 00041: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 77ms/step - loss: 1.2125 - accuracy: 0.5903 - val_loss: 1.4070 - val_accuracy: 0.6242
    Epoch 42/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2083 - accuracy: 0.5921
    Epoch 00042: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 1.2086 - accuracy: 0.5920 - val_loss: 1.3974 - val_accuracy: 0.6302
    Epoch 43/100
    250/251 [============================>.] - ETA: 0s - loss: 1.2033 - accuracy: 0.5974
    Epoch 00043: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 76ms/step - loss: 1.2033 - accuracy: 0.5973 - val_loss: 1.4173 - val_accuracy: 0.6252
    Epoch 44/100
    250/251 [============================>.] - ETA: 0s - loss: 1.1614 - accuracy: 0.6043
    Epoch 00044: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 82ms/step - loss: 1.1611 - accuracy: 0.6044 - val_loss: 1.4085 - val_accuracy: 0.6262
    Epoch 45/100
    250/251 [============================>.] - ETA: 0s - loss: 1.1169 - accuracy: 0.6217
    Epoch 00045: val_loss did not improve from 1.39536
    251/251 [==============================] - 22s 87ms/step - loss: 1.1174 - accuracy: 0.6215 - val_loss: 1.4679 - val_accuracy: 0.6272
    Epoch 46/100
    250/251 [============================>.] - ETA: 0s - loss: 1.1135 - accuracy: 0.6220
    Epoch 00046: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 1.1132 - accuracy: 0.6221 - val_loss: 1.4306 - val_accuracy: 0.6287
    Epoch 47/100
    250/251 [============================>.] - ETA: 0s - loss: 1.1191 - accuracy: 0.6244
    Epoch 00047: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 80ms/step - loss: 1.1188 - accuracy: 0.6245 - val_loss: 1.4338 - val_accuracy: 0.6342
    Epoch 48/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0953 - accuracy: 0.6346
    Epoch 00048: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 81ms/step - loss: 1.0954 - accuracy: 0.6345 - val_loss: 1.4591 - val_accuracy: 0.6267
    Epoch 49/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0603 - accuracy: 0.6415
    Epoch 00049: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 76ms/step - loss: 1.0600 - accuracy: 0.6415 - val_loss: 1.4817 - val_accuracy: 0.6407
    Epoch 50/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0684 - accuracy: 0.6388
    Epoch 00050: val_loss did not improve from 1.39536
    251/251 [==============================] - 22s 89ms/step - loss: 1.0688 - accuracy: 0.6385 - val_loss: 1.4735 - val_accuracy: 0.6302
    Epoch 51/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0324 - accuracy: 0.6457
    Epoch 00051: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 81ms/step - loss: 1.0323 - accuracy: 0.6459 - val_loss: 1.5178 - val_accuracy: 0.6417
    Epoch 52/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0012 - accuracy: 0.6544
    Epoch 00052: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 77ms/step - loss: 1.0014 - accuracy: 0.6543 - val_loss: 1.5634 - val_accuracy: 0.6247
    Epoch 53/100
    250/251 [============================>.] - ETA: 0s - loss: 1.0026 - accuracy: 0.6589
    Epoch 00053: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 1.0027 - accuracy: 0.6589 - val_loss: 1.4810 - val_accuracy: 0.6427
    Epoch 54/100
    250/251 [============================>.] - ETA: 0s - loss: 0.9872 - accuracy: 0.6624
    Epoch 00054: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 0.9870 - accuracy: 0.6625 - val_loss: 1.4839 - val_accuracy: 0.6542
    Epoch 55/100
    251/251 [==============================] - ETA: 0s - loss: 0.9591 - accuracy: 0.6700
    Epoch 00055: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 82ms/step - loss: 0.9591 - accuracy: 0.6700 - val_loss: 1.5425 - val_accuracy: 0.6452
    Epoch 56/100
    250/251 [============================>.] - ETA: 0s - loss: 0.9419 - accuracy: 0.6755
    Epoch 00056: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 85ms/step - loss: 0.9421 - accuracy: 0.6755 - val_loss: 1.5368 - val_accuracy: 0.6372
    Epoch 57/100
    250/251 [============================>.] - ETA: 0s - loss: 0.9355 - accuracy: 0.6697
    Epoch 00057: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 82ms/step - loss: 0.9354 - accuracy: 0.6697 - val_loss: 1.5792 - val_accuracy: 0.6407
    Epoch 58/100
    251/251 [==============================] - ETA: 0s - loss: 0.9021 - accuracy: 0.6909
    Epoch 00058: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 97ms/step - loss: 0.9021 - accuracy: 0.6909 - val_loss: 1.5153 - val_accuracy: 0.6442
    Epoch 59/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8918 - accuracy: 0.6971
    Epoch 00059: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 0.8916 - accuracy: 0.6971 - val_loss: 1.5472 - val_accuracy: 0.6462
    Epoch 60/100
    251/251 [==============================] - ETA: 0s - loss: 0.9094 - accuracy: 0.6851
    Epoch 00060: val_loss did not improve from 1.39536
    251/251 [==============================] - 23s 92ms/step - loss: 0.9094 - accuracy: 0.6851 - val_loss: 1.5257 - val_accuracy: 0.6492
    Epoch 61/100
    251/251 [==============================] - ETA: 0s - loss: 0.8970 - accuracy: 0.6880
    Epoch 00061: val_loss did not improve from 1.39536
    251/251 [==============================] - 22s 88ms/step - loss: 0.8970 - accuracy: 0.6880 - val_loss: 1.5352 - val_accuracy: 0.6562
    Epoch 62/100
    251/251 [==============================] - ETA: 0s - loss: 0.8353 - accuracy: 0.7114
    Epoch 00062: val_loss did not improve from 1.39536
    251/251 [==============================] - 28s 110ms/step - loss: 0.8353 - accuracy: 0.7114 - val_loss: 1.5865 - val_accuracy: 0.6552
    Epoch 63/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8335 - accuracy: 0.7088
    Epoch 00063: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 94ms/step - loss: 0.8335 - accuracy: 0.7086 - val_loss: 1.5941 - val_accuracy: 0.6412
    Epoch 64/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8582 - accuracy: 0.7015
    Epoch 00064: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 85ms/step - loss: 0.8580 - accuracy: 0.7016 - val_loss: 1.5954 - val_accuracy: 0.6552
    Epoch 65/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7998 - accuracy: 0.7180
    Epoch 00065: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 76ms/step - loss: 0.7995 - accuracy: 0.7181 - val_loss: 1.5996 - val_accuracy: 0.6577
    Epoch 66/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7935 - accuracy: 0.7219
    Epoch 00066: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 75ms/step - loss: 0.7934 - accuracy: 0.7219 - val_loss: 1.7087 - val_accuracy: 0.6577
    Epoch 67/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7785 - accuracy: 0.7297
    Epoch 00067: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 75ms/step - loss: 0.7784 - accuracy: 0.7299 - val_loss: 1.7313 - val_accuracy: 0.6537
    Epoch 68/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7697 - accuracy: 0.7333
    Epoch 00068: val_loss did not improve from 1.39536
    251/251 [==============================] - 18s 73ms/step - loss: 0.7700 - accuracy: 0.7332 - val_loss: 1.8027 - val_accuracy: 0.6482
    Epoch 69/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7926 - accuracy: 0.7265
    Epoch 00069: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 74ms/step - loss: 0.7925 - accuracy: 0.7266 - val_loss: 1.6838 - val_accuracy: 0.6637
    Epoch 70/100
    250/251 [============================>.] - ETA: 0s - loss: 0.8141 - accuracy: 0.7212
    Epoch 00070: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 77ms/step - loss: 0.8142 - accuracy: 0.7212 - val_loss: 1.6375 - val_accuracy: 0.6767
    Epoch 71/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7679 - accuracy: 0.7275
    Epoch 00071: val_loss did not improve from 1.39536
    251/251 [==============================] - 32s 128ms/step - loss: 0.7677 - accuracy: 0.7276 - val_loss: 1.6890 - val_accuracy: 0.6627
    Epoch 72/100
    251/251 [==============================] - ETA: 0s - loss: 0.7083 - accuracy: 0.7500
    Epoch 00072: val_loss did not improve from 1.39536
    251/251 [==============================] - 28s 111ms/step - loss: 0.7083 - accuracy: 0.7500 - val_loss: 1.7264 - val_accuracy: 0.6687
    Epoch 73/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7220 - accuracy: 0.7395
    Epoch 00073: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 95ms/step - loss: 0.7222 - accuracy: 0.7393 - val_loss: 1.6969 - val_accuracy: 0.6757
    Epoch 74/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6987 - accuracy: 0.7492
    Epoch 00074: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 81ms/step - loss: 0.6987 - accuracy: 0.7492 - val_loss: 1.6964 - val_accuracy: 0.6647
    Epoch 75/100
    251/251 [==============================] - ETA: 0s - loss: 0.7563 - accuracy: 0.7337
    Epoch 00075: val_loss did not improve from 1.39536
    251/251 [==============================] - 23s 94ms/step - loss: 0.7563 - accuracy: 0.7337 - val_loss: 1.7757 - val_accuracy: 0.6692
    Epoch 76/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6768 - accuracy: 0.7559
    Epoch 00076: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 96ms/step - loss: 0.6766 - accuracy: 0.7558 - val_loss: 1.7114 - val_accuracy: 0.6737
    Epoch 77/100
    250/251 [============================>.] - ETA: 0s - loss: 0.7036 - accuracy: 0.7529
    Epoch 00077: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 95ms/step - loss: 0.7035 - accuracy: 0.7530 - val_loss: 1.8412 - val_accuracy: 0.6662
    Epoch 78/100
    251/251 [==============================] - ETA: 0s - loss: 0.6714 - accuracy: 0.7633
    Epoch 00078: val_loss did not improve from 1.39536
    251/251 [==============================] - 28s 110ms/step - loss: 0.6714 - accuracy: 0.7633 - val_loss: 1.9566 - val_accuracy: 0.6612
    Epoch 79/100
    251/251 [==============================] - ETA: 0s - loss: 0.6922 - accuracy: 0.7523
    Epoch 00079: val_loss did not improve from 1.39536
    251/251 [==============================] - 30s 121ms/step - loss: 0.6922 - accuracy: 0.7523 - val_loss: 1.7092 - val_accuracy: 0.6752
    Epoch 80/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6590 - accuracy: 0.7690
    Epoch 00080: val_loss did not improve from 1.39536
    251/251 [==============================] - 28s 112ms/step - loss: 0.6589 - accuracy: 0.7690 - val_loss: 1.8503 - val_accuracy: 0.6887
    Epoch 81/100
    251/251 [==============================] - ETA: 0s - loss: 0.6538 - accuracy: 0.7717
    Epoch 00081: val_loss did not improve from 1.39536
    251/251 [==============================] - 26s 105ms/step - loss: 0.6538 - accuracy: 0.7717 - val_loss: 2.0097 - val_accuracy: 0.6662
    Epoch 82/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6508 - accuracy: 0.7670
    Epoch 00082: val_loss did not improve from 1.39536
    251/251 [==============================] - 25s 101ms/step - loss: 0.6506 - accuracy: 0.7671 - val_loss: 1.9829 - val_accuracy: 0.6762
    Epoch 83/100
    251/251 [==============================] - ETA: 0s - loss: 0.6350 - accuracy: 0.7710
    Epoch 00083: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 97ms/step - loss: 0.6350 - accuracy: 0.7710 - val_loss: 1.9669 - val_accuracy: 0.6752
    Epoch 84/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6164 - accuracy: 0.7816
    Epoch 00084: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 95ms/step - loss: 0.6164 - accuracy: 0.7816 - val_loss: 2.0317 - val_accuracy: 0.6737
    Epoch 85/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6162 - accuracy: 0.7835
    Epoch 00085: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 94ms/step - loss: 0.6160 - accuracy: 0.7836 - val_loss: 1.9181 - val_accuracy: 0.6812
    Epoch 86/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6228 - accuracy: 0.7818
    Epoch 00086: val_loss did not improve from 1.39536
    251/251 [==============================] - 23s 92ms/step - loss: 0.6228 - accuracy: 0.7817 - val_loss: 1.9226 - val_accuracy: 0.6847
    Epoch 87/100
    251/251 [==============================] - ETA: 0s - loss: 0.6602 - accuracy: 0.7672
    Epoch 00087: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 82ms/step - loss: 0.6602 - accuracy: 0.7672 - val_loss: 1.9660 - val_accuracy: 0.6807
    Epoch 88/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6757 - accuracy: 0.7626
    Epoch 00088: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 83ms/step - loss: 0.6760 - accuracy: 0.7626 - val_loss: 1.9864 - val_accuracy: 0.6827
    Epoch 89/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5945 - accuracy: 0.7884
    Epoch 00089: val_loss did not improve from 1.39536
    251/251 [==============================] - 27s 107ms/step - loss: 0.5950 - accuracy: 0.7883 - val_loss: 1.8583 - val_accuracy: 0.6892
    Epoch 90/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5702 - accuracy: 0.7943
    Epoch 00090: val_loss did not improve from 1.39536
    251/251 [==============================] - 25s 101ms/step - loss: 0.5702 - accuracy: 0.7943 - val_loss: 2.1014 - val_accuracy: 0.6702
    Epoch 91/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5755 - accuracy: 0.7925
    Epoch 00091: val_loss did not improve from 1.39536
    251/251 [==============================] - 21s 82ms/step - loss: 0.5754 - accuracy: 0.7926 - val_loss: 1.9565 - val_accuracy: 0.6892
    Epoch 92/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5894 - accuracy: 0.7887
    Epoch 00092: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 79ms/step - loss: 0.5898 - accuracy: 0.7886 - val_loss: 1.9644 - val_accuracy: 0.6797
    Epoch 93/100
    250/251 [============================>.] - ETA: 0s - loss: 0.6069 - accuracy: 0.7861
    Epoch 00093: val_loss did not improve from 1.39536
    251/251 [==============================] - 24s 97ms/step - loss: 0.6070 - accuracy: 0.7861 - val_loss: 2.0686 - val_accuracy: 0.6767
    Epoch 94/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5742 - accuracy: 0.7959
    Epoch 00094: val_loss did not improve from 1.39536
    251/251 [==============================] - 25s 101ms/step - loss: 0.5741 - accuracy: 0.7958 - val_loss: 2.0083 - val_accuracy: 0.6842
    Epoch 95/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5382 - accuracy: 0.8127
    Epoch 00095: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 77ms/step - loss: 0.5383 - accuracy: 0.8127 - val_loss: 2.0642 - val_accuracy: 0.6872
    Epoch 96/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5419 - accuracy: 0.8071
    Epoch 00096: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 74ms/step - loss: 0.5418 - accuracy: 0.8072 - val_loss: 2.2727 - val_accuracy: 0.6767
    Epoch 97/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5199 - accuracy: 0.8156
    Epoch 00097: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 76ms/step - loss: 0.5198 - accuracy: 0.8157 - val_loss: 2.0867 - val_accuracy: 0.6832
    Epoch 98/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5532 - accuracy: 0.8104
    Epoch 00098: val_loss did not improve from 1.39536
    251/251 [==============================] - 20s 78ms/step - loss: 0.5536 - accuracy: 0.8102 - val_loss: 2.0325 - val_accuracy: 0.6792
    Epoch 99/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5622 - accuracy: 0.8019
    Epoch 00099: val_loss did not improve from 1.39536
    251/251 [==============================] - 19s 75ms/step - loss: 0.5621 - accuracy: 0.8019 - val_loss: 2.2159 - val_accuracy: 0.6847
    Epoch 100/100
    250/251 [============================>.] - ETA: 0s - loss: 0.5539 - accuracy: 0.7997
    Epoch 00100: val_loss did not improve from 1.39536
    251/251 [==============================] - 18s 74ms/step - loss: 0.5538 - accuracy: 0.7998 - val_loss: 2.2374 - val_accuracy: 0.6802



```python
 model_lstm = load_model("model.h5")
```


```python
def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print(test_word)

  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  x = padding_doc(test_ls, max_length)

  pred = model_lstm.predict(x)
  
  return pred
```


```python
def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))
  
  return classes[0]
```


```python
text = "I am still waiting on my card?"
pred = predictions(text)
result = get_final_output(pred, unique_intent)
print('\nans: {}\n'.format(result))
```

    ['i', 'am', 'still', 'waiting', 'on', 'my', 'card']
    reverted_card_payment? has confidence = 0.18450873
    card_arrival has confidence = 0.15817639
    card_linking has confidence = 0.12097399
    card_delivery_estimate has confidence = 0.1064741
    compromised_card has confidence = 0.10200037
    declined_card_payment has confidence = 0.05154416
    lost_or_stolen_card has confidence = 0.049066834
    transaction_charged_twice has confidence = 0.047501627
    card_payment_not_recognised has confidence = 0.036972415
    cash_withdrawal_not_recognised has confidence = 0.031303987
    request_refund has confidence = 0.031057216
    card_not_working has confidence = 0.021079693
    pending_card_payment has confidence = 0.017942613
    contactless_not_working has confidence = 0.008497491
    pending_top_up has confidence = 0.008054751
    Refund_not_showing_up has confidence = 0.004341421
    balance_not_updated_after_bank_transfer has confidence = 0.004288543
    transfer_not_received_by_recipient has confidence = 0.0033927432
    balance_not_updated_after_cheque_or_cash_deposit has confidence = 0.0032268025
    top_up_failed has confidence = 0.0023433683
    direct_debit_payment_not_recognised has confidence = 0.0016651306
    cancel_transfer has confidence = 0.0014712141
    activate_my_card has confidence = 0.0010279266
    topping_up_by_card has confidence = 0.0007325939
    top_up_reverted has confidence = 0.00067024364
    card_about_to_expire has confidence = 0.00033075683
    extra_charge_on_statement has confidence = 0.00031525167
    pending_transfer has confidence = 0.00025376843
    card_payment_fee_charged has confidence = 0.0001694441
    pending_cash_withdrawal has confidence = 0.00014266488
    lost_or_stolen_phone has confidence = 7.8762685e-05
    verify_top_up has confidence = 7.581609e-05
    passcode_forgotten has confidence = 6.5786066e-05
    declined_cash_withdrawal has confidence = 4.742262e-05
    wrong_amount_of_cash_received has confidence = 3.7077963e-05
    transfer_fee_charged has confidence = 3.382713e-05
    declined_transfer has confidence = 3.2562737e-05
    pin_blocked has confidence = 2.2621058e-05
    visa_or_mastercard has confidence = 2.0943935e-05
    card_payment_wrong_exchange_rate has confidence = 1.3264364e-05
    top_up_limits has confidence = 9.551276e-06
    supported_cards_and_currencies has confidence = 7.5661947e-06
    card_swallowed has confidence = 7.277046e-06
    failed_transfer has confidence = 5.235689e-06
    card_acceptance has confidence = 2.6937528e-06
    edit_personal_details has confidence = 2.064899e-06
    transfer_timing has confidence = 1.7648853e-06
    unable_to_verify_identity has confidence = 1.7274397e-06
    beneficiary_not_allowed has confidence = 1.691029e-06
    apple_pay_or_google_pay has confidence = 1.5878446e-06
    terminate_account has confidence = 1.5112169e-06
    top_up_by_cash_or_cheque has confidence = 1.1815072e-06
    wrong_exchange_rate_for_cash_withdrawal has confidence = 5.2487746e-07
    top_up_by_card_charge has confidence = 4.5812382e-07
    country_support has confidence = 1.3935404e-07
    age_limit has confidence = 1.14556514e-07
    cash_withdrawal_charge has confidence = 1.0017262e-07
    get_physical_card has confidence = 9.34654e-08
    virtual_card_not_working has confidence = 6.933581e-08
    getting_spare_card has confidence = 5.927304e-08
    order_physical_card has confidence = 5.1681667e-08
    why_verify_identity has confidence = 5.0734858e-08
    receiving_money has confidence = 2.6826314e-08
    verify_source_of_funds has confidence = 2.2387393e-08
    transfer_into_account has confidence = 4.5445026e-09
    atm_support has confidence = 3.5053285e-09
    automatic_top_up has confidence = 3.34655e-09
    change_pin has confidence = 9.565742e-10
    top_up_by_bank_transfer_charge has confidence = 5.0569987e-10
    category has confidence = 2.7366087e-10
    getting_virtual_card has confidence = 1.4772895e-10
    exchange_charge has confidence = 1.4481581e-10
    disposable_card_limits has confidence = 3.252561e-11
    exchange_rate has confidence = 1.6944534e-11
    get_disposable_virtual_card has confidence = 4.0523717e-12
    verify_my_identity has confidence = 2.6752528e-12
    fiat_currency_support has confidence = 5.7656343e-13
    exchange_via_app has confidence = 3.2046e-13
    
    ans: reverted_card_payment?
    



```python
text = "What are you exchange rates?"
pred = predictions(text)
result = get_final_output(pred, unique_intent)
print('\nans: {}\n'.format(result))
```

    ['what', 'are', 'you', 'exchange', 'rates']
    exchange_rate has confidence = 0.9521722
    fiat_currency_support has confidence = 0.018024959
    exchange_via_app has confidence = 0.017376026
    exchange_charge has confidence = 0.008700637
    supported_cards_and_currencies has confidence = 0.0014381895
    card_payment_wrong_exchange_rate has confidence = 0.001223994
    wrong_exchange_rate_for_cash_withdrawal has confidence = 0.0008252348
    apple_pay_or_google_pay has confidence = 0.00023869329
    receiving_money has confidence = 2.8229067e-08
    top_up_by_card_charge has confidence = 9.1513563e-10
    direct_debit_payment_not_recognised has confidence = 5.412927e-10
    card_acceptance has confidence = 3.9122233e-10
    top_up_by_bank_transfer_charge has confidence = 2.2138892e-10
    transfer_fee_charged has confidence = 1.8366568e-10
    beneficiary_not_allowed has confidence = 1.04932375e-10
    declined_card_payment has confidence = 4.518701e-11
    reverted_card_payment? has confidence = 1.4193713e-11
    card_payment_not_recognised has confidence = 7.695896e-12
    atm_support has confidence = 4.113498e-12
    card_payment_fee_charged has confidence = 1.9096842e-12
    cash_withdrawal_charge has confidence = 1.277137e-12
    country_support has confidence = 1.1317267e-12
    top_up_limits has confidence = 7.407987e-13
    automatic_top_up has confidence = 2.682995e-13
    topping_up_by_card has confidence = 1.1231161e-13
    card_swallowed has confidence = 9.672674e-14
    compromised_card has confidence = 6.466508e-15
    cash_withdrawal_not_recognised has confidence = 4.2964145e-15
    card_about_to_expire has confidence = 4.2348903e-15
    top_up_by_cash_or_cheque has confidence = 1.2877509e-15
    transfer_into_account has confidence = 1.0762436e-15
    top_up_failed has confidence = 6.723896e-16
    declined_cash_withdrawal has confidence = 3.7613324e-16
    top_up_reverted has confidence = 1.9517779e-16
    card_not_working has confidence = 1.5230702e-16
    age_limit has confidence = 7.8826474e-17
    transaction_charged_twice has confidence = 6.1763366e-17
    contactless_not_working has confidence = 5.338938e-17
    wrong_amount_of_cash_received has confidence = 1.5566206e-17
    pending_card_payment has confidence = 4.016779e-18
    failed_transfer has confidence = 2.9752628e-18
    getting_spare_card has confidence = 8.919671e-19
    card_delivery_estimate has confidence = 4.7047846e-19
    verify_source_of_funds has confidence = 7.792048e-20
    extra_charge_on_statement has confidence = 4.580394e-20
    terminate_account has confidence = 1.9782868e-20
    request_refund has confidence = 1.3820522e-20
    lost_or_stolen_card has confidence = 2.338311e-21
    verify_top_up has confidence = 1.9709038e-21
    pending_top_up has confidence = 1.843937e-21
    order_physical_card has confidence = 5.3454255e-22
    card_arrival has confidence = 2.0773493e-22
    pending_cash_withdrawal has confidence = 1.3181637e-22
    cancel_transfer has confidence = 2.3067847e-24
    transfer_not_received_by_recipient has confidence = 1.2134126e-24
    activate_my_card has confidence = 9.3400373e-26
    declined_transfer has confidence = 5.8877675e-26
    card_linking has confidence = 1.73983e-26
    pin_blocked has confidence = 1.5728249e-26
    unable_to_verify_identity has confidence = 2.7691884e-27
    why_verify_identity has confidence = 1.5621092e-27
    pending_transfer has confidence = 1.0675882e-28
    visa_or_mastercard has confidence = 2.0461423e-29
    category has confidence = 2.4084525e-30
    balance_not_updated_after_bank_transfer has confidence = 3.7176817e-31
    transfer_timing has confidence = 5.131509e-32
    disposable_card_limits has confidence = 4.3428116e-32
    verify_my_identity has confidence = 2.568251e-35
    get_disposable_virtual_card has confidence = 2.266958e-35
    lost_or_stolen_phone has confidence = 1.287394e-35
    virtual_card_not_working has confidence = 9.0411614e-38
    get_physical_card has confidence = 0.0
    getting_virtual_card has confidence = 0.0
    edit_personal_details has confidence = 0.0
    Refund_not_showing_up has confidence = 0.0
    passcode_forgotten has confidence = 0.0
    balance_not_updated_after_cheque_or_cash_deposit has confidence = 0.0
    change_pin has confidence = 0.0
    
    ans: exchange_rate
    



```python
text = "Which countries are represented?"
pred = predictions(text)
result = get_final_output(pred, unique_intent)
print('\nans: {}\n'.format(result))
```

    ['which', 'countries', 'are', 'represented']
    country_support has confidence = 0.40900862
    fiat_currency_support has confidence = 0.21668743
    card_acceptance has confidence = 0.083111525
    supported_cards_and_currencies has confidence = 0.06816458
    getting_spare_card has confidence = 0.053985287
    atm_support has confidence = 0.04560661
    order_physical_card has confidence = 0.04163523
    exchange_via_app has confidence = 0.019778987
    card_about_to_expire has confidence = 0.01835203
    age_limit has confidence = 0.009834507
    receiving_money has confidence = 0.008849052
    card_delivery_estimate has confidence = 0.004631613
    top_up_by_card_charge has confidence = 0.0040301057
    compromised_card has confidence = 0.0023763773
    card_not_working has confidence = 0.001980662
    card_payment_fee_charged has confidence = 0.0015887762
    activate_my_card has confidence = 0.0014576323
    disposable_card_limits has confidence = 0.001454433
    exchange_rate has confidence = 0.0011822341
    lost_or_stolen_card has confidence = 0.0010140196
    visa_or_mastercard has confidence = 0.0008381222
    card_arrival has confidence = 0.0006780747
    terminate_account has confidence = 0.0006775246
    get_disposable_virtual_card has confidence = 0.000465579
    card_linking has confidence = 0.00038589464
    verify_source_of_funds has confidence = 0.00036793854
    card_swallowed has confidence = 0.00030168338
    topping_up_by_card has confidence = 0.00027173458
    pin_blocked has confidence = 0.00018961306
    exchange_charge has confidence = 0.00017695097
    top_up_by_cash_or_cheque has confidence = 0.00017318357
    transfer_into_account has confidence = 0.00013858912
    top_up_limits has confidence = 0.0001343612
    top_up_by_bank_transfer_charge has confidence = 0.00010686608
    verify_top_up has confidence = 6.671558e-05
    contactless_not_working has confidence = 6.445362e-05
    automatic_top_up has confidence = 6.400799e-05
    apple_pay_or_google_pay has confidence = 3.932909e-05
    virtual_card_not_working has confidence = 3.697887e-05
    getting_virtual_card has confidence = 3.365783e-05
    declined_card_payment has confidence = 2.7065107e-05
    reverted_card_payment? has confidence = 1.7504552e-05
    declined_cash_withdrawal has confidence = 7.463368e-06
    why_verify_identity has confidence = 2.4997935e-06
    transaction_charged_twice has confidence = 1.3058165e-06
    transfer_fee_charged has confidence = 9.0900323e-07
    request_refund has confidence = 8.7911747e-07
    get_physical_card has confidence = 5.894303e-07
    verify_my_identity has confidence = 2.3125166e-07
    top_up_failed has confidence = 1.2750947e-07
    transfer_timing has confidence = 8.6914184e-08
    card_payment_not_recognised has confidence = 8.17943e-08
    category has confidence = 7.283099e-08
    direct_debit_payment_not_recognised has confidence = 4.954824e-08
    edit_personal_details has confidence = 4.720177e-08
    cash_withdrawal_not_recognised has confidence = 1.8368498e-08
    lost_or_stolen_phone has confidence = 1.6912239e-08
    beneficiary_not_allowed has confidence = 1.5800603e-08
    change_pin has confidence = 1.3319637e-08
    unable_to_verify_identity has confidence = 1.0165063e-08
    wrong_exchange_rate_for_cash_withdrawal has confidence = 5.0885043e-09
    pending_top_up has confidence = 2.368248e-09
    wrong_amount_of_cash_received has confidence = 9.399134e-10
    balance_not_updated_after_bank_transfer has confidence = 4.2335166e-10
    top_up_reverted has confidence = 2.592017e-10
    cash_withdrawal_charge has confidence = 1.6687729e-10
    passcode_forgotten has confidence = 1.4617142e-10
    failed_transfer has confidence = 1.4010458e-10
    declined_transfer has confidence = 3.5461915e-11
    cancel_transfer has confidence = 3.0194052e-11
    card_payment_wrong_exchange_rate has confidence = 2.0527226e-11
    extra_charge_on_statement has confidence = 4.192666e-12
    pending_card_payment has confidence = 1.1055137e-12
    transfer_not_received_by_recipient has confidence = 1.2168023e-13
    pending_transfer has confidence = 1.1544019e-15
    balance_not_updated_after_cheque_or_cash_deposit has confidence = 8.8591806e-16
    Refund_not_showing_up has confidence = 2.1942902e-16
    pending_cash_withdrawal has confidence = 1.652574e-18
    
    ans: country_support
    

