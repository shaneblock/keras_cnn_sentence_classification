import csv,gensim
import numpy as np
from keras.layers import Conv1D, Embedding
from keras.models import Sequential
from keras.layers import concatenate
from keras import layers
import keras
import matplotlib.pyplot as plt
from sklearn import model_selection

f = open('sentence.csv')
reader = csv.reader(f)

new_model = gensim.models.Word2Vec.load('word2vec/smp.w2v.300d')

#初始化区域
vocabs = []
datas = []
labels = []

def getid(word):
    return vocabs.index(word)

for row in reader:
    label = row[0]
    sentence = row[1:]
    if '，' in sentence:
        sentence.remove('，')
    if '。' in sentence:
        sentence.remove('。')
    for word in sentence:
        if word not in vocabs:
            vocabs.append(word)

    data = [getid(word) for word in sentence]

    labels.append(label)
    datas.append(data)

# datas = np.array(datas)
datas = keras.preprocessing.sequence.pad_sequences(datas,maxlen=15)

labels = np.array(labels)

weights=[np.random.uniform(low=-10,high=10,size=(300,))
                 if word not in new_model.wv
                 else new_model[word] for word in vocabs]
weights = np.array(weights)

word_input = keras.Input(shape=(15,))
embedding_layer = Embedding(input_dim= weights.shape[0],output_dim= weights.shape[1], weights=[weights])(word_input)

kernel_sizes=[2,3]
conv_xs_list=[]
for kernel_size in kernel_sizes:
    tmp_conv_xs=layers.Convolution1D(256,kernel_size,activation='relu')(embedding_layer)
    tmp_conv_xs=layers.Dropout(0.3)(tmp_conv_xs)
    tmp_conv_xs=layers.GlobalMaxPool1D()(tmp_conv_xs)
    conv_xs_list.append(tmp_conv_xs)

xs=concatenate(conv_xs_list)

xs=layers.Dropout(0.5)(xs)
xs=layers.Dense(100,activation='tanh')(xs)
xs = layers.Dropout(0.5)(xs)
output=layers.Dense(1,activation='sigmoid')(xs)

model = keras.Model(word_input,output)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

十折交叉区
lol = model_selection.KFold(n_splits=10,shuffle=True)

score = []
j = 1
for train,test in lol.split(datas):
    print("第"+str(j)+"次试验")
    j += 1
    # print('train_index:%s , test_index: %s ' %(train,test))
    train_data = [datas[i] for i in train]
    train_data = np.array(train_data)
    train_label = [labels[i] for i in train]
    train_label = np.array(train_label)
    test_data = [datas[i] for i in test]
    test_data = np.array(test_data)
    test_label = [labels[i] for i in test]
    test_label = np.array(test_label)

    word_input = keras.Input(shape=(15,))
    embedding_layer = Embedding(input_dim= weights.shape[0],output_dim= weights.shape[1], weights=[weights])(word_input)

    kernel_sizes=[2,3]
    conv_xs_list=[]
    for kernel_size in kernel_sizes:
        tmp_conv_xs=layers.Convolution1D(256,kernel_size,activation='relu')(embedding_layer)
        tmp_conv_xs=layers.Dropout(0.3)(tmp_conv_xs)
        tmp_conv_xs=layers.GlobalMaxPool1D()(tmp_conv_xs)
        conv_xs_list.append(tmp_conv_xs)

    xs=concatenate(conv_xs_list)

    xs=layers.Dropout(0.5)(xs)
    xs=layers.Dense(100,activation='tanh')(xs)
    xs = layers.Dropout(0.5)(xs)
    output=layers.Dense(1,activation='sigmoid')(xs)
    model = keras.Model(word_input,output)
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(train_data,train_label,nb_epoch=20)
    temp = model.evaluate(test_data,test_label,verbose=0)
    score.append(temp[1])

print(score)
acc = 0
for i in score: acc += i
print(acc/10)



train_data = datas[:9*204]
train_label = labels[:9*204]
test_data = datas[9*204:]
test_label = labels[9*204:]

#画图验证次数
X = []
Y = []
for i in range(1,51):
    X.append(i)
    model.fit(train_data,train_label,nb_epoch=1)#nb可改轮数
    score = model.evaluate(test_data,test_label,verbose=0)#verbose可改 0／1
    # print('循环'+str(i)+'次的结果')
    # print(score[0],score[1])
    Y.append(score[1])

plt.figure()
plt.xlabel("training rounds")
plt.ylabel("accuracy")
plt.plot(X,Y)
plt.savefig('plot.jpg')

#单次试验
# model.fit(train_data,train_label,nb_epoch=1)
# score = model.evaluate(test_data,test_label,verbose=1)
# print(model.metrics_names)
# print([i for i in score])


# word_input = keras.Input(shape=(15,))
# embedding_layer = Embedding(input_dim= weights.shape[0],output_dim= weights.shape[1], weights=[weights])(word_input)

# print(embedding_layer.shape)
# print(weights.shape)
# print(datas.shape)