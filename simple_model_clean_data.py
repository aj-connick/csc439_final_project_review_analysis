
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import math
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics
'''
clean_data takes in a file and adds labels to text
'''
def clean_data(file):
    fp = open(file,errors='ignore')
    rescale = []
    label = ""
    text = ""
    #fp2 = open("data_set_full_2.txt","a")
    for i in fp:
        o = i.split(" ")[0]
        label = "no"
        text = "no"
        if '0' == o or '1' == o or '2' == o or '3' == o:
            label = 'bad'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        elif '4' == o or '5' == o or '6' == o or '7' == o:
            label = 'ok'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        elif '8' == o or '9' == o or '10' == o:
            label = 'good'
            text = " ".join(i.split()[1:])
            rescale.append((label,text))
        if label != "no" and text != "no":
            save_rev = label+ " " + text.rstrip()
            #fp2.write(save_rev)
            #fp2.write("\n")
    return rescale
'''
Builds feature matrix
'''
def build_features(text_and_labels):
    text_feature = [text for _,text in text_and_labels]
    text_label = [label for _,label in text_and_labels]
    cv = CountVectorizer(analyzer='word',ngram_range=(1,2),binary=True)
    vector_feature = cv.transform(text_feature)
    return vector_feature.toarray()


class TextToFeatures:
    def __init__(self, texts):
        self.features_names_array = []
        #binary=True, analyzer = text_prep, min_df = 10, max_df = 0.95
        #analyzer='word',ngram_range=(1,1),binary=False
        self.cv = CountVectorizer(binary=True, min_df = 30, max_df = 0.50)
        self.features_names_to_matix = self.cv.fit(texts)
        self.features_names_array = self.cv.get_feature_names()


    def index(self, feature):
        vocabulary = self.cv.vocabulary_
        return vocabulary[feature]
    def __call__(self, texts):
        vector = self.cv.transform(texts)
        return vector.toarray()


class TextToLabels:
    def __init__(self, labels):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        preprocessing.LabelEncoder()
        self.labels_list = list(self.le.classes_)


    def index(self, label):
        return self.labels_list.index(label)

    def __call__(self, labels):
        return self.le.transform(labels)


class Classifier:
    def __init__(self):
        #multi_class='ovr', solver='liblinear'
        #C=1.6,solver='liblinear',penalty='l2',intercept_scaling=.5,class_weight='balanced',tol=.001
        self.classifier = LogisticRegression(solver = 'liblinear', random_state = 42, max_iter=1000)
    def train(self, features, labels):
        self.classifier.fit(features,labels)

    def predict(self, features):
        return self.classifier.predict(features)

'''
build_data reads in a file and divids data 
into labels and data
'''
def build_data(file):
    fp = open(file)
    data = []
    for i in fp:
        k = i.split()
        label = k[0]
        text = " ".join(i.split()[1:])
        data.append((label,text))
    return data
'''
build dev test and train set from larger data set
60 20 20
'''
def cut_data_set(file):
    fp = open(file)
    #size  = len(fp.readline())
    size = 0
    for k in fp:
        size +=1
    fp.close()
    fp = open(file)
    dev = open("data/movie_dev.txt","w")
    test = open("data/movie_test.txt","w")
    train = open("data/movie_train.txt","w")
    #20/20/60 +or- 1
    dev_size = math.floor(size * .20)
    test_size = math.floor(size * .20)
    train_size = math.floor(size * .60)
    fp_pointer = 0
    print(train_size)
    print(test_size)
    print(dev_size)
    print(size)
    for i in fp:
        if fp_pointer <= train_size:
            train.write(i)
        elif fp_pointer <= train_size + test_size:
            test.write(i)
        else:
            dev.write(i) 
        fp_pointer += 1

def main():
    file_train = "data/movie_train.txt"
    file_test = "data/movie_test.txt"
    file_dev = "data/movie_dev.txt"   
    data_train = build_data(file_train)
    data_test = build_data(file_test)
    data_dev = build_data(file_dev)
    train_labels,train_texts = zip(*data_train)
    dev_labels,dev_text = zip(*data_dev)
    test_labels,test_text = zip(*data_test)
    features = TextToFeatures(train_texts)
    labels = TextToLabels(train_labels)
    model = Classifier()
    model.train(features(train_texts), labels(train_labels))
    predicted_dev = model.predict(features(dev_text))
    predicted_test = model.predict(features(test_text))
    devel = labels(dev_labels)
    test = labels(test_labels)
    accuracy_dev = accuracy_score(devel, predicted_dev)
    accuracy_test = accuracy_score(test, predicted_test)
    print("accuracy dev = "+ str(accuracy_dev))
    print("accuracy test = "+ str(accuracy_test))
    print(metrics.classification_report(devel, predicted_dev))
    print(metrics.classification_report(test, predicted_test))
if __name__=="__main__":
    main()
