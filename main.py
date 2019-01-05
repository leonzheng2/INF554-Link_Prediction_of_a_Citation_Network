import csv
import nltk
import numpy as np
import networkx as nx
import random
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

"""
    Possible changes
 - Lot of work to do on possible features from node_information.csv
 - Should try with nice random forests and XGBoost
 - Add other graph features

 Tiny details:
 -  Put binary = True in CountVectorizer constuction

"""#
with open("data/node_information.csv", "r") as f:
    file = csv.reader(f)
    node = list(file)

ID = [i[0] for i in node]
year=[i[1] for i in node]
title=[i[2] for i in node]
authors=[i[3] for i in node]
name_journal=[i[4] for i in node]
abstract=[i[5] for i in node]


"""
One_hot vectors on abstract (usefull for co_occurence computations in features construction function)
"""
one_hot = CountVectorizer(stop_words="english")
one_hot_matrix = one_hot.fit_transform(abstract)#.todense()
one_hot_matrix = one_hot_matrix.toarray()
print(one_hot_matrix.shape)
np.set_printoptions(threshold=np.nan)
print(sum(one_hot_matrix[1]))

"""
One_hot vectors on authors (usefull for co_occurence computations in features construction function)
"""
onehot_authors= CountVectorizer()
onehot_authors_matrix=onehot_authors.fit_transform(authors)
onehot_authors_matrix = onehot_authors_matrix.toarray()
print(onehot_authors_matrix.shape)
print(onehot_authors.get_feature_names())

"""
One_hot vectors on titles (usefull for co_occurence computations in features construction function)
"""
onehot_titles= CountVectorizer()
onehot_titles_matrix=onehot_titles.fit_transform(title)
onehot_titles_matrix = onehot_titles_matrix.toarray()
print(onehot_titles_matrix.shape)
print(onehot_titles.get_feature_names())

"""
TF-IDF cosine similarity
"""
tfidf_vectorizer = TfidfVectorizer(min_df = 0, max_df = 1, use_idf = True, stop_words="english")
features_TFIDF = tfidf_vectorizer.fit_transform(abstract)
tfidf_matrix = features_TFIDF.toarray()


"""
23-gram co-occurence ONLY USE THIS FOR THE FINAL GO. IT IS VERY VERY COMPUTATIONALLY DEMANDING
"""
#one_got_23gram = CountVectorizer(binary = True, stop_words = "english", ngram_range = (2,3))
#one_got_23gram_matrix = one_got_23gram.fit_transform(abstract)


#####co_occurence computation (VERY EXPENSIVE)
# co_occurance_abstract=np.dot(cv_matrix,np.transpose(cv_matrix))
# co_occurance_abstract=np.dot(cv_matrix,cv_matrix.T)
"""
construction of the graph
"""
testtrain=0.9
with open("data/training_set.txt", "r") as f:
    file =csv.reader(f, delimiter='\t')
    set_file=list(file)
set= [values[0].split(" ") for values in set_file]
#creates the graph
G=nx.Graph()
#adds the list of papers' IDs
G.add_nodes_from(ID)
#adds the corresponding links between the paper (training set), links when link_test==1
##we only keep 90% of the set for testing perpose
for ID_source_train,ID_sink_train,link_train in set[:int(len(set)*testtrain)]: #[:int(len(set)*testtrain)]
    if link_train=="1":
        G.add_edge(ID_source_train,ID_sink_train)
#G.edges() to print all the edges

#check the number of edges
# G.number_of_edges()

#########
def features(paper1,paper2):
    """
        outputs the array of the features to input for paper1 and paper 2 comparison
    """
    idx_paper1,idx_paper2=ID.index(str(paper1)),ID.index(str(paper2))
    # print(abstract[ID.index(str(paper1))])
    # print(abstract[idx_paper1])

    #features from info of the nodes
    co_occurence_abstract=np.dot(one_hot_matrix[idx_paper1],one_hot_matrix[idx_paper2].T)
    same_authors=np.dot(onehot_authors_matrix[idx_paper1],onehot_authors_matrix[idx_paper2].T)
    co_occurence_title=np.dot(onehot_titles_matrix[idx_paper1],onehot_titles_matrix[idx_paper2].T)

    #tfidf cosine similarity
    tf1 = tfidf_matrix[idx_paper1,:]#.toarray() in case tfidf mat is so large that it's stored as a sparse matrix
    tf2 = tfidf_matrix[idx_paper2,:]#.toarray() in case tfidf mat is so largs that it's stared as a sparse matrix
    tfidf_sim = np.dot(tf1,tf2)/max((np.linalg.norm(tf1)*np.linalg.norm(tf2)),1e-16)

    multiplied_idf = np.dot(tf1,tf2)
    tfidf_max = np.amax(multiplied_idf)

    #VERY COMPUTATIONALLY EXPENSIVE
    #twothree_gram = np.sum(one_got_23gram_matrix[idx_paper1].toarray() * one_got_23gram_matrix[idx_paper2].toarray())

    same_journal = int(name_journal[idx_paper1] == name_journal[idx_paper2])
    try:
        distance=len(nx.shortest_path(G, str(paper1), str(paper2)))
    except:
        distance=0
    years_diff=int(year[idx_paper1])-int(year[idx_paper2])
    ## features over the graph
    jaccard = nx.jaccard_coefficient(G, [(str(paper1), str(paper2))])
    for u, v, p in jaccard:
        jaccard_coef= p
    adamic_adar=nx.adamic_adar_index(G, [(str(paper1), str(paper2))])
    for u, v, p in adamic_adar:
        adamic_adar_coef= p
    pref_attachement = nx.preferential_attachment(G, [(str(paper1), str(paper2))])
    for u, v, p in pref_attachement:
        pref_attachement_coef= p
    common_neig=len(sorted(nx.common_neighbors(G, str(paper1), str(paper2))))
    return [co_occurence_abstract,same_authors,co_occurence_title,distance,
    years_diff,jaccard_coef,adamic_adar_coef,pref_attachement_coef,common_neig,tfidf_sim,tfidf_max,same_journal]#,twothree_gram] #

train_features=[]
y_train=[]
print("Features costruction for Learning...")
step=0
for source,sink,link in set[:int(len(set)*testtrain)]:
    step+=1
    if step%1000==0:    print("Step:",step,"/",int(len(set)*testtrain))
    train_features.append(features(source,sink))
    y_train.append(link)
train_features=np.array(train_features)
train_features = preprocessing.scale(train_features)
y_train=np.array(y_train)


test_features=[]
y_test=[]
print("Features costruction for Testing...")
step=0
for source,sink in set[int(len(set)*testtrain):len(set)]: ##set_test: ##
    step+=1
    if step%1000==0:    print("Step:",step,"/",len(set)-int(len(set)*testtrain))
    test_features.append(features(source,sink))
    # y_test.append(link)
test_features=np.array(test_features)
test_features = preprocessing.scale(test_features)
# y_test=np.array(y_test)

#### For kaggle submission
# with open("data/testing_set.txt", "r") as f:
#     file =csv.reader(f, delimiter='\t')
#     set_file=list(file)
# set_test= [values[0].split(" ") for values in set_file]
#### than make the changes in the for loops
# #MLP classsifier
# features=(0,1,2,4,5,6,7,8)
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='adam', alpha=1e-3,
#             hidden_layer_sizes=(15, 10), random_state=1)
# clf = clf.fit(train_features[:,features], y_train)
#
#
# pred = list(clf.predict(test_features[:,features]))
# # ids=[i for i in range(len(set_test))]
# predictions= zip(range(len(set_test)), pred)
#
# # write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
#
# with open("predictions.csv","w",newline="") as pred1:
#     fieldnames = ['id', 'category']
#     csv_out = csv.writer(pred1)
#     csv_out.writerow(fieldnames)
#     for row in predictions:
#         csv_out.writerow(row)
#
# #### scored 0.962876 on 90% 10% split against 0.96630 on kaggle





"""
Model phase (training and testing are in the same paragraphs for one method)
"""

# Prediction rate with SVM
classifier = svm.LinearSVC()
classifier.fit(train_features, y_train)
pred = list(classifier.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate:",success_rate)


#prediction rate with RF
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with RF:",success_rate)


#prediction using logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_features, y_train)
pred = list(model.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with Logistic regression:",success_rate)

#MLP classsifier (best so far)
features=(0,1,2,4,5,6,7,8)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-3,
            hidden_layer_sizes=(15, 10), random_state=1)
clf = clf.fit(train_features[:,features], y_train)
pred = list(clf.predict(test_features[:,features]))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with NN MLP with Adam:",success_rate)
# #MLP classsifier
# features=(0,1,2,4,5,6,7,8)
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='adam', alpha=1e-3,
#             hidden_layer_sizes=(15, 10), random_state=1)
# clf = clf.fit(train_features[:,features], y_train)



# KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(3)
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with KNN k=3:",success_rate)


# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with AdaBoost:",success_rate)


#Gaussian processs
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# clf=GaussianProcessClassifier(1.0 * RBF(1.0))
# clf = clf.fit(train_features, y_train)
# pred = list(clf.predict(test_features))
# success_rate=sum(y_test==pred)/len(pred)
# print("Success_rate with Gaussian process:",success_rate)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with Naive bayes:",success_rate)

# different SVMs
from sklearn.svm import SVC
clf = SVC(kernel="rbf")
clf = clf.fit(train_features, y_train)
pred = list(clf.predict(test_features))
success_rate=sum(y_test==pred)/len(pred)
print("Success_rate with SVM:",success_rate)

# xgboost
import xgboost as xgb
xg = xgb.XGBClassifier(max_depth=2,n_estimaters = 200)
xg.fit(train_features,y_train)
pred = list(xg.predict(test_features))
success_rate=sum(ytest==pred)/len(pred)
print("Success_rate with XGBoost:",success_rate)
