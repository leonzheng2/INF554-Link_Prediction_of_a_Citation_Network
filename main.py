import csv
import nltk
import numpy as np
import networkx as nx
import random
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


"""
    Guillaume Dufau (guillaume.dufau@polytechnique.edu)
    Christopher Murray (christophe.murray@polytechnique.edu)
    Léon Zheng (leon.zheng@polytechnique.edu)
"""

 ## Loads the node information
with open("data/node_information.csv", "r") as f:
    file = csv.reader(f)
    node = list(file)

ID = [int(i[0]) for i in node]
year=[int(i[1]) for i in node]
title=[i[2] for i in node]
authors=[i[3] for i in node]
name_journal=[i[4] for i in node]
abstract=[i[5] for i in node]

"""
    Construction of the graph
"""
testtrain=0.9 #ratio for the hold out cross validation
with open("data/training_set.txt", "r") as f:
    file =csv.reader(f, delimiter='\t')
    set_file=list(file)
set= np.array([values[0].split(" ") for values in set_file]).astype(int)

## Creates the oriented graph
diG=nx.DiGraph()
#adds the list of papers' IDs
diG.add_nodes_from(ID)
#adds the corresponding links between the paper (training set), links when link_test==1
for ID_source_train,ID_sink_train,link_train in set[:int(len(set)*testtrain)]:
    if link_train==1:
        diG.add_edge(ID_source_train,ID_sink_train)

  ## Checks the number of edges and creates the non-oriented graph G
G = nx.Graph(diG)
print(diG.nodes)


"""
    Construction of the features
"""
    ## Useful graph-based features computed at once
page_rank = nx.pagerank_scipy(G)
hub_score, authority_score = nx.hits(G)

    ##One_hot vectors on abstract (usefull for co_occurence computations in features construction function)
one_hot = CountVectorizer(stop_words="english")
one_hot_matrix = one_hot.fit_transform(abstract)#.todense()

    ## One_hot vectors on authors (usefull for co_occurence computations in features construction function)
onehot_authors= CountVectorizer()
onehot_authors_matrix=onehot_authors.fit_transform(authors)

    ##One_hot vectors on titles (usefull for co_occurence computations in features construction function)
onehot_titles= CountVectorizer()
onehot_titles_matrix=onehot_titles.fit_transform(title)

    ##TF-IDF cosine similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(abstract)

    ## 23-gram co-occurence. VERY VERY COMPUTATIONALLY DEMANDING
#one_got_23gram = CountVectorizer(binary = True, stop_words = "english", ngram_range = (2,3))
#one_got_23gram_matrix = one_got_23gram.fit_transform(abstract)


def features(paper1,paper2):
    """
        Outputs the array of the features to input in the prediction models
    """
    idx_paper1,idx_paper2=ID.index(paper1),ID.index(paper2)

    ## Features from contextual information of the nodes
    co_occurence_abstract=np.dot(one_hot_matrix[idx_paper1],one_hot_matrix[idx_paper2].T).toarray()[0][0]
    same_authors=np.dot(onehot_authors_matrix[idx_paper1],onehot_authors_matrix[idx_paper2].T).toarray()[0][0]
    co_occurence_title=np.dot(onehot_titles_matrix[idx_paper1],onehot_titles_matrix[idx_paper2].T).toarray()[0][0]

    #tfidf cosine similarity
    tf1 = tfidf_matrix[idx_paper1]# in case tfidf mat is so large that it's stored as a sparse matrix
    tf2 = tfidf_matrix[idx_paper2]# in case tfidf mat is so largs that it's stared as a sparse matrix
    tfidf_sim = cosine_similarity(tf1, tf2)[0][0]

    #VERY COMPUTATIONALLY EXPENSIVE
    #twothree_gram = np.sum(one_got_23gram_matrix[idx_paper1].toarray() * one_got_23gram_matrix[idx_paper2].toarray())

    same_journal = int(name_journal[idx_paper1] == name_journal[idx_paper2])

    ## Irrelevant tested feature
    # try:
    #     distance=1/len(nx.shortest_path(G, paper1, paper2))
    # except:
    #     distance=0
    
    years_diff=int(year[idx_paper1])-int(year[idx_paper2])

    ## Features over the graph
    jaccard = nx.jaccard_coefficient(G, [(paper1, paper2)])
    for u, v, p in jaccard:
        jaccard_coef= p
    adamic_adar=nx.adamic_adar_index(G, [(paper1, paper2)])
    for u, v, p in adamic_adar:
        adamic_adar_coef= p
    pref_attachement = nx.preferential_attachment(G, [(paper1, paper2)])
    for u, v, p in pref_attachement:
        pref_attachement_coef= p
    common_neig=len(sorted(nx.common_neighbors(G, paper1, paper2)))

    ## features over the directed graph
    triad_features = [0.0]*8
    for w in sorted(nx.common_neighbors(G, paper1, paper2)):
        if G.has_edge(paper1, w) and G.has_edge(w, paper2):
            triad_features[0]+=1
        if G.has_edge(paper1, w) and G.has_edge(paper2, w):
            triad_features[1]+=1
        if G.has_edge(w, paper1) and G.has_edge(w, paper2):
            triad_features[2] += 1
        if G.has_edge(w, paper1) and G.has_edge(paper2, w):
            triad_features[3] += 1
    for i in range(4, 8):
        if triad_features[i-4]!=0:
            triad_features[i] = triad_features[i-4]/common_neig

    #VERY COMPUTATIONALLY EXPENSIVE
    ## Katz similarity (Very expansive) -> not used a final feature
    # katz = 0
    # beta = 0.005
    # path_length = []
    # for path in nx.all_simple_paths(G, source=source, target=sink, cutoff=3):
    #     path_length.append(len(path))
    # a = np.array(path_length)
    # unique, counts = np.unique(a, return_counts=True)
    # dict_katz = dict(zip(unique, counts))
    # for length in dict_katz:
    #     katz += dict_katz[length] * beta ** length * length

    ## Sum up of all features
    degree_features = [diG.in_degree(paper1), diG.out_degree(paper1), diG.in_degree(paper2), diG.out_degree(paper2)]
    heuristic_graph_features = [jaccard_coef, adamic_adar_coef, pref_attachement_coef, common_neig] # one ccan add if computed katz
    node_info_features = [co_occurence_abstract, same_authors, co_occurence_title, years_diff, same_journal, tfidf_sim] # + [twothree_gram] if computed #

    heuristic_graph_features.append([page_rank[paper2],hub_score[paper1],authority_score[paper2]])

    return node_info_features + heuristic_graph_features + degree_features + triad_features  ## 25 features in total

"""
    Build the data sets based on given files
"""
## To save the X_train,y_train matrices. Expansive to compute
saved = False

train_features= []
if saved:
    train_features= np.load("./save/90.npy")
y_train=[]
print("Features construction for Learning...")
step=0
for source,sink,link in set[:int(len(set)*testtrain)]:
    step+=1
    if step%1000==0:    print("Step:",step,"/",int(len(set)*testtrain))
    if not saved:
        train_features.append(features(source,sink))
    y_train.append(link)
train_features=np.array(train_features)
train_features = preprocessing.scale(train_features)
y_train=np.array(y_train)
if not saved:
    np.save("./save/train_features.npy", train_features)

test_features=[]
if saved:
    test_features=np.load("./save/10.npy")
y_test=[]
print("Features construction for Testing...")
step=0
for source,sink,link in set[int(len(set)*testtrain):len(set)]: ##set_test: ##
    step+=1
    if step%1000==0:    print("Step:",step,"/",len(set)-int(len(set)*testtrain))
    if not saved:
        test_features.append(features(source,sink))
    y_test.append(link)
test_features=np.array(test_features)
test_features = preprocessing.scale(test_features)
y_test=np.array(y_test)
if not saved:
    np.save("./save/test_features.npy", test_features)


"""
    Data visualization
"""

    ## Features importance plot
model =RandomForestClassifier(n_estimators=100, max_depth=9,
                             random_state=0)
model.fit(train_features, y_train)
(pd.Series(model.feature_importances_)
.nlargest(7)
.plot(kind='barh',title="Features importance according to RandomForestClassifier"))

    ## Correlation heat map
plt.imshow(np.corrcoef(train_features.T), cmap='hot', interpolation='nearest')
plt.show()

    ## PCA decomposition
pca = PCA(n_components=2)
X = pca.fit_transform(train_features)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,0], X[:,1], c=y_train)
ax.set_xlabel('1st dimension')
ax.set_ylabel('2nd dimension')
ax.set_title("Vizualization of the PCA decomposition (2D)")
plt.show()
    ##Vizualize selected features on initial data
# feat = (i,j) #select the features to vizualize
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(train_features[:,feat[0]], train_features[:,feat[1]], c=y_train, alpha=0.8)
# ax.set_xlabel(f'Dimension {feat[0]}')
# ax.set_ylabel(f'Dimension {feat[1]}')
# ax.set_title("Vizualization of the features")
# plt.show()

    ## Another way to see least useful features (but they still change positively the prediictions quality)
print(train_features[0])
X_new = SelectKBest(f_classif, k=20).fit_transform(train_features, y_train)
print(X_new[0])

"""
    Model construction + prediction using our cross validation to select best models
    + plotting example for hyper-parameters choices
"""

def execute_prediction(classifier, classifier_name):
    classifier.fit(train_features, y_train)
    y_pred = list(classifier.predict(test_features))
    f1 = f1_score(y_test, y_pred)
    success_rate = sum(y_pred==y_test)/len(y_pred)
    print(f"F1 score for {classifier_name}: {f1}")
    print(f"Success rate for {classifier_name}: {success_rate}")
    return f1

    ## MLP (with parameters corresponding to one of 2 best kaggle scores)
test_features = preprocessing.scale(test_features)
train_features = preprocessing.scale(train_features)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1.74e-4,activation="relu",
            hidden_layer_sizes=(65,18), tol=5e-5, max_iter=250, verbose=1)
clf = clf.fit(train_features, y_train)
y_pred = list(clf.predict(test_features))
f1 = f1_score(y_test, y_pred)
print(f"F1 score for MLP:", f1)

### Fine tune parameters

# ## Regularization term
# import matplotlib.pyplot as plt
# alphas = [1.735e-4, 1.74e-4, 1.745e-4, 1.75e-4, 1.755e-4]
# f1_scores = []
# for alpha in alphas:
#     from sklearn.neural_network import MLPClassifier
#     clf = MLPClassifier(solver='adam', alpha=alpha, activation="relu",
#                         hidden_layer_sizes=(65, 18), tol=1e-4, max_iter=150, verbose=1)
#     f1 = execute_prediction(clf, f"NN MLP with Adam, alpha={alpha}")
#     f1_scores.append(f1)
# plt.plot(alphas, f1_scores)
# plt.xlabel("Alpha regularization")
# plt.ylabel("F1 score")
# plt.title("F1 score with NN MLP with Adam")
# plt.show()
# Best alpha: 1.74 for hidden_layers_size (65, 21)

    ## Example of loop for best hyper-parameters (here best NN structure)
layers = []
f1_scores = []
for size2 in range(12,31,3):
    layers.append(size2)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, activation="relu",
                        hidden_layer_sizes=(65, size2), tol=1e-4, max_iter=150, verbose=1)
    f1 = execute_prediction(clf, f"NN MLP with Adam, size2={size2}")
    f1_scores.append(f1)
plt.plot(layers, f1_scores)
plt.xlabel("size2")
plt.ylabel("F1 score")
plt.title("F1 score with NN MLP with Adam")
plt.show()
### Best hidden_layer_size for alpha=1e-5: (65, 21)

# ## Tolerance
# import matplotlib.pyplot as plt
# tols = [1e-4, 5e-5, 1e-5, 5e-6]
# f1_scores = []
# for tol in tols:
#     tols.append(tol)
#     from sklearn.neural_network import MLPClassifier
#     clf = MLPClassifier(solver='adam', alpha=4e-4, activation="relu", tol= tol, max_iter=250,
#                 hidden_layer_sizes=(65,18), verbose=1)
#     f1 = execute_prediction(clf, f"NN MLP with Adam, tol={tol}")
# plt.semilogx(tols, f1_scores)
# plt.xlabel("Tolerance")
# plt.ylabel("F1 score")
# plt.title("F1 score with NN MLP with Adam")
# plt.show()
# ### Best Tolerance rate = 5e-5


"""
    Other tested models (with tunned hyper-parameters)
"""

# # Prediction rate with SVM
# execute_prediction(svm.LinearSVC(), "SVM")
#
# #prediction rate with RF
# execute_prediction(RandomForestClassifier(n_estimators=100), "Random Forest")
#
# #prediction using logistic regression
# from sklearn.linear_model import LogisticRegression
# execute_prediction(LogisticRegression(), "Logistic Regression")
#
# #MLP classsifier (best so far)
# from sklearn.neural_network import MLPClassifier
# execute_prediction(MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(15, 10), random_state=1), "NN MLP with Adam")
#
# # KNN
# from sklearn.neighbors import KNeighborsClassifier
# execute_prediction(KNeighborsClassifier(3), "KNN k=3")
#
#
# # AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# execute_prediction(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=300, algorithm='SAMME.R'), "AdaBoost")
#
#
# #Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# execute_prediction(GaussianNB(), "Naive Bayes")
#
# # # different SVMs
# # ## doesn't converge
# # from sklearn.svm import SVC
# # execute_prediction(SVC(kernel="rbf"), "SVM (rbf kernel)")
#
# xgboost
# import xgboost as xgb
# from sklearn.metrics import f1_score

# # max_score = 0.9580077524149387
# max_depth=9
# min_child_weight=2
# gamma=3000
# colsample_bytree = 0.35
# subssample = 0.8
# reg_alpha = 3
# learning_rate = 0.0001
# best_f1 = 0
# best_params = {}
# for learning_rate in [10**(-k) for k in range(4,7)]:
#     model = xgb.XGBClassifier(silent=True,
#                               scale_pos_weight=1,
#                               learning_rate=0.0001,
#                               colsample_bytree=0.35,
#                               subsample=0.8,
#                               objective='binary:logistic',
#                               n_estimators=100,
#                               max_depth=9,
#                               min_child_weight=2,
#                               gamma=3000,
#                               reg_alpha=3,
#                               nthread=8)
#     f1 = execute_prediction(model, f"XBGBoost with learning_rate={learning_rate}")
#     if f1 > best_f1:
#         best_params = {learning_rate}
#         best_f1 = f1
# print("Best parameters:", best_params)
# print("Best score:", best_f1)

# clf = xgb.XGBClassifier(silent=False,
#                               scale_pos_weight=1,
#                               learning_rate=0.0001,
#                               colsample_bytree=0.35,
#                               subsample=0.8,
#                               objective='binary:logistic',
#                               n_estimators=100,
#                               max_depth=1,
#                               min_child_weight=2,
#                               gamma=3000,
#                               reg_alpha=3,
#                               nthread=8)
# execute_prediction(clf, "XGBoost")

# Example of code for fine tuning the parameters of XGBoost function.
# from sklearn.model_selection import GridSearchCV
# param_test1 = {
#  'max_depth':range(3,4,2),
#  'min_child_weight':range(1,2,2)
# }
# gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=140, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', scale_pos_weight=1, seed=27),
#  param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5, verbose=3)
# gsearch1.fit(train_features, y_train)
# print("Grid score:", gsearch1.grid_scores_)
# print("Best params:", gsearch1.best_params_)
# print("Best score:", gsearch1.best_score_)
