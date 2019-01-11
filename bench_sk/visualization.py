'''
Created on Jan 4, 2019

@author: ppc


'''

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

print("kjkj")
# how are pure breed vs bastard dealt with? pure: Breed2 = 0, fine for ml
# print(trainSet[["Breed1", "Breed2"]])
# print(type(trainSet["Name"][14984])) # ??? nan type for Name???


# print("type \n", trainSet.dtypes)
# des = trainSet.describe(include="all")
# with pd.option_context('display.max_columns', None):
#     print(trainSet[1:5])
#     print(des)
    
    
# pl = scatter_matrix(trainSet, figsize=(9,9))
# plt.show() 
# 

# # check corr (pearson), max = 0.7
# cor = trainSet.corr()
# cor.loc[:, :] = np.tril(cor, k=-1)
# cor_pairs = cor.stack()
# cordic = cor_pairs.to_dict()
# cordic_sorted = sorted(cordic.items(), key=lambda kv: abs(kv[1]))
# print(cordic_sorted[-5:])

# 
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(train.drop(["AdoptionSpeed", "PetID"], axis = 1), train["AdoptionSpeed"])
# 
# 
# import graphviz
# 
# featName = train.drop(["AdoptionSpeed", "PetID"], axis = 1).dtypes
#  
# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("iris") 
#  
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                      feature_names=None,  
#                      class_names=None,  
#                      filled=True, rounded=True,  
#                      special_characters=True)  
# graph = graphviz.Source(dot_data)  
# graph.view()



# x_train, x_test, y_train, y_test 
#     = train_test_split(x, y, test_size=0.2, random_state=1)



# print("_____________________________")
# XY = train
# for i in range(1,2):
#     clf = RandomForestClassifier(n_estimators= 10)
#     msk = np.random.rand(len(XY)) < 0.8
#     print("msk sum", msk.sum())
#     XY_train = XY[msk]
#     print("XY_train", XY_train.shape)
#     XY_test = XY[~msk]
#     print("XY_test", XY_test.shape)
#     clf = clf.fit(XY_train.drop(["AdoptionSpeed", "PetID"], axis = 1), XY_train["AdoptionSpeed"])
#     prediction = clf.predict(XY_test.drop(["AdoptionSpeed", "PetID"], axis = 1))
# #     prediction = np.random.rand(len(XY_test)) * 2 + 1
#     score = metrics.mean_squared_error(XY_test["AdoptionSpeed"], prediction)
#     plt.hist(XY_test["AdoptionSpeed"], bins = 10, histtype = "step")
#     plt.ylabel('real') 
#     plt.hist(prediction, bins = 10)
#     plt.ylabel('pred')
#     plt.show()  
#     print(score)
#  
 


