import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier


def confusionMX(y_test, y_pred, plt_title):
    CMX = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    sns.heatmap(CMX, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return CMX


trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

trainDataFiltered = trainData[(trainData['sc_w'] != 0) & (trainData['sc_h'] != 0) & (trainData['mobile_wt'] != 0) & 
                             (trainData['m_dep'] != 0) & (trainData['px_width'] != 0) & (trainData['px_height'] != 0)]
testDataFiltered = testData[(testData['sc_w'] != 0) & (testData['sc_h'] != 0) & (testData['mobile_wt'] != 0) & 
                            (testData['m_dep'] != 0) & (testData['px_width'] != 0) & (testData['px_height'] != 0)]


def snsGraphstrain(flag, y_pred_svm=0):
    if flag == 0:
        sns.set()
        price_plot = trainDataFiltered['price_range'].value_counts().plot(kind='bar')
        plt.xlabel('price_range')
        plt.ylabel('Count')
        plt.show()

        sns.set()
        ax = sns.displot(data=trainDataFiltered["battery_power"])
        plt.show()

        sns.set()
        ax = sns.displot(data=trainDataFiltered["blue"])
        plt.show()

    else:
        y_pred_svm_series = pd.Series(y_pred_svm)
        sns.set()
        price_plot = y_pred_svm_series.value_counts().plot(kind='bar')
        plt.xlabel('price_range')
        plt.ylabel('Count')
        plt.show()


X = trainDataFiltered.drop(['price_range'], axis=1)
y = trainDataFiltered['price_range']
Xtst = testDataFiltered.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

#---------------------------------------RFV--------------------------------------------------------------------------
def RFV(X_train, y_train, X_valid):
    n_trees = 100
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    random_state = 42
    
    tree_list = []
    for i in range(n_trees):
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + i
        )
        tree.fit(X_train, y_train)
        tree_list.append(tree)

    def predict_proba(X):
        return np.array([tree.predict_proba(X) for tree in tree_list]).mean(axis=0)
    
    def predict(X):
        return np.argmax(predict_proba(X), axis=1)

    y_pred = predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    print('Random Forest Classifier Accuracy Score: ', acc)
    cm_rfc = confusionMX(y_valid, y_pred, 'Random Forest Confusion Matrix')
    return acc

#----------------------------Gaussian NB classifier--------------------------

class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.mean = {}
        self.variance = {}
        self.priors = {}

        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean[str(c)] = np.mean(X_c, axis=0)
            self.variance[str(c)] = np.var(X_c, axis=0)
            self.priors[str(c)] = X_c.shape[0] / X_train.shape[0]

    def pdf(self, x, mean, var):
        exponent = np.exp(-((x-mean)**2 / (2 * var)))
        return (1 / (np.sqrt(2 * np.pi) * np.sqrt(var))) * exponent

    def predict(self, X):
        y_pred = []

        for x in X:
            posteriors = []

            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[str(c)])
                mean = self.mean[str(c)]
                var = self.variance[str(c)]
                class_conditional = np.sum(np.log(self.pdf(x, mean, var)))
                posterior = prior + class_conditional
                posteriors.append(posterior)

            y_pred.append(self.classes[np.argmax(posteriors)])

        return y_pred

def GaussianNB(X_train, y_train, X_valid, y_valid):
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train.values, y_train.values)
    y_pred = gnb.predict(X_valid.values)
    cm_gnb = confusionMX(y_valid, y_pred, 'Gaussian Naive Bayes Confusion Matrix')
    acc = accuracy_score(y_valid, y_pred)
    print('Gaussian Naive Bayes Classifier Accuracy Score: ', acc)
    return acc


        

     
#----------------------------
# Import required libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define SVM function
def SVM(X_train, y_train, X_valid, y_valid):
    
    # Create SVM classifier with linear kernel
    svm_classifier = SVC(kernel='linear', random_state=0)
    
    # Fit the classifier on training data
    svm_classifier.fit(X_train, y_train)
    
    # Calculate validation accuracy
    svm_acc = accuracy_score(y_valid, svm_classifier.predict(X_valid))
    
    # Use svm_classifier to make predictions on validation and test data
    y_pred_svm_valid = svm_classifier.predict(X_valid)
    y_pred_svm_test = svm_classifier.predict(Xtst)

    return svm_acc, y_pred_svm_valid, y_pred_svm_test

# Call SVM function with train and validation data
svm_acc, y_pred_svm_valid, y_pred_svm_test = SVM(X_train, y_train, X_valid, y_valid)

# Plot confusion matrix for validation data
sns.heatmap(confusion_matrix(y_valid, y_pred_svm_valid), annot=True, cmap="Blues")
plt.title('SVM Confusion Matrix (Validation Data)')
plt.show()

# Call RFV function with train and validation data
acc_rfc = RFV(X_train, y_train, X_valid)

# call knn function with train and validation data

#acc_knn = knn(X_train, y_train, X_valid, y_valid)

# Call GaussianNB function with train and validation data
acc_gnb = GaussianNB(X_train, y_train, X_valid, y_valid)




# Save predictions on test data to a CSV file
tst_data_price_range = pd.Series(y_pred_svm_test, name='price_range')
result = pd.concat([testDataFiltered['id'], tst_data_price_range], axis=1)
result.to_csv('prediction.csv', index=False)
#----------------------------------KNN Classifier------------------------------------
# def knn(X_train, y_train, X_valid, y_valid):
#     import numpy as np
    
#     # calculate pairwise distances between all training and validation samples
#     dists = np.sqrt(np.sum((X_train[:, np.newaxis, :] - X_valid) ** 2, axis=2))
    
#     # predict labels for validation set using k-nearest neighbors
#     num_neighbors = 3
#     closest_idxs = np.argsort(dists, axis=0)[:num_neighbors]
#     y_pred_knn = np.mode(y_train[closest_idxs], axis=0)
    
#     # calculate accuracy score and confusion matrix
#     accuracy = np.mean(y_pred_knn == y_valid)
#     confusion_matrix = calculate_confusion_matrix(y_valid, y_pred_knn)
    
#     print('KNN Classifier Accuracy Score: ', accuracy)
#     print('KNN Confusion Matrix:\n', confusion_matrix)
#     print("dghsvh")

def calculate_confusion_matrix(y_true, y_pred, num_classes):
        unique_labels = np.unique(y_true)
        num_labels = len(unique_labels)
        confusion_matrix = np.zeros((num_labels, num_labels))
        for i in range(num_labels):
            true_idxs = np.where(y_true == unique_labels[i])[0]
            for j in range(num_labels):
                pred_idxs = np.where(y_pred == unique_labels[j])[0]
                confusion_matrix[i,j] = len(set(true_idxs) & set(pred_idxs))
        return confusion_matrix

# ---------------------------------------------------------------------------------------------------------------------


if acc_rfc > acc_gnb: #& acc_rfc > svm_acc:
    best_classifier = 'Random Forest Classifier'
elif svm_acc  > acc_gnb:
    best_classifier = 'Gaussian Naive Bayes Classifier'
else:
    best_classifier = 'Support Vector Machine Classifier'

print('The best classifier is:', best_classifier)