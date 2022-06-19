import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

### just change variables below and run
def main():
    ### input variables
    clf = DecisionTreeClassifier()
    clf = KNeighborsClassifier()
    clf = GaussianNB()
    clf = svm.SVC(kernel='poly')  # class_weight={0:train_1, 1:train_0}
    imbalance = 'under' # under for undersampling, over for oversampling, none for no sampling
    path = "github_ratio3.csv"

    df = pd.read_csv(path)
    print("Loaded Dataset\n", df.head())

    print("Dataset information")
    print(df.info())

    ## checking for null values
    print("Any null values in the dataset:", df.isnull().values.any())

    ## remove outliers -> TODO

    ## the balance of the data -> imbalance
    g = sns.countplot(df['author'])
    # g.set_xticklabels(['NOT Author', 'Author'])
    # plt.show()
    print(f'Number of authors:{len(df[df["author"] == 1])}\tnon-authors:{len(df[df["author"] == 0])}')

    ## splitting the data into test and train
    X = df.drop(['author', 'repo', 'developer'], axis=1)
    y = df[['author']]
    print(X.head())
    print(y.head())



    experiment(X, y, clf, imbalance=imbalance, verbose=1)


def experiment(X, y, clf, imbalance='none', verbose=0):
    test_y_total = []
    pred_y_total = []
    sensitivity_total = []
    specificity_total = []
    precision_total = []
    accuracy_total = []
    f1_total = []
    n_splits = 10
    n_authors = len(y[y['author'] == 1])
    if n_authors < n_splits:
        n_splits = n_authors
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)  # keeps the ratio of teh classes in each fold

    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X, y):
        # select rows
        train_X, test_X = X.to_numpy()[train_ix], X.to_numpy()[test_ix]
        train_y, test_y = y.to_numpy().reshape((-1,))[train_ix], y.to_numpy().reshape((-1,))[test_ix]

        # summarize train and test composition
        train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
        test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])

        rus = RandomUnderSampler(random_state=42)  # fit predictor and target variable
        train_X_rus, train_y_rus = rus.fit_resample(train_X, train_y)

        ros = RandomOverSampler(random_state=42)  # fit predictor and target variable
        train_X_ros, train_y_ros = ros.fit_resample(train_X, train_y)

        train_0_rus, train_1_rus = len(train_y_rus[train_y_rus == 0]), len(train_y_rus[train_y_rus == 1])
        train_0_ros, train_1_ros = len(train_y_ros[train_y_ros == 0]), len(train_y_ros[train_y_ros == 1])
        if verbose:
            print('>>Train_Under_Sample: 0:%d, 1=%d \t Train_Over_Sample 0:%d 1:%d' % (
                train_0_rus, train_1_rus, train_0_ros, train_1_ros))

        if imbalance == 'under':
            clf.fit(train_X_rus, train_y_rus)
        elif imbalance == 'over':
            clf.fit(train_X_ros, train_y_ros)
        elif imbalance == 'none':
            clf.fit(train_X, train_y)
        else:
            raise ValueError("Wrong input for imbalance argument")

        pred_y = clf.predict(test_X)

        cf_matrix = confusion_matrix(test_y, pred_y, labels=[0, 1])
        print('test_y')
        print(test_y)
        print('pred_y')
        print(pred_y)
        print("Confusion matrix")
        print(cf_matrix)
        TP = cf_matrix[1][1]
        TN = cf_matrix[0][0]
        FP = cf_matrix[0][1]
        FN = cf_matrix[1][0]

        # for confusion matrix
        test_y_total = np.append(test_y_total, test_y)
        pred_y_total = np.append(pred_y_total, pred_y)

        # Specificity, Sensitivity, Accuracy and F1-measure.
        # Sensitivity = Recall = (True Positive)/(True Positive + False Negative)
        sensitivity = TP / (TP + FN)
        sensitivity_total += [sensitivity]
        print("sensitivity", sensitivity, TP, FN)

        # precision = tp / p = tp / (tp + fp)
        precision = TP / (TP + FP)
        precision_total += [precision]

        # accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy_total += [accuracy]

        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = precision * sensitivity * 2 / (sensitivity + precision)
        f1_total += [f1]

        if verbose:
            print('>Train: 0=%d, 1=%d \t Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

        '''
    print(classification_report(test_y, pred_y, labels=[0,1]))
    print('Sensitivity: {0:0.2f}'.format(sensitivity))
    print('Precision:   {0:0.2f}'.format(precision))
    print('Specificity: {0:0.2f}'.format(specificity))
    print('Accuracy:    {0:0.2f}'.format(accuracy))
    print('f1:          {0:0.2f}'.format(f1))
    '''

    print('')
    report = classification_report(test_y_total, pred_y_total, labels=[0, 1])
    print(report)

    sen = np.array(sensitivity_total).mean()
    print('Sensitivity: {0:0.2f}'.format(sen))

    print('Precision:   {0:0.2f}'.format(np.array(precision_total).mean()))

    '''
    spe = np.array(specificity_total).mean()
    print('Specificity: {0:0.2f}'.format(spe))'''

    acc = np.array(accuracy_total).mean()
    print('Accuracy:    {0:0.2f}'.format(acc))

    f = np.array(f1_total).mean()
    print('f1:          {0:0.2f}'.format(f))

    cf_matrix_total = confusion_matrix(test_y_total, pred_y_total)
    save_path = f'{path.split(".")[0]}_{imbalance}.png'
    showConfusionMatrix(cf_matrix_total, save_path, type_arg='A')


# confusion matrix
def showConfusionMatrix(cf_matrix, save_path, type_arg="A", ):
    if type_arg == 'A':
        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', linewidths=1, cmap='Blues')
        plt.savefig(save_path)
        plt.show()
    elif type_arg == 'S':
        sns.heatmap(cf_matrix, linewidths=1, annot=True, fmt='g')
        plt.savefig(save_path)
        plt.show()

    else:
        raise ValueError("Type argument in CFMatrix can be either 'S' or 'A'")
    return plt


if __name__ == '__main__':

    main(path)
