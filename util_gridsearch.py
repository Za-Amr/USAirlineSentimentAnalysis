from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from time import time

def grid_vect(clf, parameters, X_train, y_train, X_test,y_test, model_name):
    
    pipeline=Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', clf)])

    # grid search for parameters of the model and VectCounter,TF-IDF
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print ("\033[107m" + "Model : "+ str(model_name) +   "\033[0m")
    print("-------------------------------------------------------")
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("Time : done in %0.2fs" % (time() - t0))
    print()

    print("Best CV score: %0.2f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    print("Test score with best_estimator_: %0.2f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Train score with best_estimator_: %0.2f" % grid_search.best_estimator_.score(X_train, y_train))
    print("\n")
    print(" ** Classification Report Test Data ** ")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
    print("** Classification Report Train Data ** ")
    print(classification_report(y_train, grid_search.best_estimator_.predict(X_train)))                    
    return grid_search