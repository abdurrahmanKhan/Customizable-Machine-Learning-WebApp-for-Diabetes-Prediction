import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
st.set_option('deprecation.showPyplotGlobalUse', False) 

def main():
    st.title("Diabetes Classifier App")
    st.sidebar.title('Diabetes Classifier Web App')
    st.markdown("A fully intercative web application which provides you flexibility to tune your hyperparameters and get the results using the parameters of your own choice. Also, if you are a person from different domain NO WORRIES! this application suggests you the best parameters, all you gotta do is select the algorithm of your choice.")
    st.markdown("**Explore and See for yourself**")
    st.write('')


    @st.cache(allow_output_mutation=True)
    def load_data():
        data= pd.read_csv('clean.csv')
        data = data.drop(['Unnamed: 0'],axis=True)
        return data

    @st.cache(persist=True)
    def split(df):
        x = df.drop(['Outcome'],axis=True)
        y = df['Outcome']
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
        return x_train, x_test, y_train, y_test
    def plot_metrics(metrics_list,actual,predicted):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("**Confusion Matrix**")
            st.dataframe(confusion_matrix(actual,predicted))
            
        if 'Classification Report' in metrics_list:
            st.subheader("**Classification Report**")
            st.text(classification_report(actual,predicted))


    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    class_names = ['0','1']
    if st.button("Split the data & Scale using Min-Max Scaler"):
        st.write("Dimension of x_train dataset: ", x_train.shape)
        st.write("Dimension of y_train dataset: ", y_train.shape)
        st.write("Dimension of x_test dataset: ", x_test.shape)
        st.write("Dimension of y_test dataset: ", y_test.shape)
        scaler=MinMaxScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
 

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine",'Logistic Regression','Decision Tree','Random Forest','XG Boost','K-Nearest Neighbour'))
    st.write('')
    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameter")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step = 0.01,key='C')
        kernel = st.sidebar.radio("Kernel",('linear','rbf','poly','sigmoid'),key = "kernel")
        #svc_params = {'C': [0.001, 0.01, 0.1, 1],'kernel': [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]}
        #svc_model = GridSearchCV(SVC(random_state=0), svc_params, cv=5)
        #svc_model.fit(x_train, y_train)
        #st.write(svc_model.best_params_)
        st.subheader("**Parameters optimization ran on**")
        st.write("**C: **[0.001, 0.01, 0.1, 1]")
        st.write("**kernel: **[ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        st.subheader("Best Params for SVM")
        st.write("**C :** 1")
        st.write("**kernel :** linear")
        st.write("")

        if st.checkbox("Custom Input"):
            Pregnancies   = st.text_input('Number of pregnancies - [Range: Ideally between 1 - 12]')
            Glucose   = st.text_input('Glucose Level - [Range(mg/dL): Ideally upto 140 ]')
            BloodPressure = st.text_input('BloodPressure Level - [Range(mm Hg): 90/60 - 120/80]')
            SkinThickness  = st.text_input('Skin thickness - [Range(mm) - Ideally upto 90]')
            Insulin = st.text_input('Insulin Level - [Range(mu U/ml): Ideally between 3-25 ]')
            BMI = st.text_input('BMI - [Range : Ideally between 15-35 ]')
            DiabetesPedigreeFunction  = st.text_input('DiabetesPedigreeFunction - [Range: Ideally upto 3.0000]')
            Age = st.text_input('Age - [Range: Ideally between 1 - 105]')

            x_test = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

            model = SVC(C=C,kernel = kernel,random_state=0,probability=True)
            model.fit(x_train,y_train)
            y_pred = round(model.predict_proba(x_test)[0][1],2)
            if st.button("Predict"):
                st.markdown("The probability of having diabets is :")
                st.success(y_pred)

            
        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Support Vector Machine Results")
            st.write('')
            model = SVC(C=C,kernel = kernel,random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            st.write("**Accuracy : **",round(accuracy,2))
            y_pred = model.predict(x_test)
            plot_metrics(metrics,y_test,y_pred)
  
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameter")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step = 0.01,key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
        solver = st.sidebar.radio("Solver",('liblinear', 'saga'),key = "solver")
        penalty = st.sidebar.radio("penalty",('l1', 'l2'),key = "penalty")
        #log_params = {'penalty':['l1', 'l2'],'C': [0.0001, 0.001, 0.01, 0.1, 1,], 'solver':['liblinear', 'saga'] } 
        #log_model = GridSearchCV(LogisticRegression(random_state=0), log_params, cv=5) 
        #log_model.fit(x_train, y_train)
        #st.write(log_model.best_params_)
        st.subheader("**Parameters optimization ran on**")
        st.write("**penalty : **['l1', 'l2']")
        st.write("**C : **[0.0001, 0.001, 0.01, 0.1, 1,]")
        st.write("**solver : **['liblinear', 'saga']")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        st.subheader("Best Params for Logistic Regression")
        st.write("**C :** 1")
        st.write("**penalty :** l1")
        st.write("**solver :** liblinear")
        st.write("")
        if st.checkbox("Custom Input"):
            Pregnancies   = st.text_input('Number of pregnancies - [Range: Ideally between 1 - 12]')
            Glucose   = st.text_input('Glucose Level - [Range(mg/dL): Ideally upto 140 ]')
            BloodPressure = st.text_input('BloodPressure Level - [Range(mm Hg): 90/60 - 120/80]')
            SkinThickness  = st.text_input('Skin thickness - [Range(mm) - Ideally upto 90]')
            Insulin = st.text_input('Insulin Level - [Range(mu U/ml): Ideally between 3-25 ]')
            BMI = st.text_input('BMI - [Range : Ideally between 15-35 ]')
            DiabetesPedigreeFunction  = st.text_input('DiabetesPedigreeFunction - [Range: Ideally upto 3.0000]')
            Age = st.text_input('Age - [Range: Ideally between 1 - 105]')

            x_test = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

            model = LogisticRegression(C=C,max_iter = max_iter,solver = solver,penalty =penalty,random_state=0)
            model.fit(x_train,y_train)
            y_pred = round(model.predict_proba(x_test)[0][1],2)
            if st.button("Predict"):
                st.markdown("The probability of having diabets is :")
                st.success(y_pred)

        
        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter = max_iter,solver = solver,penalty =penalty,random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            st.write("**Accuracy : **",round(accuracy,2))
            y_pred = model.predict(x_test)
            plot_metrics(metrics,y_test,y_pred)


    if classifier == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameter")
        criterion = st.sidebar.radio("Criterion",('gini', 'entropy'),key = "criterion")
        splitter = st.sidebar.radio("splitter",('random', 'best'),key = "splitter")
        max_depth = st.sidebar.radio("Max Depth",(3, 5, 7, 9, 11, 13),key = "max_depth")
        #dt_params = {'criterion' : ['gini', 'entropy'],'splitter': ['random', 'best'], 'max_depth': [3, 5, 7, 9, 11, 13]}
        #dt_model = GridSearchCV(DecisionTreeClassifier(random_state=0), dt_params, cv=5) 
        #dt_model.fit(x_train, y_train)
        #st.write(dt_model.best_params_)
        st.subheader("**Parameters optimization ran on**")
        st.write("**criterion : **['gini', 'entropy']")
        st.write("**splitter: **['random', 'best']")
        st.write("**max_depth: **[3, 5, 7, 9, 11, 13]")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        st.subheader("Best Params for Decision Tree")
        st.write("**criterion :** gini")
        st.write("**max_depth :** 3")
        st.write("**splitter :** random")

        if st.checkbox("Custom Input"):
            Pregnancies   = st.text_input('Number of pregnancies - [Range: Ideally between 1 - 12]')
            Glucose   = st.text_input('Glucose Level - [Range(mg/dL): Ideally upto 140 ]')
            BloodPressure = st.text_input('BloodPressure Level - [Range(mm Hg): 90/60 - 120/80]')
            SkinThickness  = st.text_input('Skin thickness - [Range(mm) - Ideally upto 90]')
            Insulin = st.text_input('Insulin Level - [Range(mu U/ml): Ideally between 3-25 ]')
            BMI = st.text_input('BMI - [Range : Ideally between 15-35 ]')
            DiabetesPedigreeFunction  = st.text_input('DiabetesPedigreeFunction - [Range: Ideally upto 3.0000]')
            Age = st.text_input('Age - [Range: Ideally between 1 - 105]')

            x_test = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,random_state=0)
            a = model.fit(x_train,y_train)
            y_pred = round(model.predict_proba(x_test)[0][1],2) 
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'] 
            data = tree.export_graphviz(a, feature_names=features, out_file=None)
            if st.button("Predict"):
                st.markdown("The probability of having diabets is :")
                st.success(y_pred)
                st.write('')
                st.subheader('**Decision Tree**')
                st.graphviz_chart(data)

        
        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("**Decision Tree Results**")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            st.write("**Accuracy : **",round(accuracy,2))
            y_pred = model.predict(x_test)
            plot_metrics(metrics,y_test,y_pred)
            a = model.fit(x_train,y_train) 
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'] 
            data = tree.export_graphviz(a, feature_names=features, out_file=None)
            st.write('')
            st.subheader('**Decision Tree**')
            st.graphviz_chart(data)



    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameter")
        n_estimators = st.sidebar.number_input("The number of trees in the forest",100,300,step = 50,key = 'n_estomators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step =1, key ='max_depth')
        criterion = st.sidebar.radio("Criterion",('gini', 'entropy'),key = "criterion")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",('True','False'),key = 'bootstrap')
        #rf_params = {'criterion' : ['gini', 'entropy'],'n_estimators': list(range(100, 300, 50)),'max_depth': list(range(1, 20, 1)),'bootstrap' : ['True','False']}
        #rf_model = GridSearchCV(RandomForestClassifier(random_state=0), rf_params, cv=5) 
        #rf_model.fit(x_train, y_train)
        #rf_predict = rf_model.predict(x_test)
        #st.write(rf_model.best_params_)
        st.subheader("**Parameters optimization ran on**")
        st.write("**criterion : **['gini', 'entropy']")
        st.write("**n_estimators: **[100-300]")
        st.write("**max_depth: **[1-20]")
        st.write("**bootstrap : **['True','False']")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        st.subheader("Best Params for Random Forest")
        st.write("**criterion :** gini")
        st.write("**max_depth :** 11")
        st.write("**n_estimators :** 150")
        st.write("**bootstrap :** True")

        if st.checkbox("Custom Input"):
            Pregnancies   = st.text_input('Number of pregnancies - [Range: Ideally between 1 - 12]')
            Glucose   = st.text_input('Glucose Level - [Range(mg/dL): Ideally upto 140 ]')
            BloodPressure = st.text_input('BloodPressure Level - [Range(mm Hg): 90/60 - 120/80]')
            SkinThickness  = st.text_input('Skin thickness - [Range(mm) - Ideally upto 90]')
            Insulin = st.text_input('Insulin Level - [Range(mu U/ml): Ideally between 3-25 ]')
            BMI = st.text_input('BMI - [Range : Ideally between 15-35 ]')
            DiabetesPedigreeFunction  = st.text_input('DiabetesPedigreeFunction - [Range: Ideally upto 3.0000]')
            Age = st.text_input('Age - [Range: Ideally between 1 - 105]')

            x_test = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,bootstrap=bootstrap,criterion = criterion,n_jobs=-1,random_state=0)
            model.fit(x_train,y_train)
            y_pred = round(model.predict_proba(x_test)[0][1],2)
            if st.button("Predict"):
                st.markdown("The probability of having diabets is :")
                st.success(y_pred)

        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,bootstrap=bootstrap,criterion = criterion,n_jobs=-1,random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))

    if classifier == "XG Boost":
        st.sidebar.subheader("Model Hyperparameter")
        n_estimators = st.sidebar.radio("n_estimators",(5, 10, 15, 20, 25, 50, 100),key = "n_estimators")
        learning_rate = st.sidebar.radio("learning_rate",(0.01, 0.05, 0.1),key = "learning_rate")
        max_depth = st.sidebar.radio("Max Depth",(3, 5, 7, 9),key = "max_depth")
        #xgb_params = {'max_depth': [3, 5, 7, 9],'n_estimators': [5, 10, 15, 20, 25, 50, 100],'learning_rate': [0.01, 0.05, 0.1]}
        #xgb_model = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss',random_state=0), xgb_params, cv=5) 
        #xgb_model.fit(x_train, y_train)
        #st.write(xgb_model.best_params_)


        st.subheader("**Parameters optimization ran on**")
        st.write("**learning_rate : **[0.01, 0.05, 0.1]")
        st.write("**n_estimators: **[5, 10, 15, 20, 25, 50, 100]")
        st.write("**max_depth: **[3, 5, 7, 9, 11, 13]")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        st.subheader("Best Params for XG Boost")
        st.write("**learning_rate :** 0.05")
        st.write("**n_estimators :** 150")
        st.write("**max_depth :** 7")

        
        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("XG Boost Results")
            model = xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            st.write("**Accuracy : **",round(accuracy,2))
            y_pred = model.predict(x_test)
            plot_metrics(metrics,y_test,y_pred)

    if classifier == "K-Nearest Neighbour":
        st.sidebar.subheader("Model Hyperparameter")
        n_neighbors = st.sidebar.radio("n_neighbors",(3,5,7,11,15,20),key = "n_neighbors")
        weights = st.sidebar.radio("weights",('uniform', 'distance'),key = "weights")
        algorithm = st.sidebar.radio("algorithm",('auto', 'ball_tree', 'kd_tree', 'brute'),key = "algorithm")
        metric = st.sidebar.radio("metric",('euclidean', 'manhattan', 'chebyshev', 'minkowski'),key = "metric")

        st.subheader("**Parameters optimization ran on**")
        st.write("**n_neighbors : **[3,5,7,11,15,20]")
        st.write("**weights: **['uniform', 'distance']")
        st.write("**algorithm: **['auto', 'ball_tree', 'kd_tree', 'brute']")
        st.write("**metric: **['euclidean', 'manhattan', 'chebyshev', 'minkowski']")
        st.write("Iterationsn ran through GridsearchCv optimization with Cross-validation of 5")
        
        knn_params = {'n_neighbors': list(range(3, 20, 2)),'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
        knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
        knn_model.fit(x_train, y_train)
        st.write('**Best parameters after GridSearchcv**')
        st.write(knn_model.best_params_)


        if st.checkbox("Custom Input"):
            Pregnancies   = st.text_input('Number of pregnancies - [Range: Ideally between 1 - 12]')
            Glucose   = st.text_input('Glucose Level - [Range(mg/dL): Ideally upto 140 ]')
            BloodPressure = st.text_input('BloodPressure Level - [Range(mm Hg): 90/60 - 120/80]')
            SkinThickness  = st.text_input('Skin thickness - [Range(mm) - Ideally upto 90]')
            Insulin = st.text_input('Insulin Level - [Range(mu U/ml): Ideally between 3-25 ]')
            BMI = st.text_input('BMI - [Range : Ideally between 15-35 ]')
            DiabetesPedigreeFunction  = st.text_input('DiabetesPedigreeFunction - [Range: Ideally upto 3.0000]')
            Age = st.text_input('Age - [Range: Ideally between 1 - 105]')

            x_test = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

            model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)
            model.fit(x_train,y_train)
            y_pred = round(model.predict_proba(x_test)[0][1],2)
            if st.button("Predict"):
                st.markdown("The probability of having diabets is :")
                st.success(y_pred)


    
        metrics = st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","Classification Report"))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("K-Nearest Neighbour Results")
            model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,metric=metric)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            st.write("**Accuracy : **",round(accuracy,2))
            y_pred = model.predict(x_test)
            plot_metrics(metrics,y_test,y_pred)
    




    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Prima Diabetes Data set (Classification)")
        st.write(df.head(8))



if __name__ == '__main__':
    main()
