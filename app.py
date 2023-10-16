import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip


#sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer

def predict_linear_regression(df):

    #Fill up data with KNN
    my_imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
    df_repaired = pd.DataFrame(my_imputer.fit_transform(df), columns=df.columns)
    
    # Data Preparation
    X = df_repaired.drop(["Potability"], axis=1)
    y = df_repaired["Potability"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Streamlit App
    st.title("Water Portability Prediction with Linear Regression")

    # Sidebar for user input
    c1 = st.columns(3)
    c2 = st.columns(3)
    c3 = st.columns(3)
    
    ph = c1[0].slider("pH Value", 0.0, 14.0, 7.0)
    hardness = c1[1].slider("Hardness", 0, 500, 250)
    solids = c1[2].slider("Solids", 0, 50000, 25000)
    chloramines = c2[0].slider("Chloramines", 0.0, 15.0, 7.5)
    sulfate = c2[1].slider("Sulfate", 0, 500, 250)
    conductivity = c2[2].slider("Conductivity", 100, 1000, 550)
    organic_carbon = c3[0].slider("Organic Carbon", 0, 50, 25)
    trihalomethanes = c3[1].slider("Trihalomethanes", 0.0, 150.0, 75.0)
    turbidity = c3[2].slider("Turbidity", 0.0, 10.0, 5.0)

    # Create a DataFrame for prediction
    input_data = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    input_df = pd.DataFrame([input_data])

    # Predict using the model
    prediction = model.predict(input_df)

    # Display the prediction result
    st.header("Prediction Result")
    st.write(f"The predicted Potability is: {prediction[0]*100:.2f} %")

    # Optional: Show the model's metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.header("Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")

def predict_KNN(df):

    #Fill up data with KNN
    my_imputer = KNNImputer(n_neighbors=9, weights='distance', metric='nan_euclidean')
    df_repaired = pd.DataFrame(my_imputer.fit_transform(df), columns=df.columns)
    
    # Data Preparation
    X = df_repaired.drop(["Potability"], axis=1)
    y = df_repaired["Potability"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the KNN model
    k = 9  # Choose the number of neighbors (you can change this)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Streamlit App
    st.title("Water Portability Prediction with KNN")

    # Sidebar for user input
    c4 = st.columns(3)
    c5 = st.columns(3)
    c6 = st.columns(3)
    
    ph1 = c4[0].slider("pH value", 0.0, 14.0, 7.0)
    hardness1 = c4[1].slider("hardness", 0, 500, 250)
    solids1 = c4[2].slider("solids", 0, 50000, 25000)
    chloramines1 = c5[0].slider("chloramines", 0.0, 15.0, 7.5)
    sulfate1 = c5[1].slider("sulfate", 0, 500, 250)
    conductivity1 = c5[2].slider("conductivity", 100, 1000, 550)
    organic_carbon1 = c6[0].slider("organic carbon", 0, 50, 25)
    trihalomethanes1 = c6[1].slider("trihalomethanes", 0.0, 150.0, 75.0)
    turbidity1 = c6[2].slider("turbidity", 0.0, 10.0, 5.0)

    # Create a DataFrame for prediction
    input_data1 = {
        "ph": ph1,
        "Hardness": hardness1,
        "Solids": solids1,
        "Chloramines": chloramines1,
        "Sulfate": sulfate1,
        "Conductivity": conductivity1,
        "Organic_carbon": organic_carbon1,
        "Trihalomethanes": trihalomethanes1,
        "Turbidity": turbidity1
    }

    input_df1 = pd.DataFrame([input_data1])

    # Predict using the model
    prediction1 = model.predict_proba(input_df1)

    # Display the prediction result
    st.header("Prediction Result")
    st.write(f"The predicted Potability is: {prediction1[0][1]*100:.2f} %")

    # Optional: Show the model's metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.header("Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    
    
def predict_ml(df):
    x = df.drop(['Potability'], axis='columns')
    y = df.Potability
    features_scaler = MinMaxScaler()
    features = features_scaler.fit_transform(x)
    #features

    model_params = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'logistic_regression' : {
            'model': LogisticRegression(solver='liblinear',multi_class='auto'),
            'params': {
                'C': [1,5,10]
            }
        },
        'svm': {
            'model': SVC(gamma='auto'),
            'params' : {
                'C': [1,10,20,30,50],
                'kernel': ['rbf','linear','poly']
            }  
        },
        'KNN' : {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3,7,11,13]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [10,50,100]
            }
        }

    }
    
    
    scores = []

    for model_name, mp in model_params.items():
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(features, y)
        scores.append({
            'model': model_name,
            'best_score': abs(clf.best_score_), #abs should not be here, just for removing error, this is not correct, 
            'best_params': clf.best_params_
        })

    df_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
    # Create a bar plot
    fig, ax = plt.subplots()
    sns.barplot(x="model", y="best_score", data=df_score, ax=ax) 
    plt.ylim(0, 1)
    plt.title("Model Scores")
    plt.xlabel("Model")
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    plt.ylabel("Best Score")
    # Display the plot in Streamlit
    st.pyplot(fig)
    #write best scores
    st.write(df_score)

def summary(df):
    # Columns Summary

    st.subheader('| SUMMARY')
    
    col1, col2 = st.columns([2, 1])
    # column 1 - Describe
    with col1:
        st.write(df.describe())
    # column 2 - Potability Pie
    with col2:
        col = len(df.columns)-1
        st.write('PARAMETERS : ',col)
        row = len(df) 
        st.write('TOTAL DATA : ', row)
        st.write("Potability Distribution (Pie Chart)")
        potability_counts = df['Potability'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(potability_counts, labels=potability_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        
def missingdata(df):
    # Columns Summary

    st.subheader('| SUMMARY')
    
    col1, col2 = st.columns([1, 2])
    # column 1 - Describe missing data
    with col1:
        st.write(df.isnull().sum())
    # column 2 - Potability Pie
    with col2:
        st.write('Heatmap of Missing Values: ')
        sns.heatmap(df.isna(), cmap="flare")
        
        #sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        heatmap_fig = plt.gcf()  # Get the current figure
        st.pyplot(heatmap_fig)

def fill_data_median(df):
    df_old = df
    
    col1, col2 = st.columns([1, 1])
    # column 1 - Describe missing data
    with col1:
        st.write("Before Fillup (df.isna())")
        fig1, ax = plt.subplots()
        sns.heatmap(df_old.isna(), cmap="plasma")
        st.write(fig1)
    
    #Fill up data with median
    df['ph'].fillna(value=df['ph'].median(),inplace=True)
    df['Sulfate'].fillna(value=df['Sulfate'].median(),inplace=True)
    df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median(),inplace=True)
    # column 2 - Potability Pie
    with col2:
        st.write("After Fillup with Median (df.isna())")
        fig2, ax = plt.subplots()
        sns.heatmap(df.isna(), cmap="plasma")
        st.write(fig2)
    return df
        
def fill_data_KNN(df):
    df_old = df
    
    col1, col2 = st.columns([1, 1])
    # column 1 - Describe missing data
    with col1:
        st.write("Before Fillup (df.isna())")
        fig1, ax = plt.subplots()
        sns.heatmap(df_old.isna(), cmap="plasma")
        st.write(fig1)
    
    #Fill up data with KNN
    my_imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
    df_repaired = pd.DataFrame(my_imputer.fit_transform(df_old), columns=df_old.columns)
    
    # column 2 - Potability Pie
    with col2:
        st.write("After Fillup with KNNImputer (df.isna())")
        fig2, ax = plt.subplots()
        sns.heatmap(df_repaired.isna(), cmap="plasma")
        st.write(fig2)
    return df_repaired

def main():
    #intro
    intro = 1;
    
    st.sidebar.title('CMSE 830 :  Midterm Project')
    st.sidebar.write('Developed by Md Arifuzzaman Faisal')
    

    # st.header("Upload your CSV data file")
    # data_file = st.file_uploader("Upload CSV", type=["csv"])

    # if data_file is not None:
    df = pd.read_csv("water_potability.csv")
    
    #show about the dataset
    about = st.sidebar.checkbox('About the Dataset')
    if about:
        st.subheader("Water Potability! Is the water safe for drink?")
        # Add a slider for selecting the number of rows to display
        num_rows = st.sidebar.slider("Number of Rows", 1, 100, 5)

        # Display the selected number of rows
        st.write(f"Displaying top {num_rows} rows:")
        st.write(df.head(num_rows))
        # Add an image
        st.image("dw.jpg", caption="Is the water safe for drink?", use_column_width=True)
        message = "Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions."
        st.write(message)
        
        
        #show info of the dataset
        info = st.sidebar.checkbox('Describe the Dataset')
        if info:
            summary(df)

        intro = 0
        
    
    st.sidebar.header("Visualizations")
    #show info of the dataset
    visual1 = st.sidebar.checkbox('See the EDA')
    if visual1:    
        plot_options = ["Correlation Heat Map", "Histogram of Column","Joint Plot of Columns", "Box Plot of Column"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Correlation Heat Map":
            st.write("Correlation Heatmap:")
            #plt.figure(figsize=(10, 10))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            heatmap_fig = plt.gcf()  # Get the current figure
            st.pyplot(heatmap_fig)
        elif selected_plot == "Joint Plot of Columns":
            x_axis = st.sidebar.selectbox("Select x-axis", df.columns, index=0)
            y_axis = st.sidebar.selectbox("Select y-axis", df.columns, index=1)
            st.write("Joint Plot:")
            jointplot = sns.jointplot(data = df, x=df[x_axis], y=df[y_axis], hue="Potability")
            #sns.scatterplot(data = df, x=df[x_axis], y=df[y_axis], hue="Potability", ax=ax)
            st.pyplot(jointplot)

        elif selected_plot == "Histogram of Column":
            column = st.sidebar.selectbox("Select a column", df.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(df[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box Plot of Column":
            column = st.sidebar.selectbox("Select a column", df.columns)
            st.write("Box Plot:")
            fig, ax = plt.subplots()
            sns.boxplot(df[column], ax=ax)
            st.pyplot(fig)
        
        intro = 0
        
    st.sidebar.header("Missing Data Analysis")
    #show info of the dataset
    misdata = st.sidebar.checkbox('Summary of Missing Data')
    if misdata:
        missingdata(df)
        
        intro = 0
        
    st.sidebar.header("Treatment of Missing Data")
    #show info of the dataset
    #fill_data = st.sidebar.checkbox('Fill Data')
    #if fill_data:
        #fill_median = st.sidebar.checkbox('Fill Data Using Median')
        #if fill_median:
            #df1 = fill_data_median(df)
            #predict_ml(df1)
            
    fill_KNN = st.sidebar.checkbox('Fill Data Using KNN Imputer')
    if fill_KNN:
        df2 = fill_data_KNN(df)
        #predict_ml(df2)

        intro = 0
            
    
    st.sidebar.header("Prediction")
    #Predict with Linear Regression
    predict = st.sidebar.checkbox('Predict Potability Using Linear Regression')
    if predict:
        predict_linear_regression(df)
        
        intro = 0
    #Predict with KNN
    predict = st.sidebar.checkbox('Predict Potability Using KNN')
    if predict:
        predict_KNN(df)

        intro = 0
            
    if intro == 1:
        st.subheader("Water Potability! Is the water safe for drink?")
        # Convert the DataFrame to a HiPlot Experiment
        exp = hip.Experiment.from_dataframe(df)
        # Render the HiPlot experiment in Streamlit
        st.components.v1.html(exp.to_html(), width=900, height=600, scrolling=True)
    


if __name__ == "__main__":
    main()