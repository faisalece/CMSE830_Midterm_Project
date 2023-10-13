import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip

def summary(df):
    # Columns Summary

    st.subheader('| QUICK SUMMARY')
    
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

def main():
    #intro
    intro = 1;
    
    st.sidebar.title('CMSE 830 :  Midterm Project')
    st.sidebar.write('Developed by Md Arifuzzaman Faisal')
    

    # st.header("Upload your CSV data file")
    # data_file = st.file_uploader("Upload CSV", type=["csv"])

    # if data_file is not None:
    df = pd.read_csv("water_potability.csv")
    st.subheader("Water Potability! Is the water safe for drink?")
    
    #show about the dataset
    about = st.sidebar.checkbox('About the Dataset')
    if about:
        message = "Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions."
        st.write(message)
        # Add an image
        st.image("dw.jpg", caption="Is the water safe for drink?", use_column_width=True)
        # Add a slider for selecting the number of rows to display
        num_rows = st.sidebar.slider("Number of Rows", 1, 15, 5)

        # Display the selected number of rows
        st.write(f"Displaying top {num_rows} rows:")
        st.write(df.head(num_rows))
        
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
        st.write(df.isnull().sum())
        
        intro = 0
            
    if intro == 1:
        # Convert the DataFrame to a HiPlot Experiment
        exp = hip.Experiment.from_dataframe(df)

        # Render the HiPlot experiment in Streamlit
        st.components.v1.html(exp.to_html(), width=900, height=600, scrolling=True)
    


if __name__ == "__main__":
    main()
