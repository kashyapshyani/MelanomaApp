import os
from os import listdir
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set(style="whitegrid")

#plotly
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#pydicom
import pydicom

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

temp = train_df.groupby(['benign_malignant','sex']).count()['image_name'].to_frame()


st.sidebar.title("Melanoma")
nav = st.sidebar.radio("Go to",["Prediction","EDA","About"])

if nav=="Home":
    st.write("Prediction")

if nav=="EDA":
    st.sidebar.title("EDA")

    if st.sidebar.checkbox("Distribution of The Target columns in training set"):
        train_df_new = train_df['target'].value_counts(normalize=True).reset_index()
        fig = px.bar(train_df_new,x='target',y='target')
        st.write("Distribution of The Target columns in training set")
        fig.update_layout(
            # title="Distribution of The Target columns in training set",
            xaxis_title="Target",
            yaxis_title="Percentage",
            width=500,
            height=550,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Distribution of The Gender columns in training set"):
        train_df_new = train_df['sex'].value_counts(normalize=True).reset_index()
        fig = px.bar(x=train_df_new['sex'].index ,y=train_df_new['sex'].values)
        st.write("Distribution of The Sex columns in training set")
        fig.update_layout(
            # title="Distribution of The Sex columns in training set",
            yaxis_title="Percentage",
            width=500,
            height=550,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Missing Values"):
        st.write("Missing Values")
        dd = train_df[["sex","age_approx","anatom_site_general_challenge"]].isnull().sum()
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        fig = px.bar(dd,x='gh',y='values')
        # fig = px.bar(dd, x='gh' ,y='values')
        fig.update_layout(
            # title="Missing Values",
            xaxis_title="Count",
            yaxis_title="Columns",
            width=800,
            height=600,
        )
        # st.write(fig)
        st.plotly_chart(fig)
    # nav2 = st.sidebar.radio("EDA Options",["Missing Values","Explore Target Column","Gender Wise Distribution","Gender vs Target Dsitribution"])



    if st.sidebar.checkbox("Gender for Anatomy"):
        st.write("Gender for Anatomy")
        anatomy = train_df.copy()
        anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
        ztemp=anatomy.groupby(['sex','flag'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Missing', x=ztemp[ztemp['sex']=='male']['flag'], y=ztemp[ztemp['sex']=='male']['target']),
            go.Bar(name='Not Missing',  x=ztemp[ztemp['sex']=='female']['flag'], y=ztemp[ztemp['sex']=='female']['target'])
        ])
        # Change the bar mode
        fig.update_layout(barmode='group',
                          # title="Gender for Anatomy",
                          xaxis_title="Missing vs Not Missing",
                          yaxis_title="Count",
                          width=750,
                          height=600,)

        st.plotly_chart(fig)

    if st.sidebar.checkbox("Gender vs Target Dsitribution"):
        st.write("Gender Vs Target Distribution")
        z=train_df.groupby(['benign_malignant','sex'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Male', x=z[z['sex']=='male']['benign_malignant'], y=z[z['sex']=='male']['target']),
            go.Bar(name='Female', x=z[z['sex']=='female']['benign_malignant'], y=z[z['sex']=='female']['target'])
        ])
        # Change the bar mode

        fig.update_layout(barmode='group',
                          # title="Gender Vs Target Distribution",
                          xaxis_title="Benign:0 Vs Malignanat:1",
                          yaxis_title="Count",
                          width=750,
                          height=600,)

        st.plotly_chart(fig)

    if st.sidebar.checkbox(" Target columns Distribution in training set"):
        dd = train_df['anatom_site_general_challenge'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.bar(dd, x='values' ,y='gh', orientation="h")
        fig.update_layout(
            title="Distribution of The Target columns in training set",
            yaxis_title="Percentage",
            width=900,
            height=500,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Location of Imaged Site w.r.t. Gender"):
        st.write("Location of Imaged Site w.r.t. Gender")
        z1 = train_df.groupby(['sex','anatom_site_general_challenge'])['target'].count().to_frame().reset_index()

        fig = go.Figure(data=[
            go.Bar(name='Male', x=z1[z1['sex']=='male']['anatom_site_general_challenge'], y=z1[z1['sex']=='male']['target']),
            go.Bar(name='Female', x=z1[z1['sex']=='female']['anatom_site_general_challenge'], y=z1[z1['sex']=='female']['target'])
        ])
        # Change the bar mode
        fig.update_layout(barmode='group',
                          # title="Location of Imaged Site w.r.t. Gender",
                          xaxis_title="Location of Imaged site",
                          yaxis_title="Count of melanoma cases",
                          width=1000,
                          height=600)

        st.plotly_chart(fig)

    if st.sidebar.checkbox("Age Distribution"):
        dd = train_df['age_approx'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.histogram(dd, x='gh' ,y='values',nbins=30)
        st.write("Age Distribution of Patients")
        fig.update_layout(
            # title="Age Distribution of Patients",
            xaxis_title="Age Distribution",
            yaxis_title="Count",
            width=1000,
            height=600,
        )

        st.plotly_chart(fig)

    if st.sidebar.checkbox("Diagnosis of Target Columns in Training DataSet"):
        st.write("Diagnosis of Target Columns in Training DataSet")
        dd = train_df['diagnosis'].value_counts(normalize=True)
        dd = pd.DataFrame({'gh':dd.index , 'values':dd.values})
        # fig = px.bar(x=train_df_new['target'].index ,y=train_df_new['target'].values)
        fig = px.bar(dd, x='values' ,y='gh', orientation="h")
        fig.update_layout(
            # title="Distribution of The Target columns in training set",
            xaxis_title="Percentage",
            yaxis_title="Diagnosis",
            width=900,
            height=500,
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Distibution plot"):
        anatomy = train_df.copy()
        anatomy['flag'] = np.where(train_df['anatom_site_general_challenge'].isna()==True, 'missing', 'not_missing')
        colors_nude = ['#e0798c','#65365a','#da8886','#cfc4c4','#dfd7ca']
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
        sns.distplot(anatomy[anatomy['flag'] == 'missing']['age_approx'],
                     hist=False, rug=True, label='Missing',
                     color=colors_nude[2], kde_kws=dict(linewidth=4))
        sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age_approx'],
                     hist=False, rug=True, label='Not Missing',
                     color=colors_nude[3], kde_kws=dict(linewidth=4))
        st.pyplot()

    if st.sidebar.checkbox("Explore Dataset Columns"):
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        st.write("Train size = ",train_df.shape)
        st.write("Test Size = ",test_df.shape)
        temp = train_df.groupby(['benign_malignant','sex']).count()['image_name'].to_frame()
        st.title('Train Set ')
        temp=train_df.describe()
        st.write(temp)
        st.title('Test Set ')
        temp = test_df.describe()
        st.write(temp)
        temp = train_df["image_name"].count()
        st.write('Total Images In Training DataSet: ',temp)
        temp = train_df.groupby(["benign_malignant"]).get_group("benign").count()["sex"]
        st.write('Number of Benign Sample in Training DataSet : ', temp)
        temp = train_df.groupby(["benign_malignant"]).get_group("malignant").count()["sex"]
        st.write(' Number of Malignant Sample in Training DataSet : ',temp)
        temp = test_df["image_name"].count()
        st.write('Total Images In Training DataSet: ',temp)
        temp = train_df['patient_id'].count()
        temp2 = train_df['patient_id'].nunique()
        st.write("The total Patient IDs are ",temp," from those unique ids are ",temp2)
        columns = list(train_df.columns)
        st.title("Column Names : ")
        st.write(columns)

if nav=="About":
    st.write("About")

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
