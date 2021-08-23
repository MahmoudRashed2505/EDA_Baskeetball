#################################
# Import Libraries
#################################

from re import T
from pandas.core import base
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2020))))

# Web Scrabing of NBA player stats
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url,header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deleting Reapeating headers in content 
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'],axis=1)
    playerstats = playerstats.drop(['FG%'],axis=1)
    playerstats = playerstats.drop(['3P%'],axis=1)
    playerstats = playerstats.drop(['2P%'],axis=1)
    playerstats = playerstats.drop(['eFG%'],axis=1)
    playerstats = playerstats.drop(['FT%'],axis=1)
    return playerstats

playerstats = load_data(selected_year)

# Sidebar - Team selection
sorted_unique_teams = sorted(playerstats.Tm.unique())
selected_teams = st.sidebar.multiselect('Team',sorted_unique_teams,sorted_unique_teams)

# Sidebar - Postion Selection
unique_pos = ['C','PF','SF','PG','SG']
selected_postions = st.sidebar.multiselect('Postions',unique_pos,unique_pos)

#Filtering Data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_teams)) & (playerstats.Pos.isin(selected_postions))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NBA Player Stats data


def fileDownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
    href = '<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(fileDownload(df_selected_team),unsafe_allow_html=True)

# HeatMap

if st.button("Intercorrelation Heatmap"):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        f,ax =  plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr,mask=mask,vmax=1,square=True)
    st.pyplot(f)