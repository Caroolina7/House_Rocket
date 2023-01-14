import pandas as pd
import streamlit as st
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import geopandas
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def obter_dados(caminho):
    data = pd.read_csv(caminho)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data


def overview_data(data):

    f_zipcode = st.sidebar.multiselect('Insira zipcode',
                                       data['zipcode'].unique())
    st.write(f_zipcode)

    f_colunas = st.sidebar.multiselect('Insira a coluna', data.columns)

    st.write(f_colunas)

    st.title('Dados visão global')
    
    if (f_zipcode != []) & (f_colunas != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_colunas]
    
    elif (f_zipcode == []) & (f_colunas != []):
        data = data.loc[:, f_colunas]
    
    elif (f_zipcode != []) & (f_colunas == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    else:
        data = data.copy()
    
    st.dataframe(data)
    
    c1, c2 = st.columns((1, 1))
    
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING',
              'PRICE/M2']

    c1.header('Métricas por cógigo postais')

    c1.dataframe(df, height=600)

    num_atributos = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_atributos.apply(np.mean))
    mediana = pd.DataFrame(num_atributos.apply(np.median))
    desvio = pd.DataFrame(num_atributos.apply(np.std))
    min_ = pd.DataFrame(num_atributos.apply(np.min))
    max_ = pd.DataFrame(num_atributos.apply(np.max))

    df1 = pd.concat([max_, min_, media, mediana, desvio],
                axis=1).reset_index()

    df1.columns = ['Atributes', 'Max', 'Min', 'Media', 'Mediana', 'Desvio']

    c2.header('Estatística Descritiva')

    c2.dataframe(df1, height=600)

    return None

def portifolio_density(data, geofile ):
    
    st.title('Visão global das regiões')
    
    c1, c2 = st.columns((1, 1))
    
    c1.header('Densidade por região')
    
    df = data.sample(10)
    
    densidade_mapa = folium.Map(location=[data['lat'].mean(),
                                      data['long'].mean()], default_zoom_star=15)
    marker_cluster = MarkerCluster().add_to(densidade_mapa) 
    for name, row in df.iterrows(): 
        folium.Marker([row['lat'], row['long']], 
                  popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( 
                      row['price'], 
                      row['date'], 
                      row['sqft_living'], 
                      row['bedrooms'], 
                      row['bathrooms'], 
                      row['yr_built'])).add_to(marker_cluster) 
 
    with c1:
        folium_static(densidade_mapa)
    
    c2.header('Densidade de preço')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']
    
    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]
    
    region_price_map = folium.Map(location=[data['lat'].mean(),
                                        data['long'].mean()], default_zoom_star=15)
                                        
    region_price_map.choropleth(data=df,
                            geo_data=geofile,
                            columns=['ZIP', 'PRICE'],
                            key_on='feature.properties.ZIP',
                            fill_color='YlOrRd',
                            fill_opacity=0.7,
                            line_opacity=0.2,
                            legend_name='AVG PRICE')
    with c2:
        folium_static(region_price_map)
    
    return None
   
def comercial_distribution(data):
    
    st.sidebar.title('Opção de preços')
    st.title('Atributos preços')
    
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    
    st.sidebar.subheader('Selecione o ano máximo de construção')
    
    f_year_built = st.sidebar.slider(
    'Ano', min_year_built, max_year_built, min_year_built)
    
    st.header('Média de preço por ano')
    
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Média de preço por dia')
    st.sidebar.subheader('Selecione o dia máximo')

    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Dia', min_date, max_date, min_date)

    data['date'] = pd.to_datetime(data['date'])

    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Distribuição do preço')
    st.sidebar.subheader('Selecione o preço máx')

    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Preço', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

def atributes_distribution (data):

    st.sidebar.title('Opção de Atributos')
    st.title('Atributos das casas')
    
    f_bedrooms = st.sidebar.selectbox(
    'Número máximo de quartos', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox(
    'Número máximo de banheiros', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.columns(2)

    c1.header('Casas por número de quartos')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    c2.header('Casas por número de banheiros')
    df = data[data['bathrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    f_floors = st.sidebar.selectbox(
    'Número de andares', sorted(set(data['floors'].unique())))

    f_waterview = st.sidebar.checkbox('Casa com vista para o mar')

    c1, c2 = st.columns(2)

    c1.header('Casas por andares')
    df = data[data['floors'] < f_floors]

    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    if f_waterview:
        df = data[data['waterfront'] == 1]

    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == "__main__":

    caminho = 'csv/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = obter_dados(caminho)
    geofile = get_geofile(url)

    data = set_feature(data)

    overview_data(data)

    portifolio_density(data, geofile)

    comercial_distribution(data)

    atributes_distribution(data)

