#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import seaborn as sns
sns.set_theme()

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from bokeh.io import output_notebook ,push_notebook
from bokeh.models import ColorBar, ColorMapper ,ColumnDataSource , HoverTool , GeoJSONDataSource
from bokeh.palettes import Viridis256
from bokeh.plotting import output_notebook , figure, show
from bokeh.transform import factor_cmap ,linear_cmap,  LinearColorMapper
from bokeh.models import WMTSTileSource

import folium
import json
from folium import GeoJson, Choropleth, Choropleth, Circle, Marker
from streamlit_folium import folium_static
from branca.colormap import linear
from folium.plugins import HeatMap

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from shapely.geometry import Point

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder, StandardScaler 
from sklearn.tree import plot_tree, DecisionTreeRegressor

import xgboost as xgb
from xgboost import XGBRegressor
output_notebook()



# population
df = pd.read_csv('C:/Users/Jean-Jacques/Desktop/Datascientest/french_industry/dossiers_de_travail/Documents csv/population.csv', low_memory=False)    


# name_geographic_information
df2 = pd.read_csv('C:/Users/Jean-Jacques/Desktop/Datascientest/french_industry/dossiers_de_travail/Documents csv/name_geographic_information.csv')    


# base_etablissement_par_tranche_effectif
df3 = pd.read_csv('C:/Users/Jean-Jacques/Desktop/Datascientest/french_industry/dossiers_de_travail/Documents csv/base_etablissement_par_tranche_effectif.csv')    


# net_salary_per_town_categories
df4 = pd.read_csv('C:/Users/Jean-Jacques/Desktop/Datascientest/french_industry/dossiers_de_travail/Documents csv/net_salary_per_town_categories.csv')    


# ### Préparation des données
# 

# df : enlever NIVGEO, qui n'a qu'une seule valeur
df = df.drop(columns=['NIVGEO'])


# #### Réorganiser les régions

# Enlever les DOM TOM dans les fichiers df2 et df3
df2 = df2[~df2['code_région'].isin([1,2,3,4,5,6])]
df3 = df3[~df3['REG'].isin([1,2,3,4,5,6])]


# actualiser les numéros des régions dans df2 et dans df3
num_a_remplacer = {
    26: 27, 43: 27, 23: 28, 25: 28, 22: 32, 31: 32, 21: 44, 41: 44, 42: 44, 54: 75, 72: 75, 74: 75,
    73: 76, 91: 76, 82: 84, 83: 84
}

df2['code_région'] = df2['code_région'].replace(num_a_remplacer)
df3['REG'] = df3['REG'].replace(num_a_remplacer)


# dans df2, remplacer 'code_région' par REG
df2 = df2.rename(columns={'code_région': 'REG'})


# Modifier le nom des régions dans df2, en fonction du numéro de région
nom_regions = {
    11: 'Ile-de-France',
    24: 'Centre-Val de Loire',
    27: 'Bourgogne-Franche-Comté',
    28: 'Normandie',
    32: 'Hauts-de-France',
    44: 'Grand Est',
    52: 'Pays de la Loire',
    53: 'Bretagne',
    75: 'Nouvelle-Aquitaine',
    76: 'Occitanie',
    84: 'Auvergne-Rhône-Alpes',
    93: "PACA",
    94: 'Corse'
}

df2['nom_REG'] = df2['REG'].map(nom_regions)


# Ajouter une colonne REG à df et à df4 en fonction de df3
codgeo_to_reg = df3.set_index('CODGEO')['REG'].to_dict()
df['REG'] = df['CODGEO'].map(codgeo_to_reg)
df4['REG'] = df4['CODGEO'].map(codgeo_to_reg)

# Ajouter une colonne DEP à df
codgeo_to_dep = df3.set_index('CODGEO')['DEP'].to_dict()
df['DEP'] = df['CODGEO'].map(codgeo_to_dep)


# remplacer manuellement les Nan par les bonnes valeurs dans REG de df 
df.loc[df['LIBGEO'] == "L'Oudon", 'REG'] = 52
df.loc[df['LIBGEO'] == "Culey", 'REG'] = 44


# df et df4 : enlever les lignes où la valeur est manquante dans REG ; ces lignes correspondent aux DOM TOM
df = df.dropna(subset=['REG'])
df4 = df4.dropna(subset=['REG'])


# ajouter la colonne nom_REG à df4
df4['nom_REG'] = df4['REG'].map(nom_regions)


# changer le float en int pour 'REG' de df et df4
df['REG'] = df['REG'].astype(int)
df4['REG'] = df4['REG'].astype(int)


# mettre la colonne 'REG' de df4 et  au même format que celle de df3 en utilisant merge
df4 = df4.merge(df3[['CODGEO', 'DEP']], on='CODGEO', how='left')


# #### Réorganiser les départements

#  df3 : changer la valeur de 2A et 2B dans DEP 
df['DEP'] = df['DEP'].replace({'2A': '20', '2B': '96'})
df3['DEP'] = df3['DEP'].replace({'2A': '20', '2B': '96'})
df4['DEP'] = df4['DEP'].replace({'2A': '20', '2B': '96'})


# Remplace les valeurs de la CORSE dans CODGEO de df, df3 et df4
df['CODGEO'] = df['CODGEO'].str.replace('2A', '20').str.replace('2B', '20')
df3['CODGEO'] = df3['CODGEO'].str.replace('2A', '20').str.replace('2B', '20')
df4['CODGEO'] = df4['CODGEO'].str.replace('2A', '20').str.replace('2B', '20')


# df2 renommer nom_commune en LIBGEO , 'code_insee' en  'CODGEO' et 'numéro_département' : 'DEP' ...
df2 =df2.rename(columns={ 'nom_commune' : 'LIBGEO', 'code_insee' : 'CODGEO', 'numéro_département' : 'DEP'})

# Mettre CODGEO de df2 en str
df2.CODGEO = df2.CODGEO.astype('str')


# ... cela permet maintenant de remplacer les valeurs de la Corse
df2['DEP'] = df2['DEP'].replace({'2A': '20', '2B': '96'})


# remplacer manuellement les Nan par les bonnes valeurs dans REG de df 
df.loc[df['LIBGEO'] == "L'Oudon", 'DEP'] = 44
df.loc[df['LIBGEO'] == "Culey", 'DEP'] = 55


# ##### Renommer et enlever des colonnes ou des valeurs 

# enlever les colonnes qui provoquent des doublons
df2 =df2.drop(['numéro_circonscription', 'codes_postaux'], axis=1)


# df : enlever les ages < 15 ans et > 65 ans pour garder la population en âge de travailler (les 20-64 ans)
df = df[(df['AGEQ80_17'] >= 15)  & (df['AGEQ80_17'] < 65)]

# Enlever les lignes où NB = 0
df = df[df['NB'] != 0]

# pop active en âge de travailler en France (19-64 ans) : 39 281 641
df.NB.sum()


# ##### Réorganiser df : enlever le critère de l'âge et de MOCO et ne garder que le sexe et le reste des colonnes

# Cela permet de préparer la fusion avec fusion_2
df = df.drop(columns='AGEQ80_17')


df = df.groupby(['LIBGEO', 'SEXE', 'CODGEO',  'REG', 'DEP']).agg({'NB': 'sum'}).reset_index()


# ##### Fusion de df2 et df3 : fusion_1

# Il y aura un peu de perte, notamment pour les entreprises et enlever les doublons que cela génère
fusion_1 = df3.merge(df2, on=[ 'LIBGEO', 'DEP', 'REG'], how='inner')
#fusion_1= fusion_1.drop_duplicates(subset='CODGEO_x')


# renommer CODGEO_x et enlever CODGE_y
fusion_1 = fusion_1.rename(columns={'CODGEO_x' : 'CODGEO'})
fusion_1 = fusion_1.drop(columns='CODGEO_y')


# il y a 33 031 villes avec un nom unique
fusion_1.LIBGEO.nunique()


# Il faut enlever 64 villes qui ont '-' pour longitude
(fusion_1['longitude'] == '-').sum()
fusion_1 = fusion_1[fusion_1['longitude'] != '-']


# remplacer les 139 nan des DEP avec les 2 premiers chiffres de CODGEO
df4['DEP'].fillna(df4['CODGEO'].str[:2], inplace=True)


# ##### Fusion 2

# 19 277 732 hommes
df[df['SEXE']==1]['NB'].sum()


# 20 003 909 femmes
df[df['SEXE']==2]['NB'].sum()


fusion_2 = df4.merge(fusion_1, on=['CODGEO',  'LIBGEO','DEP', 'REG', 'nom_REG'], how='inner')


# fusion_2 : enlever les colonnes inutiles des salaires en fonction de l'âge
fusion_2 = fusion_2.drop(columns=[ 'EU_circo' , 'nom_région', 'SNHMF1814' , 'SNHMF2614' ,'SNHMF5014' , 'SNHMH1814' , 'SNHMH2614' 
                                  , 'SNHMH5014' , 'SNHM1814' , 'SNHM2614' ,'SNHM5014'])


# renommer le salaire moyen et les salaires en fonction des CSP (l'âge n'est pas retenu)
fusion_2.rename(columns={'SNHMH14': 'sal_moy_h', # salaire moyen homme
                   'SNHMF14': 'sal_moy_f', # salaire moyen femme
                   'SNHMHC14' : 'cadr_h' ,# salaire moyen d'un cadre homme
                   'SNHMFC14': 'cadr_f',# salaire moyen des cadres femmes
                   'SNHMHP14' : 'cadr_moy_h',# salaire moyen des cadres moyens hommes
                   'SNHMFP14': 'cadr_moy_f' ,# salaire moyen des cadres moyens femmes
                   'SNHMHE14': 'employé', # salaire moyen d'un employé
                   'SNHMFE14' : 'employée',# salairem moyen d'une employée
                   'SNHMHO14': 'ouvrier',# salaire moyen d'un ouvrier
                   'SNHMFO14': 'ouvrière' ,# salaire moyen d'une ouvrière
                   'SNHM14': 'sal_moyen_tot',# salaire moyen tous sexes confondus
                   'SNHMC14': 'sal_moyen_cadre', # salaire moyen des cadres
                   'SNHMP14' : 'sal_moyen_cadre_moy',  # salaire moyen des cadres moyens
                   'SNHME14' : 'sal_moyen_employés', # salaire moyen des employés
                   'SNHMO14' :'sal_moyen_ouvriers' # salaire moyen des ouvriers
                  }, inplace=True) 

# regrouper les entreprises en catégories : <10 = micro ; entre 10 et 49 = petite ; de 50 à 199 = moyenne ; > 200 grande (cf. Guide de 
# l’utilisateur pour la définition des PME, p.11)
fusion_2['Micro'] = fusion_2['E14TS1'] + fusion_2['E14TS6']
fusion_2['Petite'] = fusion_2['E14TS10'] + fusion_2['E14TS20']
fusion_2['Moyenne'] = fusion_2['E14TS50'] + fusion_2['E14TS100']
fusion_2['Grande'] = fusion_2['E14TS200'] + fusion_2['E14TS500']

# Changer le total des entreprises en enlevant les entreprises de taille inconnue, c-a-d en additionnant 
# les entreprises de taille connue
fusion_2['nb_tot_ent'] = fusion_2['Micro'] + fusion_2['Petite'] + fusion_2['Moyenne'] + fusion_2['Grande'] 

# enlever les colonnes devenues inutiles concernant  les salaires ainsi que les entreprises de taille inconnue
fusion_2 = fusion_2.drop(columns=['E14TS1' , 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500'
                                  ,'E14TS0ND', 'E14TST', 'E14TS0ND' ])


# création de nom_REG en fonctiondes valeur de REG
fusion_2['nom_REG'] = fusion_2['REG'].map(nom_regions)


# remplacer les , par .
fusion_2['longitude'] = fusion_2['longitude'].replace({',' : '.'})


# il y a des NaN qu'il faut enlever ; on enlève éloignement plutôt que les lignes car autrement ça fait disparaître les
# le DEP n. 75 ; on garde 'latitude', 'longitude',
# on enlève aussi "chef.lieu_région" et "préfecture"
fusion_2 = fusion_2.drop(columns = [ 'éloignement', "chef.lieu_région" , "préfecture"])


# ### Fusion complète

jeu = df.merge(fusion_2, on=['CODGEO', 'LIBGEO', 'REG', 'DEP'], how ='inner')


# Enlever les doublons, il faut garder uniquement les 1eres occurences de LIBGEO quand la valeur de SEXE est identique
jeu = jeu.drop_duplicates(subset=['LIBGEO', 'SEXE'], keep='first')


# Enlever les données géo de jeu 
jeu = jeu.drop(columns= [ 'CODGEO'])


# Créer jeu_simple en réduisant jeu en fonction de LIBGEO, pour éviter les répétitions 
# tout en gardant les autres colonnes, sauf SEXE,  'AGEQ80_17', MOCO et MOCO_hab
jeu_simple = jeu.groupby('LIBGEO', as_index=False).agg({
    'NB': 'sum',         
    'REG': 'first',
    'nom_REG': 'first',
    'sal_moyen_tot': 'first',
    'sal_moyen_cadre': 'first',
    'sal_moyen_cadre_moy': 'first',
    'sal_moyen_employés': 'first',
    'sal_moyen_ouvriers': 'first',
    'sal_moy_f': 'first',
    'cadr_f': 'first',
    'cadr_moy_f': 'first',
    'employée': 'first',
    'ouvrière': 'first',
    'sal_moy_h': 'first',
    'cadr_h': 'first',
    'cadr_moy_h': 'first',
    'employé': 'first',
    'ouvrier': 'first',
    'DEP': 'first',
    'Micro': 'first',
    'Petite': 'first',
    'Moyenne': 'first',
    'Grande': 'first',
    'nb_tot_ent': 'first', 
})

st.header("French Industry : présentation du rapport")

# Texte à encadrer
texte_encadre = """
Projet réalisé dans le cadre de la formation Data Analyst de Datascientest

Promotion Mai 2023, formation continue

Auteurs :
- Kévin Drion
- Olivier Régol
"""

# Affichage du texte encadré avec st.markdown
st.write(f'<div style="border: 1px solid #ddd; padding: 10px;">{texte_encadre}</div>', unsafe_allow_html=True)


# Division du code en différentes sections
section_1 = st.checkbox("Carte de la répartition des entreprises en France")
section_2 = st.checkbox("Population d'intérêt")
section_3 = st.checkbox("Entreprises")
section_4 = st.checkbox("Répartition des salaires")

st.subheader("ACP et Machine learning")
section_5 = st.checkbox("ACP")
section_6 = st.checkbox("Machine learning")





# Section "Carte de la répartition des entreprises en France"
if section_1:
    
    # ##### Carte de la répartition des entreprises en France
    st.header('Cartes de la répartition des entreprises en France')
    geo = jeu
    # enlever les valeurs manquantes de latitude et mettre la longitude en float
    geo['latitude'] = geo['latitude'].dropna()
    geo['longitude'] = geo['longitude'].str.replace(',', '.').astype(float)
    # transformer les longitudes et latitudes pour être de type mercator
    # https://stackoverflow.com/questions/73345967/how-to-recreate-coordinate-transformation-from-vesselfinder-epsg4326-to-epsg3

    def lon_to_x(longitude):
        return float(longitude) * 20037508.34 / 180
    def lat_to_y(latitude):
        return np.log(np.tan((90 + float(latitude)) * np.pi / 360)) / (np.pi / 180) * 20037508.34 / 180

    # Convertir les coordonnées dans le DataFrame
    geo['x'] = geo['longitude'].apply(lon_to_x)
    geo['y'] = geo['latitude'].apply(lat_to_y)

    # les conditions
    filtered_micro = geo[geo['Micro'] >= 1]
    filtered_grande = geo[geo['Grande'] >= 1]
    filtered_moyenne = geo[geo['Moyenne'] >= 1]
    filtered_petite = geo[geo['Petite'] >= 1]

    source_micro  = ColumnDataSource(filtered_micro)
    source_grande = ColumnDataSource(filtered_grande)
    source_moyenne = ColumnDataSource(filtered_moyenne)
    source_petite = ColumnDataSource(filtered_petite)

    # Utiliser une URL de tuile avec {s} pour la distribution de charge
    tile_options = {'url': 'https://a.basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png', 'attribution': '© OpenStreetMap contributors & © CartoDB'}
    tuile = WMTSTileSource(**tile_options)

    # la carte
    # à partir de https://epsg.io/map#srs=3857&x=-467183.116879&y=6354668.783516&z=5&layer=streets
    p = figure(x_range=(-633510, 1106449),
               y_range=(6618835, 5206282),
               x_axis_type='mercator',
               y_axis_type='mercator')
    p.add_tile(tuile)

    # les couleurs des points
    p.circle(x='x', y='y', fill_color='black', size=4, source=source_micro, legend_label='Micro')
    p.circle(x='x', y='y', fill_color='orange', size=4, source=source_petite, legend_label='Petite')
    p.circle(x='x', y='y', fill_color='blue', size=4, source=source_moyenne, legend_label='Moyenne')
    p.circle(x='x', y='y', fill_color='red', size=4, source=source_grande, legend_label='Grande')

    p.legend.title = 'Catégories'
    p.legend.label_text_font_size = '10pt'
    p.legend.click_policy = 'hide'

    st.write(p)

# Section "Population d'intérêt"
if section_2:
    st.title('Analyse des Données')
    # ### Analyse
    # ##### Répartition hommes - femmes aux niveaux national et régional
    st.header('Répartition des hommes et des femmes')

    # Nombre d'hommes et de femmes en France métropolitaine, selon 2 versions du fichier 'population'
    # boucle for pour voir le nombre totale de personnes dans les 2 fichiers
    # et le nombre d'hommes et de femmes
    fichiers = [df, jeu]
    nom_fichiers = ['df', 'jeu']
    for i in range(len(fichiers)):
        tot_hom = fichiers[i][fichiers[i]['SEXE'] == 1]['NB'].sum()
        tot_fem = fichiers[i][fichiers[i]['SEXE'] == 2]['NB'].sum()
        tot_personnes = tot_hom + tot_fem

        st.write(f'Nombre total de personnes dans {nom_fichiers[i]} : {tot_personnes}')
        st.write(f'Nombre total d\'hommes dans {nom_fichiers[i]} : {tot_hom}')
        st.write(f'Nombre total de femmes dans {nom_fichiers[i]} : {tot_fem}')
        st.write('---------------------------------------------')


    # Somme des hommes et des femmes dans les régions
    tot_hom_df = df[df['SEXE'] == 1].groupby('REG')['NB'].sum()
    tot_fem_df = df[df['SEXE'] == 2].groupby('REG')['NB'].sum()
    total_df = tot_hom_df + tot_fem_df

    tot_hom_jeu = jeu[jeu['SEXE'] == 1].groupby('REG')['NB'].sum()
    tot_fem_jeu = jeu[jeu['SEXE'] == 2].groupby('REG')['NB'].sum()
    total_jeu = tot_hom_jeu + tot_fem_jeu

    # Pourcentages d'hommes et de femmes dans les régions selon les 2 fichiers
    pourcen_hom_df = (tot_hom_df / total_df) * 100
    pourcen_fem_df = (tot_fem_df / total_df) * 100
    pourcen_hom_jeu = (tot_hom_jeu / total_jeu) * 100
    pourcen_fem_jeu = (tot_fem_jeu / total_jeu) * 100


    # Somme totale des hommes et des femmes dans les deux fichiers
    total_hommes_df = tot_hom_df.sum()
    total_femmes_df = tot_fem_df.sum()
    total_hommes_jeu = tot_hom_jeu.sum()
    total_femmes_jeu = tot_fem_jeu.sum()

    # Calcul des pourcentages d'hommes et de femmes par rapport à la somme totale dans chaque fichier
    pourcentage_hommes_df = (tot_hom_df / total_hommes_df) * 100
    pourcentage_femmes_df = (tot_fem_df / total_femmes_df) * 100
    pourcentage_hommes_jeu = (tot_hom_jeu / total_hommes_jeu) * 100
    pourcentage_femmes_jeu = (tot_fem_jeu / total_femmes_jeu) * 100

    # Création d'un DataFrame pour afficher les pourcentages en colonnes
    gender_distribution__par_reg = pd.DataFrame({
        'Region': tot_hom_df.index,  # Les régions
        '% Hommes (df)': pourcentage_hommes_df.values,
        '% Hommes (jeu)': pourcentage_hommes_jeu.values,
        '% Femmes (df)': pourcentage_femmes_df.values,
        '% Femmes (jeu)': pourcentage_femmes_jeu.values
    })

    st.write(gender_distribution__par_reg.round(2))


    # Données
    labels = df['REG'].unique()  
    hommes_df = pourcen_hom_df.values
    femmes_df = pourcen_fem_df.values
    hommes_jeu = pourcen_hom_jeu.values
    femmes_jeu = pourcen_fem_jeu.values

    # Créer les sous-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Camembert 1 pour le premier fichier
    ax1.pie([hommes_df.sum(), femmes_df.sum()], labels=['Hommes', 'Femmes'], autopct='%1.1f%%', startangle=90)
    ax1.set_title("Répartition hommes et femmes (Fichier d'origine)")

    # Camembert 2 pour le deuxième fichier
    ax2.pie([hommes_jeu.sum(), femmes_jeu.sum()], labels=['Hommes', 'Femmes'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Répartition hommes et femmes (Fichier pour notre analyse)')
    # Désactivez l'avertissement PyplotGlobalUseWarning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

    st.write("Le jeu d'analyse surreprésente l'Île-de-France par rapport aux données de départ (+5 points). Cela signifie que nous avons plus d'information concernant les salaires dans les villes de cette région que dans les autres et cela a une incidence lors des fusions des différents fichiers.") 


    # Section "Population d'intérêt"
if section_3:
    st.title('Entreprises')
    st.write("Notre jeu de données se compose de 1 157 883 entreprises.")

    # ### Entreprises
    # nombre d'entreprises par région
    entreprises_reg_jeu_simple = jeu_simple.groupby(['nom_REG'])['nb_tot_ent'].sum()
    entreprises_reg_jeu_simple= entreprises_reg_jeu_simple.reset_index()

    entreprises_reg = pd.DataFrame({
        'Région': entreprises_reg_jeu_simple['nom_REG'],
        "Nombre d'entreprises": entreprises_reg_jeu_simple['nb_tot_ent']
    })


    st.markdown("Nombre d'entreprises par région")
    st.table(entreprises_reg)


    
    # ordonner le graphe
    entreprises_reg = entreprises_reg.sort_values(by="Nombre d'entreprises", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))  
    ax.plot(entreprises_reg["Nombre d'entreprises"], entreprises_reg["Région"], marker='o', linestyle='-')
    ax.set_xlabel("Nombre d'entreprises")
    ax.set_title("Nombre d'entreprises par région")

    
    # Utiliser uniquement une valeur sur deux pour les étiquettes
    x_indices = np.arange(0, len(entreprises_reg), 2)
    ax.set_xticks(entreprises_reg["Nombre d'entreprises"].iloc[x_indices])
    ax.set_xticklabels(entreprises_reg["Nombre d'entreprises"].iloc[x_indices], ha='right', rotation=45)


    # Afficher la figure dans Streamlit
    st.pyplot(fig)

    # Il y a 1 157 836 entreprises dans notre jeu d'analyse.
    entreprises_reg_jeu_simple.nb_tot_ent.sum()

    grandes_entreprises = jeu_simple[['REG', 'nom_REG', 'Grande']]


    grandes_entreprises = grandes_entreprises.groupby('nom_REG')['Grande'].sum().reset_index()
    


    # il y a 7061 grandes entreprises
    total_grandes_entreprises = grandes_entreprises['Grande'].sum()
    st.write(f"Il y a {total_grandes_entreprises} grandes entreprises, de plus de 500 employés")

    # voir la répartition en % des grandes entreprises dans les régions
    grandes_entreprises['Pourcentage'] = (grandes_entreprises['Grande'] / total_grandes_entreprises) * 100
    # Trier les valeurs
    grandes_entreprises = grandes_entreprises.sort_values(by="Pourcentage", ascending=True)

    # Créer le graphique en courbe avec Plotly Express
    fig = px.line(
        grandes_entreprises,
        x='Pourcentage',
        y='nom_REG',
        title='Répartition des grandes entreprises dans les régions, en % du total',
        labels={'Pourcentage': '% du total', 'nom_REG': 'Régions'},
        line_shape='linear', 
        markers=True,  
        template='plotly_white',  
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)

     
   


    # faire un test de Pearson pour voir la force de la relation entre la présence de grandes entreprises dans une région et 
    # la présence d'autres entreprises 

    # Créer les variables
    nombre_entreprises = entreprises_reg["Nombre d'entreprises"]
    pourcentage_grandes_entreprises = grandes_entreprises['Pourcentage']

    # Test de corrélation de Pearson
    correlation, p_value = pearsonr(nombre_entreprises, pourcentage_grandes_entreprises)

    # Test de corrélation 
    correlation, p_value = pearsonr(pourcentage_grandes_entreprises, nombre_entreprises)
    st.write("Relation entre la présence de grandes entreprises et d'entreprises de plus petite taille dans une région")
    st.write(f"Corrélation de Pearson : {correlation}")
    st.write(f"Valeur de p (p-value) : {p_value}")



    # ##### Lien entre le nombre d'entreprises dans une région et le nombre d'habitants

    # faire un test de Pearson pour voir le lien entre le nombre d'habitants en âge de travailler dans une région et le nombre
    # d'entreprises
    tot_hab_reg_jeu_simple = jeu_simple.groupby(['nom_REG'])['NB'].sum()
    entreprises_reg_jeu_simple= jeu_simple.groupby(['nom_REG'])['nb_tot_ent'].sum()

    # Créer un DataFrame pour le nombre d'habitants par région
    corr_jeu_simple = pd.DataFrame({
        "Nombre d'entreprises": entreprises_reg_jeu_simple,
        "Nombre d'habitants": tot_hab_reg_jeu_simple,
    })

    # Calcul de la corrélation entre le nombre d'habitants et le nombre d'entreprises
    corr, pvalue = pearsonr(corr_jeu_simple["Nombre d'habitants"], corr_jeu_simple["Nombre d'entreprises"])

    st.write(f"La corrélation entre le nombre d'habitants et le nombre d'entreprises est de {corr:.2f} avec une p-value de {pvalue:.2f}")


    
    corr_jeu_simple = corr_jeu_simple.sort_values(by="Nombre d'habitants", ascending=True)

    # Créer un graphique en courbe avec les données du DataFrame corr_jeu_simple
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(corr_jeu_simple["Nombre d'habitants"], corr_jeu_simple.index,  marker='o', linestyle='-', color='b')

    ax.set_title("Nombre d'habitants en âge de travailler par région")
    ax.set_xlabel("Nombre d'habitants (en million)")

    st.pyplot()

    # scatterplot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(corr_jeu_simple["Nombre d'habitants"], corr_jeu_simple["Nombre d'entreprises"])
    ax.set_xlabel("Nombre d'habitants (en million)")
    ax.set_ylabel("Nombre d'entreprises (en million)")
    ax.set_title("Relation entre le nombre d'habitants des régions et le nombre d'entreprises")
    plt.grid(True)

    st.pyplot() ;


    

# Section "Répartition des salaires"
if section_4:
    
    nom = ['Cadre', 'Cadre moyen', 'Employé', 'Ouvrier']
    taux_globaux_reg = jeu_simple.groupby('nom_REG').agg({'sal_moyen_cadre' : 'mean'
                                                          , 'sal_moyen_cadre_moy' : 'mean'
                                                          , 'sal_moyen_employés' : 'mean'
                                                          , 'sal_moyen_ouvriers' : 'mean'
                                                          })
    taux_globaux_reg = taux_globaux_reg.sort_values(by='sal_moyen_cadre', ascending=False)
    taux_globaux_reg.columns = nom

    regions = taux_globaux_reg.index
    cadre = taux_globaux_reg['Cadre']
    cadre_moyen = taux_globaux_reg['Cadre moyen']
    employe = taux_globaux_reg['Employé']
    ouvrier = taux_globaux_reg['Ouvrier']

    # ##### # Ecarts intra-sexe selon la CSP
    # supprimer la valeur car elle est aberrante
    jeu_simple = jeu_simple.drop(jeu_simple[jeu_simple['ouvrier'] >= 50].index)
    jeu_simple.loc[jeu_simple['cadr_moy_h']>=60].drop_duplicates(subset='LIBGEO')

    
    #1er graphique
    # salaire moyen par région
    sal_moy_reg = jeu_simple.groupby('nom_REG')['sal_moyen_tot'].mean().reset_index()
    sal_moy_reg = sal_moy_reg.sort_values(by='sal_moyen_tot', ascending = True)


    st.write("Le taux horaire moyen varie entre 10 et 43 euros.")
    # boxplot de sal_moyen au niveau national : Variation du taux horaire net moyen global en France
    st.header('Variation du taux horaire net moyen global en France')
    # Utilisation de Plotly Express pour créer un boxplot
    fig = px.box(jeu, y='sal_moyen_tot', title='Variation du taux horaire net moyen global en France')
    # Afficher le graphe
    st.plotly_chart(fig)

    

    st.header('Taux horaires moyens et CSP')
    
    # 2e graphique
    st.write("Les taux régionaux varient de 12.62 euros en Corse à 17.39 euros en Île-de-France.")
    # Courbe des taux horaires moyens par région toutes CSP confondues
    fig_line = px.line(sal_moy_reg, x='sal_moyen_tot', y='nom_REG', 
                       title='Taux horaire net moyen par région, toutes CSP confondues',
                       labels={'sal_moyen_tot': 'Taux horaire net moyen (en €)', 'nom_REG': 'Région'})
    st.plotly_chart(fig_line)
    

    
    # 3e graphique
    # Faire un graphiques qui résume les taux horaires globaux par CSP et par région
    regions = taux_globaux_reg.index
    cadre = taux_globaux_reg['Cadre']
    cadre_moyen = taux_globaux_reg['Cadre moyen']
    employe = taux_globaux_reg['Employé']
    ouvrier = taux_globaux_reg['Ouvrier']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(regions, cadre, marker='o', label='Cadre')
    ax.plot(regions, cadre_moyen, marker='o', label='Cadre moyen')
    ax.plot(regions, employe, marker='o', label='Employé')
    ax.plot(regions, ouvrier, marker='o', label='Ouvrier')

    # Configuration du graphique
    ax.set_title('Variation des taux horaires nets des CSP par région')
    ax.set_ylabel('Taux horaires nets moyens (en €)')
    ax.set_xticklabels(regions, rotation=60, ha='right')
    ax.legend()
    st.pyplot(fig)

    #4     
    
    # Calculer les salaires moyens entre les hommes et les femmes par région
    sal_hom_fem_reg = jeu_simple.groupby('nom_REG').agg(
    sal_moyen_hom=('sal_moy_h', 'mean'),  
    sal_moyen_fem=('sal_moy_f', 'mean'))

    # Calculer l'écart salarial en %
    sal_hom_fem_reg['écart_salaire_%'] = ((sal_hom_fem_reg['sal_moyen_hom'] - sal_hom_fem_reg['sal_moyen_fem']) / sal_hom_fem_reg['sal_moyen_hom']) * 100
    sal_hom_fem_reg = sal_hom_fem_reg.sort_values(by='écart_salaire_%', ascending=True).round(2)

    

    # Données
    regions = sal_hom_fem_reg.index
    ecart_salaire_percent = sal_hom_fem_reg['écart_salaire_%']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Créer un axe pour la courbe (écart de salaire en pourcentage)
    plt.plot(ecart_salaire_percent, regions, color='red', marker='o')

    ax.set_xlabel("Écart salarial en %")
    ax.set_title("Écart salarial en % par région, en fonction du sexe")
    st.pyplot()

    
    st.header("Variation des taux horaires  selon les sexes")
    st.write("Différences entre les CSP au sein du même sexe")

    

    # salaire moyen par région
    sal_moy_reg = jeu_simple.groupby('nom_REG')['sal_moyen_tot'].mean().reset_index()
    sal_moy_reg = sal_moy_reg.sort_values(by='sal_moyen_tot', ascending = True)


   
    # salaire des hommes et des femmes selon la CSP
    data_hom = [jeu_simple['cadr_h'], jeu_simple['cadr_moy_h'], jeu_simple['employé'], jeu_simple['ouvrier']]
    labels_hom = ['Cadre', 'Cadre moyen', 'employé', 'ouvrier']

    data_fem = [jeu_simple['cadr_f'], jeu_simple['cadr_moy_f'], jeu_simple['employée'], jeu_simple['ouvrière']]
    labels_fem = ['Cadre', 'Cadre moyen', 'employée', 'ouvrière']


    # Boxplot des salaires moyens des hommes selon leur CSP
    fig_hom = px.box(jeu_simple, y=data_hom, labels=labels_hom,
                 title='Taux horaire net moyen des hommes en fonction de leur CSP')
    fig_hom.update_yaxes(title_text='Taux horaire Moyen (en €)', range=(0, 65))
    st.plotly_chart(fig_hom)




    





    # Boxplot des salaires moyens des femmes selon leur CSP
    fig_fem = px.box(jeu_simple,y=data_fem, labels=labels_fem,
                     title='Taux horaire net moyen des femmes en fonction de leur CSP')
    fig_fem.update_yaxes(title_text='Taux horaire Moyen (en €)', range=(0, 65))
    st.plotly_chart(fig_fem)


       

    st.write("Différences entre les sexes au sein de la même CSP (supérieure ou inférieure)")
    # Écarts entre les cadres hommes et femmes et les cadres moyens hommes et femmes
    fig_ecart = px.box(jeu_simple[['cadr_h', 'cadr_f', 'cadr_moy_h', 'cadr_moy_f']], 
                       labels={'variable': 'Catégorie', 'value': 'Taux horaire'},
                       title='Variation du taux horaire moyen des CSP supérieures selon le sexe')
    fig_ecart.update_yaxes(range=(0, 65))

    # Affichage du graphique
    st.plotly_chart(fig_ecart)


    # Écarts entre les employés hommes et femmes et les ouvriers hommes et femmes
    fig_ecart_inf = px.box(jeu_simple[['employé', 'employée', 'ouvrier', 'ouvrière']], 
                           labels={'variable': 'Catégorie', 'value': 'Taux horaire'},
                           title='Variation du taux horaire moyen des CSP inférieures selon le sexe')
    fig_ecart_inf.update_yaxes(range=(0, 65))

    # Affichage du graphique
    st.plotly_chart(fig_ecart_inf)

     

    #st.subheader("ACP et Machine learning")

# Section "ACP pour les moyennes des variables des CSP par département"
if section_5:
    
    # ##### # ### Répartition par quartiles des salaires au niveau régional en fonction de la CSP et du sexe

    # ##### Cadres hommes

    # dispersion des taux horaires nets des cadres hommes dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les cadres hommes (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['cadr_h'].quantile(0.25)  
    cadr_h_Q1 = jeu_simple[jeu_simple['cadr_h'] < Q1]
    cadr_h_Q1 = cadr_h_Q1.drop_duplicates(subset='LIBGEO')
    cadr_h_Q1 = cadr_h_Q1[['nom_REG', 'DEP','LIBGEO', 'cadr_h', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les cadres hommes (Q1 - Q2)
    Q2 = jeu_simple['cadr_h'].quantile(0.5)
    cadr_h_Q2 = jeu_simple[(jeu_simple['cadr_h'] >= jeu_simple['cadr_h'].quantile(0.25) )& (jeu_simple['cadr_h'] <= jeu_simple['cadr_h'].quantile(0.5))]
    cadr_h_Q2 = cadr_h_Q2.drop_duplicates(subset='LIBGEO')
    cadr_h_Q2 = cadr_h_Q2[['nom_REG', 'DEP','LIBGEO', 'cadr_h', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['cadr_h'].quantile(0.75)  
    cadr_h_Q3 = jeu_simple[(jeu_simple['cadr_h'] > jeu_simple['cadr_h'].quantile(0.5) )& (jeu_simple['cadr_h'] <= jeu_simple['cadr_h'].quantile(0.75))]
    cadr_h_Q3 = cadr_h_Q3.drop_duplicates(subset='LIBGEO')
    cadr_h_Q3 = cadr_h_Q3[['nom_REG', 'DEP','LIBGEO', 'cadr_h', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les cadres hommes
    Q4 = cadr_h_Q4 = jeu_simple[(jeu_simple['cadr_h'] > Q3 )]
    cadr_h_Q4 = cadr_h_Q4.drop_duplicates(subset='LIBGEO')
    cadr_h_Q4 = cadr_h_Q4[['nom_REG', 'DEP','LIBGEO', 'cadr_h', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des cadres hommes
    data = {"Quartiles du salaire moyen des cadres hommes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_cadr_h = pd.DataFrame(data).T
    


    # création de villes_cadr_h qui prend le nom de la ville et sa catégorie
    cadr_h_Q1['Cadre_h'] = 'Q1'
    cadr_h_Q2['Cadre_h'] = 'Q2'
    cadr_h_Q3['Cadre_h'] = 'Q3'
    cadr_h_Q4['Cadre_h'] = 'Q4'

    # concaténation
    villes_cadr_h = pd.concat([cadr_h_Q1, cadr_h_Q2, cadr_h_Q3, cadr_h_Q4], ignore_index=True)
    villes_cadr_h = villes_cadr_h.reset_index(drop=True)

    # nom de la ville et catégorie d'interet
    villes_cadr_h = villes_cadr_h[['LIBGEO', 'Cadre_h','nom_REG', 'DEP']]
    


    # ##### Cadres femmes

    # dispersion des taux horaires nets des cadres femmes dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les cadres femmes (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['cadr_f'].quantile(0.25)  
    cadr_f_Q1 = jeu_simple[jeu_simple['cadr_f'] < Q1]
    cadr_f_Q1 = cadr_f_Q1.drop_duplicates(subset='LIBGEO')
    cadr_f_Q1 = cadr_f_Q1[['nom_REG', 'DEP','LIBGEO', 'cadr_f', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les cadres femmes (Q1 - Q2)
    Q2 = jeu_simple['cadr_f'].quantile(0.5)
    cadr_f_Q2 = jeu_simple[(jeu_simple['cadr_f'] >= jeu_simple['cadr_f'].quantile(0.25) )& (jeu_simple['cadr_f'] <= jeu_simple['cadr_f'].quantile(0.5))]
    cadr_f_Q2 = cadr_f_Q2.drop_duplicates(subset='LIBGEO')
    cadr_f_Q2 = cadr_f_Q2[['nom_REG', 'DEP','LIBGEO', 'cadr_f', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['cadr_f'].quantile(0.75)  
    cadr_f_Q3 = jeu_simple[(jeu_simple['cadr_f'] > jeu_simple['cadr_f'].quantile(0.5) )& (jeu_simple['cadr_f'] <= jeu_simple['cadr_f'].quantile(0.75))]
    cadr_f_Q3 = cadr_f_Q3.drop_duplicates(subset='LIBGEO')
    cadr_f_Q3 = cadr_f_Q3[['nom_REG', 'DEP','LIBGEO', 'cadr_f', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les cadres femmes
    Q4 = cadr_f_Q4 = jeu_simple[(jeu_simple['cadr_f'] > Q3 )]
    cadr_f_Q4 = cadr_f_Q4.drop_duplicates(subset='LIBGEO')
    cadr_f_Q4 = cadr_f_Q4[['nom_REG', 'DEP','LIBGEO', 'cadr_f', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des cadres femmes
    data = {"Quartiles du salaire moyen des cadres femmes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_cadr_f = pd.DataFrame(data).T
    


    # création de villes_cadr_f qui prend le nom de la ville et sa catégorie
    cadr_f_Q1['Cadre_f'] = 'Q1'
    cadr_f_Q2['Cadre_f'] = 'Q2'
    cadr_f_Q3['Cadre_f'] = 'Q3'
    cadr_f_Q4['Cadre_f'] = 'Q4'

    # concaténation
    villes_cadr_f = pd.concat([cadr_f_Q1, cadr_f_Q2, cadr_f_Q3, cadr_f_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_cadr_f = villes_cadr_f[['LIBGEO', 'Cadre_f','nom_REG', 'DEP']]


    # ##### Cadres moyens hommes

    # dispersion des taux horaires nets des cadres moyens masculins dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les cadres moyens hommes (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['cadr_moy_h'].quantile(0.25)  
    cadr_moy_h_Q1 = jeu_simple[jeu_simple['cadr_moy_h'] < Q1]
    cadr_moy_h_Q1 = cadr_moy_h_Q1.drop_duplicates(subset='LIBGEO')
    cadr_moy_h_Q1 = cadr_moy_h_Q1[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_h', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les cadres moyens hommes (Q1 - Q2)
    Q2 = jeu_simple['cadr_moy_h'].quantile(0.5)
    cadr_moy_h_Q2 = jeu_simple[(jeu_simple['cadr_moy_h'] >= jeu_simple['cadr_moy_h'].quantile(0.25) )& (jeu_simple['cadr_moy_h'] <= jeu_simple['cadr_moy_h'].quantile(0.5))]
    cadr_moy_h_Q2 = cadr_moy_h_Q2.drop_duplicates(subset='LIBGEO')
    cadr_moy_h_Q2 = cadr_moy_h_Q2[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_h', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['cadr_moy_h'].quantile(0.75)  
    cadr_moy_h_Q3 = jeu_simple[(jeu_simple['cadr_moy_h'] > jeu_simple['cadr_moy_h'].quantile(0.5) )& (jeu_simple['cadr_moy_h'] <= jeu_simple['cadr_moy_h'].quantile(0.75))]
    cadr_moy_h_Q3 = cadr_moy_h_Q3.drop_duplicates(subset='LIBGEO')
    cadr_moy_h_Q3 = cadr_moy_h_Q3[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_h', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les cadres moyens hommes
    Q4 = cadr_moy_h_Q4 = jeu_simple[(jeu_simple['cadr_moy_h'] > Q3 )]
    cadr_moy_h_Q4 = cadr_moy_h_Q4.drop_duplicates(subset='LIBGEO')
    cadr_moy_h_Q4 = cadr_moy_h_Q4[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_h', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des cadres moyens hommes
    data = {"Quartiles du salaire moyen des cadres moyens hommes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_cadr_moy_h = pd.DataFrame(data).T
   


    # création de villes_cadr_moy_h qui prend le nom de la ville et sa catégorie
    cadr_moy_h_Q1['cadr_moy_h'] = 'Q1'
    cadr_moy_h_Q2['cadr_moy_h'] = 'Q2'
    cadr_moy_h_Q3['cadr_moy_h'] = 'Q3'
    cadr_moy_h_Q4['cadr_moy_h'] = 'Q4'

    # concaténation
    villes_cadr_moy_h = pd.concat([cadr_moy_h_Q1, cadr_moy_h_Q2, cadr_moy_h_Q3, cadr_moy_h_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_cadr_moy_h = villes_cadr_moy_h[['LIBGEO','cadr_moy_h', 'nom_REG', 'DEP']]


    # ##### Cadres moyens femmes

    # dispersion des taux horaires nets des cadres moyens féminins dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les cadres moyens femmes (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['cadr_moy_f'].quantile(0.25)  
    cadr_moy_f_Q1 = jeu_simple[jeu_simple['cadr_moy_f'] < Q1]
    cadr_moy_f_Q1 = cadr_moy_f_Q1.drop_duplicates(subset='LIBGEO')
    cadr_moy_f_Q1 = cadr_moy_f_Q1[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_f', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les cadres moyens femmes (Q1 _ Q2)
    Q2 = jeu_simple['cadr_moy_f'].quantile(0.5)
    cadr_moy_f_Q2 = jeu_simple[(jeu_simple['cadr_moy_f'] >= jeu_simple['cadr_moy_f'].quantile(0.25) )& (jeu_simple['cadr_moy_f'] <= jeu_simple['cadr_moy_f'].quantile(0.5))]
    cadr_moy_f_Q2 = cadr_moy_f_Q2.drop_duplicates(subset='LIBGEO')
    cadr_moy_f_Q2 = cadr_moy_f_Q2[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_f', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['cadr_moy_f'].quantile(0.75)  
    cadr_moy_f_Q3 = jeu_simple[(jeu_simple['cadr_moy_f'] > jeu_simple['cadr_moy_f'].quantile(0.5) )& (jeu_simple['cadr_moy_f'] <= jeu_simple['cadr_moy_f'].quantile(0.75))]
    cadr_moy_f_Q3 = cadr_moy_f_Q3.drop_duplicates(subset='LIBGEO')
    cadr_moy_f_Q3 = cadr_moy_f_Q3[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_f', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les cadres moyens femmes
    Q4 = cadr_moy_f_Q4 = jeu_simple[(jeu_simple['cadr_moy_f'] > Q3 )]
    cadr_moy_f_Q4 = cadr_moy_f_Q4.drop_duplicates(subset='LIBGEO')
    cadr_moy_f_Q4 = cadr_moy_f_Q4[['nom_REG', 'DEP','LIBGEO', 'cadr_moy_f', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des cadres moyens femmes
    data = {"Quartiles du salaire moyen des cadres moyens femmes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_cadr_moy_f = pd.DataFrame(data).T
    


    # création de villes_cadr_moy_f qui prend le nom de la ville et sa catégorie
    cadr_moy_f_Q1['cadr_moy_f'] = 'Q1'
    cadr_moy_f_Q2['cadr_moy_f'] = 'Q2'
    cadr_moy_f_Q3['cadr_moy_f'] = 'Q3'
    cadr_moy_f_Q4['cadr_moy_f'] = 'Q4'

    # concaténation
    villes_cadr_moy_f = pd.concat([cadr_moy_f_Q1, cadr_moy_f_Q2, cadr_moy_f_Q3, cadr_moy_f_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_cadr_moy_f = villes_cadr_moy_f[['LIBGEO', 'cadr_moy_f','nom_REG', 'DEP']]


    # ##### Employés hommes

    # dispersion des taux horaires nets des employés dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les employés hommes (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['employé'].quantile(0.25)  
    employé_Q1 = jeu_simple[jeu_simple['employé'] < Q1]
    employé_Q1 = employé_Q1.drop_duplicates(subset='LIBGEO')
    employé_Q1 = employé_Q1[['nom_REG', 'DEP','LIBGEO', 'employé', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les employés hommes (Q1 _ Q2)
    Q2 = jeu_simple['employé'].quantile(0.5)
    employé_Q2 = jeu_simple[(jeu_simple['employé'] >= jeu_simple['employé'].quantile(0.25) )& (jeu_simple['employé'] <= jeu_simple['employé'].quantile(0.5))]
    employé_Q2 = employé_Q2.drop_duplicates(subset='LIBGEO')
    employé_Q2 = employé_Q2[['nom_REG', 'DEP','LIBGEO', 'employé', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['employé'].quantile(0.75)  
    employé_Q3 = jeu_simple[(jeu_simple['employé'] > jeu_simple['employé'].quantile(0.5) )& (jeu_simple['employé'] <= jeu_simple['employé'].quantile(0.75))]
    employé_Q3 = employé_Q3.drop_duplicates(subset='LIBGEO')
    employé_Q3 = employé_Q3[['nom_REG', 'DEP','LIBGEO', 'employé', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les employés hommes
    Q4 = employé_Q4 = jeu_simple[(jeu_simple['employé'] > Q3 )]
    employé_Q4 = employé_Q4.drop_duplicates(subset='LIBGEO')
    employé_Q4 = employé_Q4[['nom_REG', 'DEP','LIBGEO', 'employé', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des employés hommes
    data = {"Quartiles du salaire moyen des employés hommes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_employé_H = pd.DataFrame(data).T
   


    # création de villes_employé qui prend le nom de la ville et sa catégorie
    employé_Q1['employé'] = 'Q1'
    employé_Q2['employé'] = 'Q2'
    employé_Q3['employé'] = 'Q3'
    employé_Q4['employé'] = 'Q4'

    # concaténation
    villes_employé = pd.concat([employé_Q1, employé_Q2, employé_Q3, employé_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_employé = villes_employé[['LIBGEO', 'employé','nom_REG', 'DEP']]


    #villes_employé[villes_employé['employé']=='Q4']


    # ##### Employées femmes

    # dispersion des taux horaires nets des employées dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les employées (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['employée'].quantile(0.25)  
    employée_Q1 = jeu_simple[jeu_simple['employée'] < Q1]
    employée_Q1 = employée_Q1.drop_duplicates(subset='LIBGEO')
    employée_Q1 = employée_Q1[['nom_REG', 'DEP','LIBGEO', 'employée', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les employées (Q1 _ Q2)
    Q2 = jeu_simple['employée'].quantile(0.5)
    employée_Q2 = jeu_simple[(jeu_simple['employée'] >= jeu_simple['employée'].quantile(0.25) )& (jeu_simple['employée'] <= jeu_simple['employée'].quantile(0.5))]
    employée_Q2 = employée_Q2.drop_duplicates(subset='LIBGEO')
    employée_Q2 = employée_Q2[['nom_REG', 'DEP','LIBGEO', 'employée', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['employée'].quantile(0.75)  
    employée_Q3 = jeu_simple[(jeu_simple['employée'] > jeu_simple['employée'].quantile(0.5) )& (jeu_simple['employée'] <= jeu_simple['employée'].quantile(0.75))]
    employée_Q3 = employée_Q3.drop_duplicates(subset='LIBGEO')
    employée_Q3 = employée_Q3[['nom_REG', 'DEP','LIBGEO', 'employée', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les employéess
    Q4 = employée_Q4 = jeu_simple[(jeu_simple['employée'] > Q3 )]
    employée_Q4 = employée_Q4.drop_duplicates(subset='LIBGEO')
    employée_Q4 = employée_Q4[['nom_REG', 'DEP','LIBGEO', 'employée', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des employées femmes
    data = {"Quartiles du salaire moyen des employées femmes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_employée_F = pd.DataFrame(data).T
    


    # création de villes_employée qui prend le nom de la ville et sa catégorie
    employée_Q1['employée'] = 'Q1'
    employée_Q2['employée'] = 'Q2'
    employée_Q3['employée'] = 'Q3'
    employée_Q4['employée'] = 'Q4'

    # concaténation
    villes_employée = pd.concat([employée_Q1, employée_Q2, employée_Q3, employée_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_employée = villes_employée[['LIBGEO', 'employée','nom_REG', 'DEP']]


    # ##### Ouvriers

    # dispersion des taux horaires nets des ouvriers dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les ouvriers (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['ouvrier'].quantile(0.25)  
    ouvrier_Q1 = jeu_simple[jeu_simple['ouvrier'] < Q1]
    ouvrier_Q1 = ouvrier_Q1.drop_duplicates(subset='LIBGEO')
    ouvrier_Q1 = ouvrier_Q1[['nom_REG', 'DEP','LIBGEO', 'ouvrier', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les ouvriers (Q1 _ Q2)
    Q2 = jeu_simple['ouvrier'].quantile(0.5)
    ouvrier_Q2 = jeu_simple[(jeu_simple['ouvrier'] >= jeu_simple['ouvrier'].quantile(0.25) )& (jeu_simple['ouvrier'] <= jeu_simple['ouvrier'].quantile(0.5))]
    ouvrier_Q2 = ouvrier_Q2.drop_duplicates(subset='LIBGEO')
    ouvrier_Q2 = ouvrier_Q2[['nom_REG', 'DEP','LIBGEO', 'ouvrier', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['ouvrier'].quantile(0.75)  
    ouvrier_Q3 = jeu_simple[(jeu_simple['ouvrier'] > jeu_simple['ouvrier'].quantile(0.5) )& (jeu_simple['ouvrier'] <= jeu_simple['ouvrier'].quantile(0.75))]
    ouvrier_Q3 = ouvrier_Q3.drop_duplicates(subset='LIBGEO')
    ouvrier_Q3 = ouvrier_Q3[['nom_REG', 'DEP','LIBGEO', 'ouvrier', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les ouvriers
    Q4 = ouvrier_Q4 = jeu_simple[(jeu_simple['ouvrier'] > Q3 )]
    ouvrier_Q4 = ouvrier_Q4.drop_duplicates(subset='LIBGEO')
    ouvrier_Q4 = ouvrier_Q4[['nom_REG', 'DEP','LIBGEO', 'ouvrier', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des ouvriers hommes
    data = {"Quartiles du salaire moyen des ouvriers hommes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_ouvrier_H = pd.DataFrame(data).T
   


    # création de villes_ouvrier qui prend le nom de la ville et sa catégorie
    ouvrier_Q1['ouvrier'] = 'Q1'
    ouvrier_Q2['ouvrier'] = 'Q2'
    ouvrier_Q3['ouvrier'] = 'Q3'
    ouvrier_Q4['ouvrier'] = 'Q4'

    # concaténation
    villes_ouvrier = pd.concat([ouvrier_Q1, ouvrier_Q2, ouvrier_Q3, ouvrier_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_ouvrier = villes_ouvrier[['LIBGEO', 'ouvrier','nom_REG', 'DEP']]
    


    # ##### Ouvrières

    # dispersion des taux horaires nets des ouvrières dans 4 catégories:
    # inférieur à 0.25 ; entre 0.25 et 0.5 ; entre 0.5 et 0.75 ; supérieur à 0.75

    # taux horaire net minimum chez les ouvrières (Q1 : inférieur à 0.25)
    Q1 = jeu_simple['ouvrière'].quantile(0.25)  
    ouvrière_Q1 = jeu_simple[jeu_simple['ouvrière'] < Q1]
    ouvrière_Q1 = ouvrière_Q1.drop_duplicates(subset='LIBGEO')
    ouvrière_Q1 = ouvrière_Q1[['nom_REG', 'DEP','LIBGEO', 'ouvrière', 'nb_tot_ent']]


    # Q1 - Q2 (Q2 : entre 0.25 et 0.5)
    # taux horaire net chez les ouvrières (Q1 _ Q2)
    Q2 = jeu_simple['ouvrière'].quantile(0.5)
    ouvrière_Q2 = jeu_simple[(jeu_simple['ouvrière'] >= jeu_simple['ouvrière'].quantile(0.25) )& (jeu_simple['ouvrière'] <= jeu_simple['ouvrière'].quantile(0.5))]
    ouvrière_Q2 = ouvrière_Q2.drop_duplicates(subset='LIBGEO')
    ouvrière_Q2 = ouvrière_Q2[['nom_REG', 'DEP','LIBGEO', 'ouvrière', 'nb_tot_ent']]

    # Q2 - Q3
    Q3 = jeu_simple['ouvrière'].quantile(0.75)  
    ouvrière_Q3 = jeu_simple[(jeu_simple['ouvrière'] > jeu_simple['ouvrière'].quantile(0.5) )& (jeu_simple['ouvrière'] <= jeu_simple['ouvrière'].quantile(0.75))]
    ouvrière_Q3 = ouvrière_Q3.drop_duplicates(subset='LIBGEO')
    ouvrière_Q3 = ouvrière_Q3[['nom_REG', 'DEP','LIBGEO', 'ouvrière', 'nb_tot_ent']]

    # Q4 (Q4 : supérieur à 0.75)
    # taux horaire net maximum chez les ouvrières
    Q4 = ouvrière_Q4 = jeu_simple[(jeu_simple['ouvrière'] > Q3 )]
    ouvrière_Q4 = ouvrière_Q4.drop_duplicates(subset='LIBGEO')
    ouvrière_Q4 = ouvrière_Q4[['nom_REG', 'DEP','LIBGEO', 'ouvrière', 'nb_tot_ent']]


    # tableau récapitulatif des taux moyens des ouvrières femmes
    data = {"Quartiles du salaire moyen des ouvrières femmes": ["Q1", "Q2", "Q3"],
            "Valeur (en €)": [Q1, Q2, Q3]}
    Q_ouvrière_F = pd.DataFrame(data).T
   


    # création de villes_ouvrière qui prend le nom de la ville et sa catégorie
    ouvrière_Q1['ouvrière'] = 'Q1'
    ouvrière_Q2['ouvrière'] = 'Q2'
    ouvrière_Q3['ouvrière'] = 'Q3'
    ouvrière_Q4['ouvrière'] = 'Q4'

    # concaténation
    villes_ouvrière = pd.concat([ouvrière_Q1, ouvrière_Q2, ouvrière_Q3, ouvrière_Q4], ignore_index=True)

    # nom de la ville et catégorie d'interet
    villes_ouvrière = villes_ouvrière[['LIBGEO', 'ouvrière','nom_REG', 'DEP']]
    


    # ##### Regrouper les tableaux 

    # villes communes à villes_cadr_h et villes_cadr_f
    villes_cadr = pd.merge(villes_cadr_h, villes_cadr_f, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")

    

    # villes communes à villes_cadr_moy_h et villes_cadr_moy_f
    villes_cadr_moy = pd.merge(villes_cadr_moy_h, villes_cadr_moy_f, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")

    


    # villes communes à villes_employé et villes_employée
    villes_employés = pd.merge(villes_employé, villes_employée, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")

    

    # villes communes à villes_ouvrier et villes_ouvrière
    villes_ouvriers = pd.merge(villes_ouvrier, villes_ouvrière, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")

    


    # ###### Il n'y a pas de villes où un sexe est présent et pas l'autre, selon la CSP

    # faire un tableau avec les villes uniques à villes_cadr_h ou villes_cadr_f
    merged_villes_cadr = pd.merge(villes_cadr_h, villes_cadr_f, on=["LIBGEO", 'DEP', 'nom_REG'], how="outer", indicator=True)
    unique_villes_cadr = merged_villes_cadr[merged_villes_cadr['_merge'].isin(['left_only', 'right_only'])]

    

    # faire un tableau avec les villes uniques à villes_cadr_moy_h ou villes_cadr_moy_f
    merged_villes_cadr_moy = pd.merge(villes_cadr_moy_h, villes_cadr_moy_f, on=["LIBGEO", 'DEP', 'nom_REG'], how="outer", indicator=True)
    unique_villes_cadr_moy = merged_villes_cadr_moy[merged_villes_cadr_moy['_merge'].isin(['left_only', 'right_only'])]

   


    # faire un tableau avec les villes uniques à villes_employé ou villes_employée
    merged_villes_employés = pd.merge(villes_employé, villes_employée, on=["LIBGEO", 'DEP', 'nom_REG'], how="outer", indicator=True)
    unique_villes_employés = merged_villes_employés[merged_villes_employés['_merge'].isin(['left_only', 'right_only'])]

    


    # faire un tableau avec les villes uniques à villes_ouvrier ou villes_ouvrières
    merged_villes = pd.merge(villes_ouvrier, villes_ouvrière, on=["LIBGEO", 'DEP', 'nom_REG'], how="outer", indicator=True)
    unique_villes_ouvrier = merged_villes[merged_villes['_merge'].isin(['left_only', 'right_only'])]

    

    # Fusion successive des tableaux
    villes_communes = pd.merge(villes_cadr, villes_cadr_moy, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")
    villes_communes = pd.merge(villes_communes, villes_employés, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")
    villes_communes = pd.merge(villes_communes, villes_ouvriers, on=["LIBGEO", 'DEP', 'nom_REG'], how="inner")

    # Réorganiser l'ordre des colonnes dans le DataFrame
    villes_communes = villes_communes[['LIBGEO', 'DEP', 'nom_REG', 'Cadre_h', 'Cadre_f'
                                  , 'cadr_moy_h', 'cadr_moy_f', 'employé', 'employée', 'ouvrier', 'ouvrière']]

    st.header("ACP et Machine learning")
    # Vérification de la longueur du tableau résultant
    st.write(f'Le tableau de répartiton des quartiles par CSP et sexe contient {len(villes_communes)} villes')
    st.write('Afin de réaliser ce tableau qui est à la base de la carte ci-dessous, nous avons procédé à une Analyse en composantes principales.')

    villes_communes

    # garder un double de villes_communes pour ne pas avoir à relancer tout le code si je veux 
    # récupérer villes_communes en l'état
    copie_villes_communes = villes_communes
    #villes_communes = copie_villes_communes


    # réorganiser les colonnes
    # Réorganiser l'ordre des colonnes dans le DataFrame
    villes_communes = villes_communes[['LIBGEO', 'DEP', 'nom_REG', 'Cadre_h', 'Cadre_f'
                                      , 'cadr_moy_h', 'cadr_moy_f', 'employé', 'employée', 'ouvrier', 'ouvrière']]


    #villes_communes


    # Voir s'il y a des doublons dans DEP : 0
    len(villes_communes[villes_communes['DEP'].isna()])


    # mettre les DEP en int
    villes_communes.DEP = villes_communes.DEP.astype(int)


    # remettre les numéros des régions 
    # changer float en int
    numeros_regions = {
        'Ile-de-France' :11,
        'Centre-Val de Loire' :24,
        'Bourgogne-Franche-Comté' :27,
        'Normandie' :28,
        'Hauts-de-France' : 32 ,
        'Grand Est' :44,
        'Pays de la Loire' :52,
        'Bretagne' :53,
        'Nouvelle-Aquitaine' :75,
        'Occitanie' :76,
        'Auvergne-Rhône-Alpes' :84,
        "PACA" :93,
        'Corse' : 94
    }

    villes_communes['REG'] = villes_communes['nom_REG'].map(numeros_regions)
    villes_communes['REG']= villes_communes['REG'].astype('int')


    # enlever nom_REG qui est de type str 
    villes_communes = villes_communes.drop(columns=['nom_REG'])


    
    stats = villes_communes[['Cadre_h', 'Cadre_f', 'cadr_moy_h', 'cadr_moy_f', 'employé', 'employée', 'ouvrier', 'ouvrière']].describe()
    stats.round(2)


    csp_num = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    colonnes = ['Cadre_h', 'Cadre_f', 'cadr_moy_h', 'cadr_moy_f', 'employé', 'employée', 'ouvrier', 'ouvrière']

    villes_communes[colonnes] = villes_communes[colonnes].replace(csp_num)


    CSP_DEP = villes_communes.groupby('DEP').agg({
        'Cadre_h': 'mean', 
        'Cadre_f': 'mean', 
        'cadr_moy_h': 'mean', 
        'cadr_moy_f': 'mean', 
        'employé': 'mean', 
        'employée': 'mean', 
        'ouvrier': 'mean', 
        'ouvrière': 'mean', 
        
    })


    CSP_DEP.info()


    CSP_DEP.head()


    CSP_DEP.reset_index(inplace=True)


    # voir dans quelles régions les différences entre le Q4 cadre homme et le Q1 ouvrière sont les + importantes
    cadr_h_max = jeu_simple[jeu_simple['cadr_h'] > Q3 ]
    ouvrière_min = jeu_simple[jeu_simple['ouvrière'] < Q1]

    cadr_h_Q4_reg = cadr_h_max.groupby('nom_REG')['cadr_h'].mean()
    ouvrière_Q1_reg = ouvrière_min.groupby('nom_REG')['ouvrière'].mean()

    # Calcul des écarts de salaires
    ecarts_salaires = ((cadr_h_Q4_reg - ouvrière_Q1_reg) / cadr_h_Q4_reg)*100

    # Création d'un nouveau DataFrame pour stocker les écarts de salaires par région
    ecarts_salaires_df = pd.DataFrame({'nom_REG': ecarts_salaires.index, 'Ecart salaire en %': ecarts_salaires.values})

    # Tri du DataFrame par la colonne 'Ecart_salaire' dans l'ordre décroissant
    ecarts_salaires_df = ecarts_salaires_df.sort_values(by='Ecart salaire en %', ascending=False)

    #st.title("Écarts régionaux des salaires entre un cadre homme de salaire Q4 et une ouvrière de salaire Q1")
    #st.dataframe(ecarts_salaires_df.round(2))


    ecarts_salaires_df = ecarts_salaires_df.sort_values(by='Ecart salaire en %', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(ecarts_salaires_df['Ecart salaire en %'], ecarts_salaires_df['nom_REG'], marker='o', linestyle='-')
    ax.set_title("Écarts de salaires entre Q4 cadre homme et Q1 ouvrière par région")
    ax.set_xlabel("Écart de salaire en pourcentage")
    ax.set_ylabel("Région")
    plt.grid(True)
    #st.pyplot()





    # Enlever la colonne DEP, qui fausse l'ACP
    CSP_DEP = CSP_DEP.drop(columns='DEP')

    # Mettre les variables 
    variables = ['Cadre_h', 'Cadre_f', 'cadr_moy_h', 'cadr_moy_f', 'employé', 'employée', 'ouvrier', 'ouvrière']

    CSP_DEP_acp = CSP_DEP[variables]

    # Standardisation des données
    scaler = StandardScaler()
    standar = scaler.fit_transform(CSP_DEP_acp)

    # Création de l'objet PCA
    acp = PCA()

    # Application de l'ACP sur les données standardisées
    resultats_acp = acp.fit_transform(standar)

    # Analyse des composantes principales
    variance_expliquee = acp.explained_variance_ratio_
    composantes_principales = acp.components_


    # Visualisation de la variance expliquée par chaque composante principale
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, len(variance_expliquee) + 1), variance_expliquee, marker='o', linestyle='-')
    ax.set_title('Variance Expliquée par Composante Principale')
    ax.set_xlabel('Composante Principale')
    ax.set_ylabel('Variance Expliquée')
    plt.grid()
    st.pyplot()
    st.write("La 1ère composante est très équilibrée, car comme le montre le heatmap, chacune des variables contribue à hauteur de 0.35 environ, ce qui signifie qu'il y a une relation positive modérée entre les variables et la première composante principale.")

    # Sélection des 2 premières composantes principales
    compo_principales = resultats_acp[:, :2]

    # DataFrame avec les coeff de corrélation sur les deux premiers axes
    coefficients_pca = pd.DataFrame({'Composante_Principale_1': acp.components_[0, :], 'Composante_Principale_2': acp.components_[1, :]})
    


    # Création d'un heatmap pour les coeff de corrélation
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(coefficients_pca, annot=True, cmap='viridis')
    ax.set_title('Coefficients de corrélation des variables d\'origine sur les deux premiers axes de l\'ACP')
    st.pyplot()


    # cercle
    # Calcul des corr entre les vars 
    racine_valeurs_propres = np.sqrt(acp.explained_variance_)
    corvar = np.zeros((8, 2))

    for k in range(2):
        corvar[:, k] = acp.components_[:, k] * racine_valeurs_propres[k]

    # Délimitation de la figure
    fig, axes = plt.subplots(figsize=(10, 10))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)

    # Affichage des variables
    for j in range(8):
        plt.annotate(variables[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
        plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.6, alpha=0.5, head_width=0.03, color='b')

    # Ajout des axes
    plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

    # Cercle et légendes
    cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
    axes.add_artist(cercle)
    ax.set_xlabel('AXE 1')
    ax.set_ylabel('AXE 2')
    st.pyplot(fig)
    


    # Normalisation des données avec Min-Max Scaler pour mettre les valeurs entre 0 et 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(CSP_DEP)

    # Méthode du coude pour déterminer le nombre de clusters
    range_n_clusters = [2, 3, 4, 5, 6]
    distorsions = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(CSP_DEP)
        distorsions.append(sum(np.min(cdist(CSP_DEP, kmeans.cluster_centers_, 'euclidean'), axis=1) / np.size(CSP_DEP, axis=0)))


    # Visualisation du coude
    st.write("Pour déterminer le nombre de cluster à utiliser, nous avons utilisé la méthode du coude. Cette méthode permet de voir à partir de combien de clusters la variance expliquée du modèle est optimale. Dans notre ca, nous devons utiliser 4 clusters.")
    fig = px.line(x=range_n_clusters, y=distorsions, labels={'x': 'Nombre de clusters', 'y': 'Distorsions'}, title='Nombre de clusters à utiliser')
    st.plotly_chart(fig)


    # Utilisation de K-Means avec 4 clusters 
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(CSP_DEP)


    # Centroids + labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_


    # Visualisation des données ACP avec clusters
    scatter = plt.scatter(compo_principales[:, 0], compo_principales[:, 1], c=labels, cmap='viridis')

    # Légende des clusters
    legend_labels = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"]
    # Création d'une figure avec Plotly Express
    fig = px.scatter(x=compo_principales[:, 0], y=compo_principales[:, 1], color=labels, labels={'x': 'Composante Principale 1', 'y': 'Composante Principale 2'}, color_discrete_sequence=px.colors.qualitative.Set1, title='Données ACP avec Clusters', template="plotly_white")
    # Ajout de la légende
    fig.update_layout(legend_title_text="Clusters", legend=dict(itemsizing="constant"))
    # Affichage de la figure avec Streamlit
    st.plotly_chart(fig)




    
    # voir les moyennes des CSP dans les culsters
    CSP_DEP['Cluster'] = labels
    cluster_means = CSP_DEP.groupby(['Cluster']).mean().reset_index()
    


    # relier les points aux departements
    departements = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18'
                     , '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'
                     , '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52'
                     , '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69'
                     , '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86'
                     , '87', '88', '89', '90', '91', '92', '93', '94', '95', '96']
     
    # stocker les départements et leur cluster
    dep_cluster = pd.DataFrame({'DEP': departements, 'Cluster': labels})

    # Trier le DataFrame par le numéro du département
    dep_cluster = dep_cluster.sort_values('DEP')

    # Réinitialiser les index
    dep_cluster.reset_index(drop=True, inplace=True)


    # Afficher toutes les lignes du DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    #dep_cluster.T


    # restaurer l'affichage par défaut 
    pd.reset_option('display.max_rows')


    # ##### Méthode t-SNE

    # modèle t-SNE avec les paramètres 
    tsne = TSNE(n_components=2, perplexity=80, n_iter=1000)

    # Normalisation
    tsne_results = tsne.fit_transform(scaled_data)

    department_labels = departements

    # résultats
    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dimension 1', 'Dimension 2'])


    # Ajouter la colonne 'Cluster' à tsne_df
    tsne_df['Cluster'] = labels  
    tsne_df['Department'] = department_labels
 
    st.write("Nous avons utilisé ensuite la méthode t-distributed Stochastic Neighbor Embedding (t-SNE) afin de rendre plus efficace la visualisation des données.")
    custom_palette = ["red", "green", "blue", "orange"]
    # Création d'une figure avec Plotly Express pour relier les points aux départements
    #fig = px.scatter(tsne_df, x='Dimension 1', y='Dimension 2', color='Cluster', hover_data=['Department'])
    fig = px.scatter(tsne_df, x='Dimension 1', y='Dimension 2', color='Cluster', hover_data=['Department'], color_discrete_sequence=custom_palette)
    # Mettre à jour le layout
    fig.update_layout(title='Points reliés aux départements avec t-SNE')
    st.plotly_chart(fig, theme=None)



    
    st.write("La carte est découpée en fonction de l'importance des taux horaires des départements. Cette carte montre les 4 niveaux de richesse du pays. Les départements les plus riches sont au nombre de 8, ce qui est assez faible. Les autres départements au-dessus de la moyenne nationale sont au nombre de 19. Au total, 27 départements peuvent être considérés comme riches.")
    st.write("Ensuite, 41 départements sont dans la moyenne basse et 19 peuvent êter considérés comme pauvres. On peut en conclure que la richesse est concentrée et que de larges pans, au niveau géographique, sont relégués dans la pauvreté.") 
    # Importer le fichier avec les départements
    gdf = gpd.read_file(r"C:\Users\Jean-Jacques\Desktop\Datascientest\french_industry\dossiers_de_travail\Documents externes\contour-des-departements.GeoJSon")

    # Renommer la colonne du fichier externe pour la jointure
    gdf = gdf.rename(columns={'code': 'DEP'})

    # Renommer les départements de la Corse pour qu'ils soient identiques à ceux du code
    gdf.DEP = gdf.DEP.replace({'2A': '20', '2B': '96'})

    # Fusionner avec les données de cluster
    gdf = gdf.merge(dep_cluster, on='DEP', how='left')

    # Carte des départements par couleur de cluster
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Utiliser une carte de couleur basée sur le cluster
    gdf.plot(column='Cluster', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Create the legend
    leg = ax.legend(loc=1)

    # Set the legend position using the bbox_to_anchor parameter
    leg.set_bbox_to_anchor((1, 0.5))

    # Add a title to the legend
    leg.set_title('Cluster')

    # Set the title of the plot
    ax.set_title('Carte des départements français par cluster', fontdict={'fontsize': '15', 'fontweight': '3'})

    # Display the plot
    st.pyplot(fig)

    # Comptage du nombre de départements dans chaque cluster
    cluster_counts = dep_cluster['Cluster'].value_counts()

    # Affichage du nombre de départements par cluster
    st.write("Nombre de départements par cluster :")
    st.write(cluster_counts)


if section_6 :
    

    st.header('Machine learning')
    st.write("Nous avons testé 4 modèles : la régression linéaire, l'arbre à décision, le randon forest et le xgboost, dans 2 conditions différentes.")
    st.write("La 1ère n'a pris que des variables ne nécessitant pas de OneHotEncoder, à la différence de la seconde. Ici, 3 variables ont été ajouté : 2 variables géographiques (les départments et les régions) et le sexe.")
    st.header('Régression linéaire')
    # enlever les colonnes concernant les salaires moyens déjà calculés + les autres colonnes inutiles
    colonnes_a_supprimer = ['LIBGEO', 'sal_moyen_cadre', 'sal_moyen_cadre_moy', 'sal_moyen_employés', 'sal_moyen_ouvriers', 'sal_moy_f', 'sal_moy_h', 'nom_REG', 'nom_département', 'latitude', 'longitude', 'x', 'y','SEXE', 'DEP', 'REG']
    jeu2 = jeu.copy()
    jeu = jeu.drop(columns=colonnes_a_supprimer, errors='ignore')
    jeu_RL = jeu
    # séparer le jeu en 2 et isoler la variable cible
    feats_RL = jeu_RL.drop("sal_moyen_tot", axis=1)
    target_RL = jeu_RL['sal_moyen_tot']


    # séparer les jeux d'entrainement et de test
    X_train_RL, X_test_RL, y_train_RL, y_test_RL = train_test_split(feats_RL, target_RL, test_size=0.25, random_state=42)
    X_train_RL = X_train_RL.apply(pd.to_numeric, errors='coerce').dropna()


    # Réinitialiser les indices
    X_train_RL.reset_index(drop=True, inplace=True)


    # Créer une instance du StandardScaler 
    sc = StandardScaler()

    # Appliquer la transformation de mise à l'échelle aux données
    X_train_scaled_RL = sc.fit_transform(X_train_RL)
    X_test_scaled_RL = sc.transform(X_test_RL)


    # Instancier un modèle LinearRegression pour l'entrainement.
    regressor = LinearRegression()
    regressor.fit(X_train_scaled_RL, y_train_RL)


    # Voir les coefficients de régression
    feat_importances_RL = pd.DataFrame({'Coefficient': regressor.coef_}, index=feats_RL.columns)


    #L'intercept représente la valeur moyenne de la variable cible lorsque toutes les variables prédictives du modèle sont 
    # égales à zéro. Donc si toutes les variables explicatives sont à zéro, le modèle prédira le salaire moyen à 13.74


    st.write('Coefficient de détermination du modèle sur train (entraînement):', regressor.score(X_train_scaled_RL, y_train_RL))
    st.write('Coefficient de détermination du modèle sur test (projection):', regressor.score(X_test_scaled_RL, y_test_RL))

    # obtenir les prévisions du modèle
    predictions = regressor.predict(X_test_scaled_RL)

    # Calculer le salaire moyen prédit
    sal_moyen_pred_RL = np.mean(predictions)

    # Afficher le résultat
    st.write("Salaire moyen prédit :", sal_moyen_pred_RL)

    # Graphique de la régression linéaire
    pred_test_RL = regressor.predict(X_test_scaled_RL)

    fig, ax = plt.subplots()  # Crée une nouvelle figure et un nouvel axe

    # Scatter plot
    ax.scatter(pred_test_RL, y_test_RL, c='green')

    # Ligne de régression
    ax.plot((y_test_RL.min(), y_test_RL.max()), (y_test_RL.min(), y_test_RL.max()), color='red')

    # Étiquettes et titre
    ax.set_xlabel("prédiction")
    ax.set_ylabel("vraie valeur")
    ax.set_title('Régression Linéaire pour la prédiction du salaire moyen')

    # Affiche la figure
    st.pyplot(fig)


    # Obtenir les coefficients du modèle
    coefficients_RL = regressor.coef_

    # Créer un DataFrame pour visualiser les coefficients
    feat_importances_RL = pd.DataFrame(coefficients_RL, index=feats_RL.columns, columns=["Coefficient"])

    # Trier les valeurs par ordre décroissant
    feat_importances_RL = feat_importances_RL.abs().sort_values(by='Coefficient', ascending=False)

    # Afficher le graphique
    feat_importances_RL.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables dans la régression linéaire')
    st.pyplot() ;



    # Arbre à décision
    jeu_AD = jeu

    # Séparer le jeu de données
    feats_AD = jeu_AD.drop("sal_moyen_tot", axis=1)
    target_AD = jeu_AD['sal_moyen_tot']

    # Charger les données
    num = ['NB', 'cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h',
           'employé', 'ouvrier', 'Micro', 'Petite', 'Moyenne', 'Grande', 'nb_tot_ent']

    # Séparer les données
    X_train_AD, X_test_AD, y_train_AD, y_test_AD = train_test_split(feats_AD, target_AD, test_size=0.25, random_state=42)

    # Utiliser des données non mises à l'échelle pour voir les taux horaires en €
    regressor_tree_unscaled = DecisionTreeRegressor(random_state=42, max_depth=3)

    # Copier les données non mises à l'échelle
    X_train_unscaled_AD = X_train_AD.copy()

    X_train_unscaled_reduce_AD = X_train_unscaled_AD[['cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h', 'employé'
                                                , 'ouvrier' ,'Micro', 'Petite', 'Moyenne', 'Grande', 'nb_tot_ent']]

    # Entraînement du modèle non mis à l'échelle
    regressor_tree_unscaled.fit(X_train_unscaled_reduce_AD, y_train_AD)

    # Titre
    st.title("Visualisation de l'arbre de décision non mis à l'échelle")

    # Créer une figure pour les deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 30))

    # Graphique de l'arbre de décision
    plot_tree(regressor_tree_unscaled, feature_names=['cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h'
                                                       , 'employé' , 'ouvrier', 'Micro','Petite', 'Moyenne', 'Grande', 'nb_tot_ent']
              , filled=True, rounded=True, ax=ax1)

    # Voir l'importance des fonctionnalités
    feature_importance_tree = regressor_tree_unscaled.feature_importances_

    # DataFrame pour visualiser l'importance des fonctionnalités
    feat_importances_tree = pd.DataFrame({'Importance': feature_importance_tree}, index=X_train_unscaled_reduce_AD.columns)

    # Trier les valeurs par ordre décroissant
    feat_importances_tree = feat_importances_tree.sort_values(by='Importance', ascending=False)

    # Graphique d'importance des variables
    feat_importances_tree.plot(kind='bar', rot=45, ax=ax2)
    ax2.set_title("Importance des variables dans l'arbre de décision")

    # Afficher la figure avec les deux sous-graphiques
    st.pyplot(fig)




    # ##### Random forest
    st.header('Random forest')
    jeu_RF = jeu
    # séparer le jeu en 2 et isoler la variable cible
    feats_RF = jeu_RF.drop("sal_moyen_tot", axis=1)
    target_RF = jeu_RF['sal_moyen_tot']

    # charger les données : num = ['NB',  'cadr_f', ...]

    # Séparer les données
    X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(feats_RF, target_RF, test_size=0.25, random_state=42)


    # Normaliser les données
    sc = StandardScaler()
    X_train_RF[num] = sc.fit_transform(X_train_RF[num])
    X_test_RF[num] = sc.transform(X_test_RF[num])

    # Initialiser le modèle de forêt aléatoire
    regressor_RF = RandomForestRegressor(random_state=42)

    # Entraîner le modèle sur l'ensemble d'entraînement
    regressor_RF.fit(X_train_RF, y_train_RF)

    # voir les résultats
    st.write("Coefficient de détermination du modèle sur train (entraînement): ", regressor_RF.score(X_train_RF,y_train_RF))
    st.write("Coefficient de détermination du modèle sur test (projection): ",regressor_RF.score(X_test_RF,y_test_RF))

    # obtenir les prévisions du modèle
    predictions = regressor.predict(X_test_RF)

    # Calculer le salaire moyen
    sal_moyen_pred = np.mean(predictions)

    # Afficher le résultat
    st.write("Salaire moyen prédit :", sal_moyen_pred)



    # Voir l'importance des caractéristiques du modèle
    importances_RF = regressor_RF.feature_importances_

    # Créer un DataFrame pour visualiser l'importance des caractéristiques
    feat_importances_RF = pd.DataFrame(importances_RF, index=['NB','cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h'
                                                        , 'Micro', 'Petite', 'Moyenne', 'Grande', 'nb_tot_ent'
                                                        , 'employé', 'ouvrier']  , columns=["Importance"])

    # Trier les valeurs par ordre décroissant
    feat_importances_RF = feat_importances_RF.sort_values(by='Importance', ascending=False)


    # Le graphique
    feat_importances_RF.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables dans le Random Forest')
    st.pyplot()


    # ##### xgboostregressor

    st.header('XGBoost')
    jeu_xg = jeu
    
    # séparer le jeu en 2 et isoler la variable cible
    feats_xg = jeu_xg.drop("sal_moyen_tot", axis=1)
    target_xg = jeu_xg['sal_moyen_tot']

    # charger les données : num = ['NB',  'cadr_f', ...]

    # Séparer les données
    X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(feats_xg, target_xg, test_size=0.25, random_state=42)

    # Normaliser les données
    sc = StandardScaler()
    X_train_xg[num] = sc.fit_transform(X_train_xg[num])
    X_test_xg[num] = sc.transform(X_test_xg[num])

    # xgboostregressor
    xgb_regressor = XGBRegressor(random_state=42)

    # Entraîner le modèle sur l'ensemble d'entraînement
    xgb_regressor.fit(X_train_xg, y_train_xg)

    # Voir les résultats
    st.write("Score sur l'ensemble d'entraînement:", xgb_regressor.score(X_train_xg, y_train_xg))
    st.write("Score sur l'ensemble de test:", xgb_regressor.score(X_test_xg, y_test_xg))

    # Voir l'importance des caractéristiques du modèle
    importances_xg = xgb_regressor.feature_importances_

    # Créer un DataFrame pour visualiser l'importance des caractéristiques
    feat_importances_xg = pd.DataFrame(importances_xg, index=['NB', 'cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h'
                                                               , 'cadr_moy_h', 'Micro', 'Petite', 'Moyenne', 'Grande'
                                                               , 'nb_tot_ent',  'employé', 'ouvrier'], columns=["Importance"])

    # Trier les valeurs par ordre décroissant
    feat_importances_xg = feat_importances_xg.sort_values(by='Importance', ascending=False)

    # Afficher le graphique
    fig, ax = plt.subplots()
    feat_importances_xg.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables avec XGBoostregressor')
    st.pyplot() ;




    # ### Tests de prédiction avec OneHotEncoder pour SEXE, REG et DEP
    jeu = jeu2
    colonnes_a_supprimer = ['LIBGEO', 'sal_moyen_cadre', 'sal_moyen_cadre_moy', 'sal_moyen_employés', 'sal_moyen_ouvriers', 'sal_moy_f', 'sal_moy_h', 'nom_REG', 'nom_département', 'latitude', 'longitude', 'x', 'y']

    jeu = jeu.drop(columns=colonnes_a_supprimer, errors='ignore')
    # ##### Régression linéaire avec OHE
    st.header('Régression linéaire avec OHE')

    # Supprimer les colonnes du DataFrame X_train_RLo
    feats_RLo = jeu.drop('sal_moyen_tot', axis=1)
    target_RLo = jeu['sal_moyen_tot']


    # Variables non affectées par OHE
    num = ['NB', 'cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h', 'employé', 'ouvrier',
           'Micro', 'Petite', 'Moyenne', 'Grande', 'nb_tot_ent']


    # séparer les jeux d'entrainement et de test
    X_train_RLo, X_test_RLo, y_train_RLo, y_test_RLo = train_test_split(feats_RLo, target_RLo, test_size=0.25, random_state=42)



    # Encodage one-hot des variables catégorielles
    oneh = OneHotEncoder(drop='first', sparse_output=False)


    # Séparer les variables qui vont subir un OHE des autres
    cat = ['SEXE', 'REG', 'DEP']


    # Encodage one-hot des variables catégorielles pour l'ensemble d'entraînement
    X_train_encoded_RLo = pd.DataFrame(oneh.fit_transform(X_train_RLo[cat]), columns=oneh.get_feature_names_out(X_train_RLo[cat].columns), index=X_train_RLo.index)
    X_train_RLo = pd.concat([X_train_RLo.drop(columns=cat), X_train_encoded_RLo], axis=1)

    # Encodage one-hot des variables catégorielles pour l'ensemble de test
    X_test_encoded_RLo = pd.DataFrame(oneh.transform(X_test_RLo[cat]), columns=oneh.get_feature_names_out(X_test_RLo[cat].columns), index=X_test_RLo.index)
    X_test_RLo = pd.concat([X_test_RLo.drop(columns=cat), X_test_encoded_RLo], axis=1)


    # regrouper les variables qui ne font pas l'objet d'un OneHotEncoder
    num_feats = X_train_RLo.columns.intersection(num)


    # normaliser les données
    sc = StandardScaler()

    X_train_RLo.loc[:,num] = sc.fit_transform(X_train_RLo[num])
    X_test_RLo.loc[:,num] = sc.transform(X_test_RLo[num])


    # Créer une instance du LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train_RLo, y_train_RLo)


    # Obtenir les coefficients du modèle
    coeffs = list(regressor.coef_)
    coeffs.insert(0, regressor.intercept_)

    # Prendre le nombre total de caractéristiques dans X_train après l'encodage one-hot
    num_features = X_train_RLo.shape[1]
    feats2 = list(X_train_RLo.columns)
    feats2.insert(0, 'intercept')

    pd.DataFrame({'valeur estimée': coeffs[:num_features]}, index=feats2[:num_features])


    # Créer une instance de la régression Ridge pour résoudre le problème de la multicollarité des données d'entraînement 
    # (les DEP et les Régions)
    regressor_ridge = Ridge(alpha=0.0001)  

    # Adapter le modèle sur les données d'entraînement
    regressor_ridge.fit(X_train_RLo, y_train_RLo)

    # Obtenir les coefficients du modèle
    coeffs_ridge = list(regressor_ridge.coef_)
    coeffs_ridge.insert(0, regressor_ridge.intercept_)

    # Créer un DataFrame pour afficher les résultats
    pd.DataFrame({'valeur estimée (Ridge)': coeffs_ridge[:num_features]}, index=feats2[:num_features])


    # faire le graphique
    fig = fig, ax = plt.subplots(figsize = (10,10))

    pred_test = regressor.predict(X_test_RLo)

    plt.scatter(pred_test, y_test_RLo, c='green')

    plt.plot((y_test_RLo.min(), y_test_RLo.max()), (y_test_RLo.min(), y_test_RLo.max()), color = 'red')
    ax.set_xlabel("prédiction")
    ax.set_ylabel("vraie valeur")
    ax.set_title('Prédiction avec la régression linéaire ')

    st.pyplot() ;


    st.write('Coefficient de détermination du modèle de la régression linéaire sur train:', regressor.score(X_train_RLo, y_train_RLo))
    st.write('Coefficient de détermination du modèle de la régression linéaire sur test:', regressor.score(X_test_RLo, y_test_RLo))


    # Obtenir les coefficients du modèle
    coefficients = regressor.coef_

    # Créer un DataFrame pour visualiser les coefficients ; flatten sert à convertir les coefficients en une liste
    feat_importances = pd.DataFrame({'Coefficient': coefficients.flatten()}, index=X_train_RLo.columns)

    # Trier les valeurs par ordre décroissant
    feat_importances = feat_importances.sort_values(by='Coefficient', ascending=False).head(20)

    # Afficher le graphique
    feat_importances.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables dans la régression linéaire')
    st.pyplot()


    # #### Arbre à décision
    st.header('Arbre à décision avec OHE')
    # Séparer le jeu de données
    feats_ADo = jeu.drop('sal_moyen_tot', axis=1)
    target_ADo = jeu['sal_moyen_tot']


    # train test split
    X_train_ADo, X_test_ADo, y_train_ADo, y_test_ADo = train_test_split(feats_ADo, target_ADo, test_size=0.25, random_state=42)


    # Encodage one-hot des variables 
    oneh = OneHotEncoder(drop='first', sparse_output=False)


    # Séparation des variables à encoder des autres
    cat = ['SEXE', 'REG', 'DEP']


    # Encodage one-hot des variables catégorielles pour l'ensemble d'entraînement
    X_train_encoded_ADo = pd.DataFrame(oneh.fit_transform(X_train_ADo[cat]), columns=oneh.get_feature_names_out(cat), index=X_train_ADo.index)
    X_train_ADo = pd.concat([X_train_ADo.drop(columns=cat), X_train_encoded_ADo], axis=1)

    # Encodage one-hot des variables catégorielles pour l'ensemble de test
    X_test_encoded_ADo = pd.DataFrame(oneh.transform(X_test_ADo[cat]), columns=oneh.get_feature_names_out(cat), index=X_test_ADo.index)
    X_test_ADo = pd.concat([X_test_ADo.drop(columns=cat), X_test_encoded_ADo], axis=1)


    # standardisation des données
    sc = StandardScaler()


    # Création d'une instance de DecisionTreeRegressor (remplace RandomForestRegressor)
    tree_regressor = DecisionTreeRegressor(random_state=42)


    # Entraînement du modèle 
    tree_regressor.fit(X_train_ADo, y_train_ADo)


    # Évaluation du modèle 
    train_score_ADo = tree_regressor.score(X_train_ADo, y_train_ADo)
    test_score_ADo = tree_regressor.score(X_test_ADo, y_test_ADo)


    # Affichage des performances du modèle
    st.write("Coefficient de détermination du modèle de l'arbre à décision sur train: ", train_score_ADo)
    st.write("Coefficient de détermination du modèle de l'arbre à décision sur test: ", test_score_ADo)


    # Prédiction sur l'ensemble de test et graphique
    pred_test_tree = tree_regressor.predict(X_test_ADo)

    fig = fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(pred_test_tree, y_test_ADo, c='blue')
    plt.plot((y_test_ADo.min(), y_test_ADo.max()), (y_test_ADo.min(), y_test_ADo.max()), color='red')
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie valeur")
    ax.set_title('Modèle prédictif de l\'arbre de décision')
    st.pyplot();


    # Graphique de l'importance des variables pour un arbre de décision
    importances_tree = tree_regressor.feature_importances_
    feat_importances_tree = pd.DataFrame({'Importance': importances_tree}, index=X_train_ADo.columns)
    feat_importances_tree = feat_importances_tree.sort_values(by='Importance', ascending=False).head(20)

    feat_importances_tree.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables dans le modèle de l\'arbre de décision')
    st.pyplot() ;


    # ##### Random forest avec OHE
    st.header('Random forest avec OHE')
    # Suppression des colonnes inutiles

    # Séparer le jeu de données
    feats_RFo = jeu.drop('sal_moyen_tot', axis=1)
    target_RFo = jeu['sal_moyen_tot']


    # train test split
    X_train_RFo, X_test_RFo, y_train_RFo, y_test_RFo = train_test_split(feats_RFo, target_RFo, test_size=0.25, random_state=42)


    # Encodage one-hot des variables 
    oneh = OneHotEncoder(drop='first', sparse_output=False)


    # Séparation des variables à encoder des autres
    cat = ['SEXE', 'REG', 'DEP']


    # Encodage one-hot des variables catégorielles pour l'ensemble d'entraînement
    X_train_encoded_RFo = pd.DataFrame(oneh.fit_transform(X_train_RFo[cat]), columns=oneh.get_feature_names_out(cat), index=X_train_RFo.index)
    X_train_RFo = pd.concat([X_train_RFo.drop(columns=cat), X_train_encoded_RFo], axis=1)

    # Encodage one-hot des variables catégorielles pour l'ensemble de test
    X_test_encoded_RFo = pd.DataFrame(oneh.transform(X_test_RFo[cat]), columns=oneh.get_feature_names_out(cat), index=X_test_RFo.index)
    X_test_RFo = pd.concat([X_test_RFo.drop(columns=cat), X_test_encoded_RFo], axis=1)


    # Création d'une instance de RandomForestRegressor
    rfo_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


    # Regrouper les variables qui ne font pas l'objet d'un OneHotEncoder
    num_feats = X_train_RFo.columns.intersection(num)

    # Normaliser les données
    X_train_RFo.loc[:, num] = sc.fit_transform(X_train_RFo[num])
    X_test_RFo.loc[:, num] = sc.transform(X_test_RFo[num])


    # Entraînement du modèle 
    rfo_regressor.fit(X_train_RFo, y_train_RFo)


    # Évaluation du modèle 
    train_score_RFo = rfo_regressor.score(X_train_RFo, y_train_RFo)
    test_score_RFo = rfo_regressor.score(X_test_RFo, y_test_RFo)


    # Affichage des performances du modèle
    st.write('Coefficient de détermination du modèle sur train:', train_score_RFo)
    st.write('Coefficient de détermination du modèle sur test:', test_score_RFo)


    # Prédiction sur l'ensemble de test et graphique
    pred_test_rfo = rfo_regressor.predict(X_test_RFo)
     
    plt.scatter(pred_test_rfo, y_test_RFo, c='blue')
    plt.plot((y_test_RFo.min(), y_test_RFo.max()), (y_test_RFo.min(), y_test_RFo.max()), color='red')
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vraie valeur")
    ax.set_title('Modèle prédictif du Random forest avec OHE')
    st.pyplot()

    # Calculer le salaire moyen
    sal_moyen_pred = np.mean(pred_test_rfo)

    # Afficher le résultat
    st.write("Salaire moyen prédit :", sal_moyen_pred)


    # Graphique de l'importance des variables
    importances = rfo_regressor.feature_importances_
    feat_importances_rfo = pd.DataFrame({'Importance': importances}, index=X_train_RFo.columns)
    feat_importances_rfo = feat_importances_rfo.sort_values(by='Importance', ascending=False).head(20)
    feat_importances_rfo.plot(kind='bar', figsize=(10, 6), rot=45)
    ax.set_title('Importance des variables dans le modèle du Randon forest')
    st.pyplot() 


    # ##### xgboostregressor avec OHE
    # Suppression des colonnes inutiles
    st.header('XGBoost avec OHE')
    # Séparation des données
    feats_xgo = jeu.drop('sal_moyen_tot', axis=1)
    target_xgo = jeu['sal_moyen_tot']


    # Division du jeu de données en ensembles d'entraînement et de test
    X_train_xgo, X_test_xgo, y_train_xgo, y_test_xgo = train_test_split(feats_xgo, target_xgo, test_size=0.25, random_state=42)


    # Encodage one-hot des variables catégorielles
    oneh = OneHotEncoder(drop='first', sparse_output=False)


    # Séparation des variables à encoder des autres
    cat = ['SEXE', 'REG', 'DEP']
    num = ['NB', 'cadr_f', 'cadr_moy_f', 'employée', 'ouvrière', 'cadr_h', 'cadr_moy_h', 'employé', 'ouvrier',
           'Micro', 'Petite', 'Moyenne', 'Grande', 'nb_tot_ent']


    # Encodage one-hot des variables catégorielles pour l'ensemble d'entraînement
    X_train_encoded_xgo = pd.DataFrame(oneh.fit_transform(X_train_xgo[cat]), columns=oneh.get_feature_names_out(cat), index=X_train_xgo.index)
    X_train_xgo = pd.concat([X_train_xgo.drop(columns=cat), X_train_encoded_xgo], axis=1)


    # Encodage one-hot des variables catégorielles pour l'ensemble de test
    X_test_encoded_xgo = pd.DataFrame(oneh.transform(X_test_xgo[cat]), columns=oneh.get_feature_names_out(cat), index=X_test_xgo.index)
    X_test_xgo = pd.concat([X_test_xgo.drop(columns=cat), X_test_encoded_xgo], axis=1)


    # Créer une instance du StandardScaler
    sc = StandardScaler()


    # Regrouper les variables qui ne font pas l'objet d'un OneHotEncoder
    num_feats = X_train_xgo.columns.intersection(num)


    # Normaliser les données
    X_train_xgo.loc[:, num] = sc.fit_transform(X_train_xgo[num])
    X_test_xgo.loc[:, num] = sc.transform(X_test_xgo[num])


    # Créer une instance du XGBRegressor
    xgb_regressor_o = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)  
    xgb_regressor_o.fit(X_train_xgo, y_train_xgo)


    # Voir les résultats
    st.write("Score sur l'ensemble d'entraînement:", xgb_regressor_o.score(X_train_xgo, y_train_xgo))
    st.write("Score sur l'ensemble de test:", xgb_regressor_o.score(X_test_xgo, y_test_xgo))


    # Obtenir les coefficients du modèle
    coeffs_xgb_o = list(xgb_regressor_o.feature_importances_)


    # Créer un DataFrame pour afficher les résultats
    pd.DataFrame({'Importance': coeffs_xgb_o}, index=X_train_xgo.columns).sort_values(by='Importance', ascending=False).head(20)


    # Graphique des variables importantes
    importance_df = pd.DataFrame({'Importance': coeffs_xgb_o}, index=X_train_xgo.columns).sort_values(
        by='Importance', ascending=False).head(20).head(16)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.plot(kind='bar', legend=False, rot=45)
    ax.set_title('Importance des variables dans le modèle XGBoostRegressor')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Importance')
    st.pyplot() 



    # #### Voir les métriques des 4 modèles + un tableau de résultats
    st.header('Comparaison des résultats avec OneHotEncoder et sans')

    st.write('tableau récapitulatif des résultats sans OneHotEncoder')
    # #### Voir les métriques des 4 modèles + un tableau de résultats

    # Régression linéaire
    regressor_linear = LinearRegression()
    regressor_linear.fit(X_train_scaled_RL, y_train_RL)

    # Calcul des prédictions
    y_pred_linear_train_RL = regressor_linear.predict(X_train_scaled_RL)
    y_pred_linear_test_RL = regressor_linear.predict(X_test_scaled_RL)

    # Calcul des métriques
    # jeu d'entraînement 
    mae_linear_train = mean_absolute_error(y_train_RL, y_pred_linear_train_RL)
    mse_linear_train = mean_squared_error(y_train_RL, y_pred_linear_train_RL, squared=True)
    rmse_linear_train = mean_squared_error(y_train_RL, y_pred_linear_train_RL, squared=False)

    # jeu de test 
    mae_linear_test = mean_absolute_error(y_test_RL, y_pred_linear_test_RL)
    mse_linear_test = mean_squared_error(y_test_RL, y_pred_linear_test_RL, squared=True)
    rmse_linear_test = mean_squared_error(y_test_RL, y_pred_linear_test_RL, squared=False)


    ### Arbre à décision
    regressor_decision_tree = DecisionTreeRegressor(random_state=42) 
    regressor_decision_tree.fit(X_train_AD, y_train_AD)

    # Calcul des métriques
    y_pred_decision_tree = regressor_decision_tree.predict(X_test_AD)
    y_pred_train_decision_tree = regressor_decision_tree.predict(X_train_AD)


    # jeu d'entraînement 
    mae_decision_tree_train = mean_absolute_error(y_train_AD, y_pred_train_decision_tree)
    mse_decision_tree_train = mean_squared_error(y_train_AD, y_pred_train_decision_tree, squared=True)
    rmse_decision_tree_train = mean_squared_error(y_train_AD, y_pred_train_decision_tree, squared=False)

    # jeu de test 
    mae_decision_tree_test = mean_absolute_error(y_test_AD, y_pred_decision_tree)
    mse_decision_tree_test = mean_squared_error(y_test_AD, y_pred_decision_tree, squared=True)
    rmse_decision_tree_test = mean_squared_error(y_test_AD, y_pred_decision_tree, squared=False)


    ### RandomForest
    regressor_random_forest = RandomForestRegressor(random_state=42) 
    regressor_random_forest.fit(X_train_AD, y_train_AD)

    # Calcul des métriques 
    y_pred_random_forest = regressor_random_forest.predict(X_test_AD)
    y_pred_random_forest_train = regressor_random_forest.predict(X_train_AD)

    # jeu d'entraînement 
    mae_random_forest_train = mean_absolute_error(y_train_AD, y_pred_random_forest_train)
    mse_random_forest_train = mean_squared_error(y_train_AD, y_pred_random_forest_train, squared=True)
    rmse_random_forest_train = mean_squared_error(y_train_AD, y_pred_random_forest_train, squared=False)

    # jeu de test 
    mae_random_forest_test = mean_absolute_error(y_test_AD, y_pred_random_forest)
    mse_random_forest_test = mean_squared_error(y_test_AD, y_pred_random_forest, squared=True)
    rmse_random_forest_test = mean_squared_error(y_test_AD, y_pred_random_forest, squared=False)


    ### XGBoost
    xgb_regressor = XGBRegressor(random_state=42)
    xgb_regressor.fit(X_train_xg, y_train_xg)

    # Calcul des métriques
    # Calcul des métriques 
    y_pred_xgb = xgb_regressor.predict(X_test_xg)
    y_pred_xgb_train = xgb_regressor.predict(X_train_xg)

    # jeu d'entraînement
    mae_xgb_train = mean_absolute_error(y_train_xg, y_pred_xgb_train)
    mse_xgb_train = mean_squared_error(y_train_xg, y_pred_xgb_train, squared=True)
    rmse_xgb_train = mean_squared_error(y_train_xg, y_pred_xgb_train, squared=False)

    # jeu de test
    mae_xgb_test = mean_absolute_error(y_test_xg, y_pred_xgb)
    mse_xgb_test = mean_squared_error(y_test_xg, y_pred_xgb, squared=True)
    rmse_xgb_test = mean_squared_error(y_test_xg, y_pred_xgb, squared=False)


    # Creation d'un dataframe pour comparer les metriques des 4 algorithmes 
    data = {'MAE train': [mae_linear_train, mae_decision_tree_train, mae_random_forest_train, mae_xgb_train],
            'MAE test': [mae_linear_test, mae_decision_tree_test, mae_random_forest_test, mae_xgb_test],
            'MSE train': [mse_linear_train, mse_decision_tree_train, mse_random_forest_train, mse_xgb_train],
            'MSE test': [mse_linear_test, mse_decision_tree_test, mse_random_forest_test, mse_xgb_test],
            'RMSE train': [rmse_linear_train, rmse_decision_tree_train, rmse_random_forest_train, rmse_xgb_train],
            'RMSE test': [rmse_linear_test, rmse_decision_tree_test, rmse_random_forest_test, rmse_xgb_test]}

    # Creer le dataFrame
    df = pd.DataFrame(data, index=['Régression linéaire', 'Arbre à décision', 'Random forest', 'XGBoostregressor'])

    st.write(df)


    st.write('tableau récapitulatif des résultats avec OneHotEncoder')
    # Régression linéaire
    # Régression linéaire avec Ridge
    regressor_ridge = Ridge(alpha=0.0001)
    regressor_ridge.fit(X_train_RLo, y_train_RLo)

    # Prédictions
    y_pred_train_ridge = regressor_ridge.predict(X_train_RLo)
    y_pred_test_ridge = regressor_ridge.predict(X_test_RLo)

    # Métriques
    mae_ridge_train = mean_absolute_error(y_train_RLo, y_pred_train_ridge)
    mse_ridge_train = mean_squared_error(y_train_RLo, y_pred_train_ridge)
    rmse_ridge_train = np.sqrt(mse_ridge_train)

    mae_ridge_test = mean_absolute_error(y_test_RLo, y_pred_test_ridge)
    mse_ridge_test = mean_squared_error(y_test_RLo, y_pred_test_ridge)
    rmse_ridge_test = np.sqrt(mse_ridge_test)


    # Arbre à décision
    regressor_decision_tree = DecisionTreeRegressor(random_state=42) 
    regressor_decision_tree.fit(X_train_ADo, y_train_ADo)

    # Calcul des prédictions
    y_pred_decision_tree_train = regressor_decision_tree.predict(X_train_ADo)
    y_pred_decision_tree_test = regressor_decision_tree.predict(X_test_ADo)

    # Calcul des métriques
    mae_decision_tree_train2 = mean_absolute_error(y_train_ADo, y_pred_decision_tree_train)
    mse_decision_tree_train2 = mean_squared_error(y_train_ADo, y_pred_decision_tree_train, squared=True)
    rmse_decision_tree_train2 = mean_squared_error(y_train_ADo, y_pred_decision_tree_train, squared=False)

    mae_decision_tree_test2 = mean_absolute_error(y_test_ADo, y_pred_decision_tree_test)
    mse_decision_tree_test2 = mean_squared_error(y_test_ADo, y_pred_decision_tree_test, squared=True)
    rmse_decision_tree_test2 = mean_squared_error(y_test_ADo, y_pred_decision_tree_test, squared=False)


    # Random Forest
    regressor_random_forest2 = RandomForestRegressor(random_state=42) 
    regressor_random_forest2.fit(X_train_RFo, y_train_RFo)

    # Calcul des prédictions
    y_pred_random_forest_train2 = regressor_random_forest2.predict(X_train_RFo)
    y_pred_random_forest_test2 = regressor_random_forest2.predict(X_test_RFo)

    # Calcul des métriques
    mae_random_forest_train2 = mean_absolute_error(y_train_RFo, y_pred_random_forest_train2)
    mse_random_forest_train2 = mean_squared_error(y_train_RFo, y_pred_random_forest_train2, squared=True)
    rmse_random_forest_train2 = mean_squared_error(y_train_RFo, y_pred_random_forest_train2, squared=False)

    mae_random_forest_test2 = mean_absolute_error(y_test_RFo, y_pred_random_forest_test2)
    mse_random_forest_test2 = mean_squared_error(y_test_RFo, y_pred_random_forest_test2, squared=True)
    rmse_random_forest_test2 = mean_squared_error(y_test_RFo, y_pred_random_forest_test2, squared=False)


    # XGBoostRegressor
    xgb_regressor2 = XGBRegressor(random_state=42)
    xgb_regressor2.fit(X_train_xgo, y_train_xgo)

    # Calcul des prédictions
    y_pred_xgb_train2 = xgb_regressor2.predict(X_train_xgo)
    y_pred_xgb_test2 = xgb_regressor2.predict(X_test_xgo)

    # Calcul des métriques
    mae_xgb_train2 = mean_absolute_error(y_train_xgo, y_pred_xgb_train2)
    mse_xgb_train2 = mean_squared_error(y_train_xgo, y_pred_xgb_train2, squared=True)
    rmse_xgb_train2 = mean_squared_error(y_train_xgo, y_pred_xgb_train2, squared=False)

    mae_xgb_test2 = mean_absolute_error(y_test_xgo, y_pred_xgb_test2)
    mse_xgb_test2 = mean_squared_error(y_test_xgo, y_pred_xgb_test2, squared=True)
    rmse_xgb_test2 = mean_squared_error(y_test_xgo, y_pred_xgb_test2, squared=False)


    # Creation d'un dataframe pour comparer les metriques des 4 algorithmes 
    data2 = {'MAE train': [mae_ridge_train, mae_decision_tree_train2, mae_random_forest_train2, mae_xgb_train2],
            'MAE test': [mae_ridge_test, mae_decision_tree_test2, mae_random_forest_test2, mae_xgb_test2],
            'MSE train': [mse_ridge_train, mse_decision_tree_train2, mse_random_forest_train2, mse_xgb_train2],
            'MSE test': [mse_ridge_test, mse_decision_tree_test2, mse_random_forest_test2, mse_xgb_test2],
            'RMSE train': [rmse_ridge_train, rmse_decision_tree_train2, rmse_random_forest_train2, rmse_xgb_train2],
            'RMSE test': [rmse_ridge_test, rmse_decision_tree_test2, rmse_random_forest_test2, rmse_xgb_test2]}

    # Creer le dataFrame
    df2 = pd.DataFrame(data2, index=['Régression linéaire', 'Arbre à décision', 'Random forest', 'XGBoostregressor'])

    st.write(df2)






    






    


    
