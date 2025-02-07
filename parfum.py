import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Charger le dataset 'final_perfume_data.csv' avec pandas
perfumes = pd.read_csv('final_perfume_data.csv', encoding='ISO-8859-1', low_memory=False)

# Sélectionner les colonnes nécessaires pour l'analyse
perfumes = perfumes[['Name', 'Brand', 'Description', 'Notes', 'Image URL']]

# Supprimer les parfums sans description ou image
perfumes.dropna(subset=['Description', 'Image URL'], inplace=True)

# Convertir les descriptions des parfums en vecteurs TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(perfumes['Description'])  # Utiliser 'Description' à la place de 'type'

# Calculer la similarité cosinus entre les vecteurs TF-IDF des descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fonction pour obtenir des recommandations de parfums basées sur la similarité des descriptions
def get_recommendations_by_brand(brand, cosine_sim=cosine_sim):
    try:
        # Trouver l'index des parfums de la marque spécifiée
        indices = perfumes[perfumes['Brand'] == brand].index.tolist()
        if not indices:
            raise IndexError
    except IndexError:
        # Afficher un message d'erreur si aucune marque n'est trouvée
        st.write("Brand not found. Please check the brand name and try again.")
        return pd.DataFrame(columns=['Brand', 'Name', 'Notes', 'Image URL'])

    # Obtenir les scores de similarité pour tous les parfums de la marque
    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))
    
    # Trier les scores de similarité en ordre décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Sélectionner les indices des 5 parfums les plus similaires
    sim_scores = sim_scores[1:6]
    perfume_indices = [i[0] for i in sim_scores]
    
    # Retourner les informations des parfums recommandés
    return perfumes[['Name', 'Brand', 'Notes', 'Image URL']].iloc[perfume_indices]

# Titre de l'application Streamlit
st.title("Perfume Recommender System")

# Ajouter du CSS personnalisé pour changer la couleur du texte et du bouton avec des couleurs plus claires
st.markdown("""
    <style>
        h1 {
            color: #88C7D6;  /* Couleur claire du titre */
        }
        .stButton>button {
            background-color: #A7D8D3;  /* Couleur de fond claire du bouton */
            color: white;  /* Couleur du texte du bouton */
        }
        p {
            color: #4E7F7B;  /* Couleur claire du texte des paragraphes */
        }
        .stTextInput>div>input {
            border-color: #A7D8D3;  /* Bordure claire du champ de saisie */
        }
    </style>
""", unsafe_allow_html=True)

# Champ de saisie pour la marque du parfum de l'utilisateur
brand_name = st.text_input("Enter the perfume's brand you're looking for :", "Di Ser")

# Afficher les infos et recommandations du parfum lorsque l'utilisateur clique sur le bouton
if st.button('Recommend'):
    try:
        # Afficher les informations du parfum choisi (pour afficher un exemple de parfum de cette marque)
        selected_perfume = perfumes[perfumes['Brand'] == brand_name].iloc[0]
        st.subheader(f"Selected Perfume: {selected_perfume['Name']}")
        st.write(f"Brand: {selected_perfume['Brand']}")
        st.write(f"Description: {selected_perfume['Description']}")
        st.write(f"Notes: {selected_perfume['Notes']}")
        st.image(selected_perfume['Image URL'], caption=selected_perfume['Name'])

        # Afficher les recommandations
        st.subheader("Recommended perfumes from the same brand:")
        recommendations = get_recommendations_by_brand(brand_name)
        if not recommendations.empty:
            for idx, row in recommendations.iterrows():
                st.markdown(f"**{row['Name']}** - brand: {row['Brand']} - notes: {row['Notes']}")
                st.image(row['Image URL'], caption=row['Name'])

    except IndexError:
        # Afficher un message d'erreur si la marque n'est pas trouvée
        st.write("Brand not found. Please check the brand name and try again.")
