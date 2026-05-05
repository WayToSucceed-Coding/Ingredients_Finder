import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Ingredients Finder', page_icon='🍳')

with open('finder.pkl', 'rb') as f:
    data = pickle.load(f)

tfidf = data['tfidf']
tfidf_matrix = data['tfidf_matrix']
df = data['df']

def find_recipe(query):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    best_idx = np.argmax(scores)
    
    return {
        'ingredients': df['ingredients'][best_idx]
    }

st.title('🍳 Ingredients Finder')
st.write('Describe dish and I will find ingredients!')
st.divider()

query = st.text_input('Describe your ingredients or dish:')
search_button = st.button('Find Recipe', type='primary', use_container_width=True)

if search_button and query.strip():

    result = find_recipe(query)

    st.divider()
    st.subheader('Ingredients:')
    st.code(result['ingredients'], language='python')

elif search_button and not query.strip():

    st.warning('Please describe a dish before clicking Find Recipe.')