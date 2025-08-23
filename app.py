import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os

# --- 1. SETUP & LOADING ---
st.set_page_config(layout="centered", page_title="PersonaMatch")


@st.cache_resource
def load_assets():
    """Loads all data files, the AI model, and local image paths."""
    df = pd.read_csv('data/okcupid_cleaned.csv')
    essay_embeddings = np.load('data/okcupid_essay_embeddings_v2.npy')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lifestyle_cols = ['body_type', 'diet_type', 'drink_frequency',
                      'drugs', 'smokes', 'sign_seriousness', 'religious_seriousness']
    df_lifestyle_encoded = pd.get_dummies(
        df[lifestyle_cols], drop_first=True, dtype=int)
    lifestyle_column_order = df_lifestyle_encoded.columns

    persona_name_map = {
        'The Sincere Seeker': 'The Genuine Heart', 'The Free Spirit': 'The Wanderer',
        'The Ambitious Professional': 'The Achiever', 'The Homebody': 'The Nester',
        'The Quirky Intellectual': 'The Thinker', 'The Eclectic Creative': 'The Artist',
        'The Gibberish Profile': 'The Adventurer'
    }
    df['persona_name'] = df['persona_name'].map(
        persona_name_map).fillna('The Individualist')

    male_image_dir = os.path.join('images', 'male')
    female_image_dir = os.path.join('images', 'female')
    male_images = [os.path.join(male_image_dir, f) for f in os.listdir(
        male_image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
    female_images = [os.path.join(female_image_dir, f) for f in os.listdir(
        female_image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

    return df, essay_embeddings, model, df_lifestyle_encoded.values, lifestyle_column_order, male_images, female_images


df, essay_embeddings, model, df_lifestyle_encoded, lifestyle_column_order, MALE_IMAGES, FEMALE_IMAGES = load_assets()

# --- Initialize Session State ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "view": "create_profile", "user_profile": None, "recommendations": None,
        "likes": [], "dislikes": []
    }

# --- 2. CORE FUNCTIONS (No changes here) ---


def get_eligible_candidates(new_user_sex, new_user_orientation, df):
    if new_user_orientation == 'straight':
        if new_user_sex == 'm':
            return df[(df['sex'] == 'f') & (df['orientation'].isin(['straight', 'bisexual']))]
        else:
            return df[(df['sex'] == 'm') & (df['orientation'].isin(['straight', 'bisexual']))]
    return df


def recommend_for_new_user(new_user_data, model, df, essay_embeddings, df_lifestyle_encoded, lifestyle_columns, num_recs=200):
    candidates_df = get_eligible_candidates(
        new_user_data['sex'], new_user_data['orientation'], df).copy()
    candidate_indices = candidates_df.index
    target_embedding = model.encode([new_user_data['essays']]).reshape(1, -1)
    new_user_lifestyle_df = pd.DataFrame([new_user_data])
    new_user_lifestyle_encoded = pd.get_dummies(
        new_user_lifestyle_df, drop_first=True, dtype=int)
    target_lifestyle = new_user_lifestyle_encoded.reindex(
        columns=lifestyle_columns, fill_value=0).values
    pers_scores = cosine_similarity(
        target_embedding, essay_embeddings[candidate_indices]).flatten()
    life_scores = 1 - pairwise_distances(
        target_lifestyle, df_lifestyle_encoded[candidate_indices], metric='jaccard').flatten()
    demo_scores = np.exp(-0.1 *
                         np.abs(new_user_data['age'] - candidates_df['age'].values))
    weights = {'personality': 0.50, 'lifestyle': 0.30, 'demographics': 0.20}
    final_scores = (weights['personality'] * pers_scores + weights['lifestyle']
                    * life_scores + weights['demographics'] * demo_scores)
    candidates_df['score'] = final_scores
    return candidates_df.sort_values(by='score', ascending=False).head(num_recs)


def rerank_recommendations(recs_df, likes, dislikes, df):
    if not likes and not dislikes:
        return recs_df
    rerank_scores = np.zeros(len(recs_df))
    if likes:
        liked_personas = df.loc[likes, 'persona_name'].unique()
        rerank_scores += np.where(
            recs_df['persona_name'].isin(liked_personas), 0.15, 0)
    if dislikes:
        disliked_personas = df.loc[dislikes, 'persona_name'].unique()
        rerank_scores -= np.where(
            recs_df['persona_name'].isin(disliked_personas), 0.20, 0)
    recs_df['score'] += rerank_scores
    return recs_df.sort_values(by='score', ascending=False)

# --- 3. UI HELPER DICTIONARIES AND FUNCTIONS ---


SEX_OPTIONS = {'m': 'Male', 'f': 'Female'}
ORIENTATION_OPTIONS = {'straight': 'Straight',
                       'gay': 'Gay', 'bisexual': 'Bisexual'}
SMOKES_OPTIONS = {'no': 'No', 'sometimes': 'Sometimes',
                  'when drinking': 'When Drinking', 'yes': 'Yes', 'trying to quit': 'Trying to Quit'}
DRUGS_OPTIONS = {'never': 'Never', 'sometimes': 'Sometimes', 'often': 'Often'}
DIET_OPTIONS = {'mostly anything': 'Mostly Anything', 'anything': 'Anything', 'strictly anything': 'Strictly Anything', 'mostly vegetarian': 'Mostly Vegetarian', 'mostly vegan': 'Mostly Vegan',
                'strictly vegetarian': 'Strictly Vegetarian', 'strictly vegan': 'Strictly Vegan', 'vegetarian': 'Vegetarian', 'vegan': 'Vegan', 'mostly kosher': 'Mostly Kosher', 'kosher': 'Kosher', 'mostly halal': 'Mostly Halal', 'halal': 'Halal'}
DRINKING_OPTIONS = {val: val.replace(
    '_', ' ').capitalize() for val in df['drink_frequency'].unique()}
if 'unknown' in DRINKING_OPTIONS:
    DRINKING_OPTIONS['unknown'] = 'Prefer not to say'
RELIGIOUS_OPTIONS = {'very_serious': 'Very Serious', 'serious': 'Serious', 'somewhat_serious': 'Somewhat Serious',
                     'not_too_serious': 'Not Too Serious', 'laughing': 'Secular / Non-Religious', 'unknown': 'Prefer Not to Say'}
for val in df['religious_seriousness'].unique():
    if val not in RELIGIOUS_OPTIONS:
        RELIGIOUS_OPTIONS[val] = val.replace('_', ' ').capitalize()

# --- REWRITTEN: MODERN FORM LAYOUT ---


def create_profile_form(profile_data=None):
    with st.container(border=True):
        st.subheader("✨ Create Your Profile")
        st.caption("Fill this out to get your first set of personalized matches.")

        dynamic_defaults = {'age': 30, 'sex': 'm', 'orientation': 'straight', 'body_type': df['body_type'].unique()[0], 'diet_type': df['diet_type'].unique()[0], 'drink_frequency': df['drink_frequency'].unique()[0], 'drugs': 'never', 'smokes': 'no', 'sign_seriousness': df['sign_seriousness'].unique(
        )[0], 'religious_seriousness': df['religious_seriousness'].unique()[0], 'essays': "I'm adventurous and love hiking on weekends. I work in tech but my real passion is landscape photography. Looking for someone with a good sense of humor who doesn't take themselves too seriously."}
        if profile_data:
            dynamic_defaults.update(profile_data)

        with st.form(key='profile_form'):
            st.markdown("##### The Basics")
            col1, col2, col3 = st.columns(3)
            age = col1.number_input(
                "Your Age", min_value=18, max_value=100, value=dynamic_defaults['age'])
            sex = col2.selectbox("Your Gender", options=list(SEX_OPTIONS.keys(
            )), format_func=lambda x: SEX_OPTIONS[x], index=list(SEX_OPTIONS.keys()).index(dynamic_defaults['sex']))
            orientation = col3.selectbox("Your Orientation", options=list(ORIENTATION_OPTIONS.keys(
            )), format_func=lambda x: ORIENTATION_OPTIONS[x], index=list(ORIENTATION_OPTIONS.keys()).index(dynamic_defaults['orientation']))

            st.markdown("##### Your Lifestyle")
            col1, col2 = st.columns(2)
            drink_frequency = col1.selectbox("Drinking Frequency", options=list(DRINKING_OPTIONS.keys(
            )), format_func=lambda x: DRINKING_OPTIONS[x], index=list(DRINKING_OPTIONS.keys()).index(dynamic_defaults['drink_frequency']))
            smokes = col2.selectbox("Smoking Habits", options=list(SMOKES_OPTIONS.keys(
            )), format_func=lambda x: SMOKES_OPTIONS[x], index=list(SMOKES_OPTIONS.keys()).index(dynamic_defaults['smokes']))

            col1, col2 = st.columns(2)
            drugs = col1.selectbox("Drug Use", options=list(DRUGS_OPTIONS.keys(
            )), format_func=lambda x: DRUGS_OPTIONS[x], index=list(DRUGS_OPTIONS.keys()).index(dynamic_defaults['drugs']))
            diet_type = col2.selectbox("Your Diet", options=list(DIET_OPTIONS.keys(
            )), format_func=lambda x: DIET_OPTIONS[x], index=list(DIET_OPTIONS.keys()).index(dynamic_defaults['diet_type']))

            religious_seriousness = st.selectbox("Religious Attitude", options=list(RELIGIOUS_OPTIONS.keys(
            )), format_func=lambda x: RELIGIOUS_OPTIONS[x], index=list(RELIGIOUS_OPTIONS.keys()).index(dynamic_defaults['religious_seriousness']))

            st.markdown("##### About You")
            essays = st.text_area(
                "Tell us about yourself...", value=dynamic_defaults['essays'], height=150, label_visibility="collapsed")

            submitted = st.form_submit_button(
                "Find My Matches!", use_container_width=True)
            if submitted:
                user_profile = {'age': age, 'sex': sex, 'orientation': orientation, 'drink_frequency': drink_frequency, 'smokes': smokes, 'drugs': drugs, 'essays': essays,
                                'diet_type': diet_type, 'religious_seriousness': religious_seriousness, 'body_type': dynamic_defaults['body_type'], 'sign_seriousness': dynamic_defaults['sign_seriousness']}
                st.session_state.app_state.update({"user_profile": user_profile, "view": "swiping", "likes": [
                ], "dislikes": [], "recommendations": None})
                st.rerun()

# --- REWRITTEN: SLEEKER SWIPING UI ---


def swiping_interface():
    st.title("💘 PersonaMatch")
    st.caption(
        "Here are your top recommendations. Like or dislike to help the AI learn your type!")

    with st.sidebar:
        st.header("Your Profile")
        profile = st.session_state.app_state['user_profile']
        st.markdown(f"**Age:** {profile['age']}")
        st.markdown(f"**Gender:** {SEX_OPTIONS[profile['sex']]}")
        st.markdown(
            f"**Orientation:** {ORIENTATION_OPTIONS[profile['orientation']]}")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.app_state['view'] = 'create_profile'
            st.rerun()
        with st.expander("**Swipe History**"):
            st.write(f"**Likes:** {len(st.session_state.app_state['likes'])}")
            st.write(
                f"**Dislikes:** {len(st.session_state.app_state['dislikes'])}")
        st.divider()
        st.markdown(
            "[View on GitHub](https://github.com/your-username/your-repository-name)")

    if st.session_state.app_state['recommendations'] is None:
        with st.spinner("Analyzing your profile and finding top matches..."):
            initial_recs = recommend_for_new_user(
                st.session_state.app_state['user_profile'], model, df, essay_embeddings, df_lifestyle_encoded, lifestyle_column_order)
            st.session_state.app_state['recommendations'] = initial_recs

    reranked_recs = rerank_recommendations(st.session_state.app_state['recommendations'].copy(
    ), st.session_state.app_state['likes'], st.session_state.app_state['dislikes'], df)
    seen_indices = st.session_state.app_state['likes'] + \
        st.session_state.app_state['dislikes']
    display_recs = reranked_recs[~reranked_recs.index.isin(seen_indices)]

    if not display_recs.empty:
        top_profile = display_recs.iloc[0]
        profile_index = top_profile.name

        with st.container(border=True):
            col_img, col_details = st.columns([2, 3])
            with col_img:
                image_path = None
                if top_profile['sex'] == 'm' and MALE_IMAGES:
                    image_path = MALE_IMAGES[profile_index % len(MALE_IMAGES)]
                elif top_profile['sex'] == 'f' and FEMALE_IMAGES:
                    image_path = FEMALE_IMAGES[profile_index % len(
                        FEMALE_IMAGES)]
                if image_path:
                    st.image(image_path, use_container_width=True)

            with col_details:
                st.markdown(
                    f"**{top_profile['persona_name']}** (`{top_profile['job_grouped']}`)")
                c1, c2 = st.columns(2)
                c1.metric("Age", top_profile['age'])
                c2.metric("Gender", SEX_OPTIONS[top_profile['sex']])
                st.divider()
                st.markdown("##### Lifestyle")
                drinks_text = DRINKING_OPTIONS.get(top_profile['drink_frequency'], str(
                    top_profile['drink_frequency']).capitalize())
                smokes_text = SMOKES_OPTIONS.get(
                    top_profile['smokes'], str(top_profile['smokes']).capitalize())
                diet_text = DIET_OPTIONS.get(top_profile['diet_type'], str(
                    top_profile['diet_type']).capitalize())
                st.markdown(
                    f"🍸 {drinks_text} | 🚬 {smokes_text} | 🥗 {diet_text}")
                st.progress(
                    top_profile['score'], text=f"Match Score: {top_profile['score']:.2%}")

        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("👎 Dislike", key=f"dislike_{profile_index}", use_container_width=True, type="secondary"):
            st.session_state.app_state['dislikes'].append(profile_index)
            st.rerun()
        if btn_col2.button("👍 Like", key=f"like_{profile_index}", use_container_width=True, type="primary"):
            st.session_state.app_state['likes'].append(profile_index)
            st.rerun()
    else:
        st.success(
            "🎉 You've seen all available recommendations! Try editing your profile to see more.")


# --- 4. MAIN APP LOGIC (CONTROLS THE VIEW) ---
if st.session_state.app_state['view'] == 'create_profile':
    st.title("💘 PersonaMatch")
    st.markdown("Welcome to **PersonaMatch**! This app uses AI to find your most compatible matches based on personality, lifestyle, and demographics.")
    st.divider()
    create_profile_form(st.session_state.app_state['user_profile'])
else:
    swiping_interface()
