# PersonaMatch: An Exploration in AI-Powered Recommendation Systems

This project was a hands-on exercise to build and understand an end-to-end machine learning application.  
The goal was to create a **dating recommendation system** that uses **Natural Language Processing (NLP)** to match users based on personality, moving beyond simple filters.

---

## 💡 Key Concepts Explored
- **Natural Language Processing (NLP):** Used a Sentence-Transformer model to convert unstructured user essays into meaningful vector embeddings for comparison.  
- **Hybrid Recommendation Engine:** Combined cosine similarity (personality) with Jaccard similarity (lifestyle attributes) for a balanced compatibility score.  
- **Interactive Web Development:** Built a functional UI with Streamlit, including a real-time feedback loop where swipes adjust future recommendations.  
- **DevOps & Version Control:** Managed large datasets with Git LFS and solved deployment issues for reproducibility.  

---

## ⚙️ Project Workflow & Technical Decisions
1. **Data Cleaning & Optimization**
   - Started with a **108 MB OkCupid dataset**, reduced to **5 MB** by keeping only essential columns.  
   - Cleaned structured features (age, diet, smoking, drinking, income, etc.).  
   - Encoded categorical variables and handled missing values (e.g., `"Unknown"` instead of dropping).  

2. **Text Preprocessing**
   - Tokenized and cleaned user essays (essay0–essay9).  
   - Removed stopwords and normalized text for better embeddings.  

3. **Personality Encoding**
   - Used **`all-MiniLM-L6-v2`** Sentence-Transformer for essay embeddings.  
   - Balanced performance and size, capturing semantic meaning efficiently.  

4. **Scoring Algorithm**
   - Weighted compatibility function:  
     - 50% Personality (cosine similarity of embeddings)  
     - 30% Lifestyle (Jaccard similarity)  
     - 20% Demographics (age, location, etc.)  

5. **Recommendation & Feedback**
   - Hybrid recommender generates matches.  
   - Implemented **user swipe feedback loop** in Streamlit to adjust recommendations in real time.  

6. **Exploratory Analysis & Clustering (Extra Work)**
   - Experimented with **KMeans clustering** on embeddings to group users by latent personality traits.  
   - Compared cluster quality with inertia and silhouette scores.  
   - Helped validate personality-driven groupings before finalizing similarity-based matching.  

---

## ✅ What I Learned
- Practical use of **NLP embeddings** for semantic similarity tasks.  
- Building a structured **end-to-end ML workflow** (data cleaning → modeling → deployment).  
- Developing a **live ML-powered web app** with Streamlit + Flask backend.  
- Handling large datasets with **Git LFS**, version control best practices, and debugging deployment issues.  
- How clustering and similarity-based approaches differ in recommendation systems.  

---

## 🛠️ Technology & Libraries Used
- **Backend:** Python, Flask  
- **Data Science:** Pandas, NumPy, Scikit-learn  
- **NLP:** Sentence-Transformers (Hugging Face)  
- **Web Framework:** Streamlit  
- **Version Control:** Git, GitHub, Git LFS  

---

## 📌 Future Improvements
- Integrate **deep learning–based ranking models** (e.g., fine-tuned BERT).  
- Add **A/B testing** for scoring weights to optimize compatibility formula.  
- Expand to **real-time database (PostgreSQL/SQLite)** instead of static CSV.  
- Deploy with **Docker + CI/CD pipeline** for scalable production use.  
