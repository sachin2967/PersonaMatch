PersonaMatch: An Exploration in AI-Powered Recommendation Systems

This project was a hands-on exercise to build and understand an end-to-end machine learning application. The goal was to create a dating recommendation system that uses Natural Language Processing (NLP) to match users based on personality, moving beyond simple filters.
💡 Key Concepts Explored

    Natural Language Processing (NLP): Used a Sentence-Transformer model to convert unstructured user essays into meaningful vector embeddings for comparison.

    Hybrid Recommendation Engine: Designed a system that combines cosine similarity for personality with Jaccard similarity for lifestyle attributes to create a balanced compatibility score.

    Interactive Web Development: Built a functional user interface for the model using Streamlit, including a real-time feedback loop where user swipes adjust future recommendations.

    DevOps & Version Control: Solved real-world deployment challenges by learning and implementing Git LFS to manage large data files in a Git repository.

⚙️ Project Workflow & Technical Decisions

    Data Optimization: The original 108 MB OkCupid dataset was processed to a 5 MB file containing only the essential columns. This was a deliberate choice to improve the app's performance and simplify data handling.

    Personality Encoding: A pre-trained Sentence-Transformer model (all-MiniLM-L6-v2) was chosen to generate vector embeddings from user essays. This model offered a strong balance of performance and size, enabling the capture of semantic meaning.

    Scoring Algorithm: A weighted scoring function was designed, assigning 50% to personality, 30% to lifestyle, and 20% to demographics to prioritize deep compatibility while acknowledging practical preferences.

✅ What I Learned

    The practical application of NLP embedding models to solve a semantic similarity problem.

    The importance of a structured project workflow, from data cleaning to deployment.

    How to build a simple but complete web application that serves a live machine learning model.

    Essential version control skills for handling large data files with Git LFS and troubleshooting common deployment errors.

🛠️ Technology & Libraries Used

    Backend: Python, Flask

    Data Science: Pandas, NumPy, Scikit-learn

    NLP: Sentence-Transformers (Hugging Face)

    Web Framework: Streamlit

    Version Control: Git, GitHub, Git LFS
