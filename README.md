
# Support Signal Radar 📡

Turn raw customer support tickets into structured, queryable, and mathematically validated business intelligence. 

This interactive **Streamlit** dashboard processes real-world support language from the banking sector to surface underlying customer friction points, group them automatically into coherent themes using Machine Learning, and compare traditional keyword search against advanced semantic search.

---
## LIVE DEMO:https://support-signal-radargit-ttukyaln2qnebpzubkpvc4.streamlit.app/
## 🚀 Key Features

- **Automated Topic Clustering**: Uses a cached embedding pipeline to vectorise text via `all-MiniLM-L6-v2` and cluster them into 12 distinct customer issue topics using **K-Means**.
- **Cohesion-Based Ranking**: Ranks clusters using a transparent **Signal Score** heuristic (ticket volume multiplied by centroid-distance-based cohesion).
- **Keyword vs. Semantic Search**: Direct comparison of a simple, stop-word-filtered keyword search baseline versus high-dimensional vector search.
- **Enterprise-Grade Validation Metrics**: 
  - **Silhouette Score**: Validates the compactness and separation of our mathematical issue clusters.
  - **Adjusted Rand Index (ARI)**: Benchmarks unsupervised clusters against ground-truth intent labels to measure semantic alignment.
- **Optimised Architecture**: Implements `@st.cache_resource` session caching to store processed embeddings and prevent expensive recomputation on dashboard reruns.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Embeddings & NLP**: Sentence Transformers (`all-MiniLM-L6-v2`), Hugging Face `datasets`
- **Machine Learning**: Scikit-Learn (KMeans, Metrics)
- **Data Engineering**: Pandas, NumPy

---

## 📥 Setup and Installation (Windows)

Follow these steps to configure your environment and run the application locally.

### 1. Clone the Repository
```bash
git clone 
cd support-signal-radar
```

### 2. Set Up a Virtual Environment
Initialize a Python virtual environment to manage dependencies securely.
```powershell
python -m venv venv
```

Activate the environment in Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you encounter an execution policy error on Windows, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, then run the activation script.)*

### 3. Install Dependencies
Make sure you have a `requirements.txt` file in your root folder with the necessary libraries:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the local server:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit dashboard code & layout
├── requirements.txt        # Python dependency manifest
├── README.md               # Project documentation
└── .gitignore              # Ignores venv/, caches, and temporary files
```
