
from collections import Counter

import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 1. CONSTANTS
DATASET_ID = "mteb/banking77"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_COUNT = 12
SEARCH_LIMIT = 5
STOP_WORDS = {
    "and",
    "can",
    "did",
    "for",
    "from",
    "has",
    "have",
    "how",
    "the",
    "that",
    "this",
    "was",
    "why",
    "with",
    "you",
    "your",
}


# 2. HELPER FUNCTIONS
def keyword_search(query, records, limit=SEARCH_LIMIT):
    terms = [
        term
        for term in query.lower().split()
        if len(term) > 2 and term not in STOP_WORDS
    ]
    ranked = []

    for record in records:
        text = record["text"]
        score = sum(term in text.lower() for term in terms)
        if score:
            ranked.append(
                {
                    "keyword_hits": score,
                    "ticket": text,
                    "known_intent": record["label_text"],
                }
            )

    ranked.sort(key=lambda result: result["keyword_hits"], reverse=True)
    return ranked[:limit]


def semantic_search(query, pipeline, limit=SEARCH_LIMIT):
    query_embedding = pipeline["model"].encode(
        [query],
        normalize_embeddings=True,
    )
    similarities = pipeline["model"].similarity(
        query_embedding,
        pipeline["embeddings"],
    )[0]
    ranked_indices = sorted(
        range(len(pipeline["records"])),
        key=lambda index: float(similarities[index]),
        reverse=True,
    )[:limit]

    return [
        {
            "similarity": round(float(similarities[index]), 3),
            "ticket": pipeline["records"][index]["text"],
            "known_intent": pipeline["records"][index]["label_text"],
        }
        for index in ranked_indices
    ]


# 3. PIPELINE GENERATOR (CACHED)
@st.cache_resource(
    scope="session",
    show_spinner="Building embeddings and issue themes...",
)
def build_pipeline():
    dataset = load_dataset(DATASET_ID, split="train")
    records = [dict(record) for record in dataset]
    texts = [record["text"] for record in records]
    true_labels = [record["label"] for record in records]

    model = SentenceTransformer(
        MODEL_ID,
        similarity_fn_name="cosine",
    )
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    clusterer = KMeans(
        n_clusters=CLUSTER_COUNT,
        random_state=42,
        n_init=10,
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    distances = clusterer.transform(embeddings)
    
    summaries = []
    for cluster_id in range(CLUSTER_COUNT):
        indices = [
            index
            for index, label in enumerate(cluster_labels)
            if int(label) == cluster_id
        ]
        mean_distance = sum(
            float(distances[index][cluster_id]) for index in indices
        ) / len(indices)
        cohesion = 1 / (1 + mean_distance)
        representative_index = min(
            indices,
            key=lambda index: float(distances[index][cluster_id]),
        )
        dominant_intent, dominant_count = Counter(
            records[index]["label_text"] for index in indices
        ).most_common(1)[0]

        summaries.append(
            {
                "cluster": cluster_id,
                "ticket_count": len(indices),
                "share_pct": round(100 * len(indices) / len(records), 1),
                "cohesion": round(cohesion, 3),
                "signal_score": round(len(indices) * cohesion, 1),
                "dominant_known_intent": dominant_intent,
                "intent_purity_pct": round(
                    100 * dominant_count / len(indices),
                    1,
                ),
                "representative_ticket": records[representative_index]["text"],
            }
        )
        
    # Sort summaries outside of the cluster loop
    summaries.sort(
        key=lambda summary: summary["signal_score"],
        reverse=True,
    )

    return {
        "records": records,
        "model": model,
        "embeddings": embeddings,
        "summaries": summaries,
        "silhouette": silhouette_score(
            embeddings,
            cluster_labels,
            sample_size=2000,
            random_state=42,
        ),
        "adjusted_rand": adjusted_rand_score(
            true_labels,
            cluster_labels,
        ),
    }


# 4. STREAMLIT APP RENDERING
st.set_page_config(
    page_title="Support Signal Radar",
    layout="wide",
)
st.write("# Support Signal Radar")
st.write(
    "Turn customer-support language into semantic search results, "
    "ranked issue themes, and decision evidence."
)

# Load pipeline data
pipeline = build_pipeline()

# Step 5 Validation Portfolio Snapshot
st.write("## Portfolio snapshot")
st.dataframe(
    [
        {
            "tickets_analyzed": len(pipeline["records"]),
            "clusters": CLUSTER_COUNT,
            "silhouette_score": round(pipeline["silhouette"], 3),
            "adjusted_rand_index": round(pipeline["adjusted_rand"], 3),
        }
    ],
    hide_index=True,
)

st.write("## Ranked issue themes")
st.write(
    "Signal score is a transparent heuristic: ticket volume multiplied "
    "by distance-based cohesion."
)
st.dataframe(
    pipeline["summaries"],
    hide_index=True,
    width="stretch",
)  

st.write("## Search a customer problem")
query = st.text_input(
    "Describe the issue in natural language",
    value="Why did the ATM take extra money?",
    type="search",
)

if query.strip():
    keyword_results = keyword_search(query, pipeline["records"])

    st.write("### Keyword baseline")
    if keyword_results:
        st.dataframe(
            keyword_results,
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("No literal keyword matches survived the baseline filter.")

    st.write("### Semantic search")
    st.dataframe(
        semantic_search(query, pipeline),
        hide_index=True,
        width="stretch",
    )
