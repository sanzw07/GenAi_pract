import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for x in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(x)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="NLP Analyzer AI",
    page_icon="🧠",
    layout="wide"
)


# ---------------- DARK MODERN UI ----------------

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}


.stApp {

    background:
    radial-gradient(circle at top left,#1e1b4b 0%,transparent 35%),
    radial-gradient(circle at bottom right,#0f766e 0%,transparent 35%),
    #050816;

    color:white;
}


/* Header */

h1 {

    font-size:48px !important;
    background:linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

}


/* Text */

p, label {

    color:#cbd5e1 !important;

}


/* Cards */

.card {

background:
rgba(255,255,255,0.06);

border:
1px solid rgba(255,255,255,0.12);

backdrop-filter:
blur(12px);

padding:25px;

border-radius:20px;

box-shadow:
0 10px 30px rgba(0,0,0,.35);

}


/* Metrics */

.metric {

background:
linear-gradient(
135deg,
rgba(56,189,248,.25),
rgba(129,140,248,.15)
);

border:
1px solid rgba(255,255,255,.15);

border-radius:18px;

padding:20px;

text-align:center;

}


.metric h2 {

font-size:35px;
color:#38bdf8;

}


/* Input */

textarea {

background:#0f172a !important;

color:white !important;

border-radius:15px !important;

border:1px solid #334155 !important;

}



/* Button */

.stButton button {

background:
linear-gradient(
90deg,
#2563eb,
#7c3aed
);

color:white;

border:none;

border-radius:14px;

height:50px;

font-size:18px;

font-weight:600;

transition:.3s;

}


.stButton button:hover {

transform:scale(1.03);

box-shadow:
0 0 25px rgba(99,102,241,.7);

}



/* Tabs */

button[data-baseweb="tab"] {

color:#94a3b8;

font-size:16px;

}


button[data-baseweb="tab"][aria-selected="true"] {

color:#38bdf8;

}



/* Dataframe */

[data-testid="stDataFrame"] {

border-radius:15px;

}



/* Sidebar */

section[data-testid="stSidebar"] {

background:#020617;

border-right:
1px solid rgba(255,255,255,.1);

}


</style>

""",
unsafe_allow_html=True)



# ---------------- HEADER ----------------


st.title("🧠 NLP Pipeline Analyzer")

st.write(
"""
AI-powered text processing dashboard.

Analyze:
- Tokenization
- Sentence Detection
- Stopword Removal
- Lemmatization
"""
)



# ---------------- SIDEBAR ----------------


with st.sidebar:

    st.markdown(
    """
    ## ⚡ NLP Engine
    
    Pipeline
    
    🔹 Tokenizer  
    🔹 Sentence Parser  
    🔹 Stopword Filter  
    🔹 Lemmatizer  
    
    """
    )



# ---------------- INPUT ----------------


st.markdown(
"""
<div class="card">

<h3>✍️ Input Text</h3>

</div>
""",
unsafe_allow_html=True
)


text = st.text_area(
"",
"Red Bull gives you wings",
height=180
)


run = st.button("🚀 Run NLP Pipeline")



# ---------------- ANALYSIS ----------------


if run:

    with st.spinner("Analyzing text..."):

        words = word_tokenize(text)

        sentences = sent_tokenize(text)

        filtered = [
            w for w in words
            if w.isalpha()
            and w.lower() not in stop_words
        ]

        lemmas = [
            lemmatizer.lemmatize(w)
            for w in filtered
        ]


    # Metrics

    c1,c2,c3 = st.columns(3)


    with c1:

        st.markdown(
        f"""
        <div class="metric">

        <h2>{len(words)}</h2>

        Total Tokens

        </div>
        """,
        unsafe_allow_html=True
        )


    with c2:

        st.markdown(
        f"""
        <div class="metric">

        <h2>{len(sentences)}</h2>

        Sentences

        </div>
        """,
        unsafe_allow_html=True
        )


    with c3:

        st.markdown(
        f"""
        <div class="metric">

        <h2>{len(filtered)}</h2>

        Clean Words

        </div>
        """,
        unsafe_allow_html=True
        )


    st.divider()


    tabs = st.tabs(
    [
        "🔤 Tokens",
        "📄 Sentences",
        "🧹 Clean",
        "🌱 Lemmas"
    ])



    with tabs[0]:

        st.markdown(
        "<div class='card'>Token output</div>",
        unsafe_allow_html=True
        )

        st.json(words)



    with tabs[1]:

        st.markdown(
        "<div class='card'>Sentence segmentation</div>",
        unsafe_allow_html=True
        )

        for i,s in enumerate(sentences):

            st.write(
            f"**{i+1}.** {s}"
            )



    with tabs[2]:

        st.markdown(
        "<div class='card'>Filtered words</div>",
        unsafe_allow_html=True
        )

        st.write(filtered)



    with tabs[3]:

        st.markdown(
        "<div class='card'>Lemma mapping</div>",
        unsafe_allow_html=True
        )

        st.dataframe(
            {
            "Word":filtered,
            "Lemma":lemmas
            },
            use_container_width=True
        )

else:

    st.info(
    "👆 Enter text and launch the NLP pipeline."
    )
