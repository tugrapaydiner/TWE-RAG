# demo/app.py
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from twe_rag.pipeline import TWERAGPipeline, PipelineConfig

st.set_page_config(page_title='TWE-RAG (CPU)', layout='wide')

st.title('TWE-RAG: Time-Weighted Evidence RAG (CPU-Only)')
st.markdown('**CPU-only RAG with time-decay, evidence centrality, and budgeted halting**')

q = st.text_input('Query', value='Who is the current CEO of ExampleCorp?')

# Core weights
st.subheader('Scoring Weights')
col1, col2, col3 = st.columns(3)
with col1:
    alpha = st.slider('alpha (BM25)', 0.0, 2.0, 1.0, 0.05)
with col2:
    beta = st.slider('beta (Dense)', 0.0, 2.0, 1.0, 0.05)
with col3:
    gamma = st.slider('gamma (Centrality)', 0.0, 2.0, 0.5, 0.05)

# Time decay parameters (expandable)
with st.expander('⏰ Time Decay Settings (Advanced)', expanded=False):
    st.markdown("""
    Control how document age affects ranking. Higher `base_delta` makes recency more important.
    Longer `tau` means slower decay (documents stay relevant longer).
    """)
    base_delta = st.slider(
        'base_delta (Decay Weight)',
        0.0, 5.0, 2.5, 0.1,
        help='Weight multiplier for time decay. Higher = stronger recency preference.'
    )
    min_tau = st.slider(
        'min_tau (Days for Recency Queries)',
        30, 365, 90, 10,
        help='Tau for "latest/current" queries. Lower = faster decay. e^(-90/90)=37% at 3 months.'
    )
    max_tau = st.slider(
        'max_tau (Days for Historical Queries)',
        180, 1825, 730, 30,
        help='Tau for general queries. Higher = slower decay. e^(-365/730)=61% at 1 year.'
    )

# K stages
st.subheader('Retrieval Budget (K Stages)')
K1, K2, K3 = st.columns(3)
with K1:
    k1 = st.number_input('K1', 10, 100, 30, 5)
with K2:
    k2 = st.number_input('K2', 10, 200, 60, 5)
with K3:
    k3 = st.number_input('K3', 10, 300, 100, 5)

if st.button('🔍 Run Query', type='primary'):
    pipe = TWERAGPipeline(PipelineConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        K_stages=[int(k1), int(k2), int(k3)],
        base_delta=base_delta,
        min_tau=min_tau,
        max_tau=max_tau
    ))
    out = pipe.run(q)
    st.subheader('Meta')
    st.json(out['meta'])
    st.subheader('Top Results')
    for r in out['results']:
        with st.expander(f"{r['id']} | {r['timestamp']} | score={r['score']:.3f}"):
            st.write('Parts:', r['parts'])
            st.write(r['snippet'])
