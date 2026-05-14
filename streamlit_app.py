import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="Market Basket Analyzer Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
        .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            transition: transform 0.3s ease;
        }
        .glass-card:hover { transform: translateY(-5px); border: 1px solid rgba(255, 255, 255, 0.2); }

        .info-card {
            background: rgba(0, 210, 255, 0.1);
            border-left: 3px solid #00d2ff;
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            color: #e0e0e0;
        }
        .warning-card {
            background: rgba(255, 193, 7, 0.1);
            border-left: 3px solid #ffc107;
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            color: #e0e0e0;
        }
        .success-card {
            background: rgba(56, 239, 125, 0.1);
            border-left: 3px solid #38ef7d;
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            color: #e0e0e0;
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #00d2ff;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(0, 210, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }
        }
        .kpi-value {
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stButton > button {
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 25px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 20px rgba(0, 210, 255, 0.3);
        }

        .badge-strong { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600; }
        .badge-medium { background: linear-gradient(135deg, #f2994a, #f2c94c); color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600; }
        .badge-weak { background: linear-gradient(135deg, #eb5757, #f2994a); color: white; padding: 5px 15px; border-radius: 20px; font-weight: 600; }

        .header-title {
            font-size: 3em;
            font-weight: 700;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 30px;
        }

        /* Learning Cards */
        .learn-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 30px 25px;
            height: 100%;
            min-height: 380px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        }
        .learn-card:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(0,210,255,0.3);
            box-shadow: 0 12px 32px rgba(0,210,255,0.15);
        }
        .learn-icon {
            font-size: 2.8em;
            margin-bottom: 15px;
        }
        .learn-title {
            font-size: 1.6em;
            font-weight: 700;
            color: #00d2ff;
            margin-bottom: 15px;
        }
        .learn-formula {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 12px 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #b8b8b8;
            border-left: 3px solid #00d2ff;
        }
        .learn-highlight {
            background: rgba(0,210,255,0.06);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid rgba(0,210,255,0.15);
        }
        .learn-highlight-item {
            padding: 6px 0;
            color: #d0d0d0;
            font-size: 0.95em;
        }
        .learn-tip {
            color: #888;
            font-style: italic;
            font-size: 0.9em;
            margin-top: 12px;
            text-align: center;
        }
        .learn-divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin: 25px 0;
        }

        /* Section headers */
        .section-header {
            font-size: 1.8em;
            font-weight: 700;
            color: #00d2ff;
            margin: 40px 0 20px 0;
            text-align: center;
        }
        .section-subheader {
            color: #a0a0a0;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        /* Make sidebar compact */
        [data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.8);
            backdrop-filter: blur(10px);
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---- HELPER FUNCTIONS ----
def get_strength(row):
    if row['lift'] >= 3 and row['confidence'] >= 0.6:
        return 'Strong'
    elif row['lift'] >= 1.5 and row['confidence'] >= 0.4:
        return 'Medium'
    else:
        return 'Weak'

def style_strength(val):
    if val == 'Strong':
        return 'background-color: #11998e; color: white; padding: 5px; border-radius: 10px;'
    elif val == 'Medium':
        return 'background-color: #f2994a; color: white; padding: 5px; border-radius: 10px;'
    else:
        return 'background-color: #eb5757; color: white; padding: 5px; border-radius: 10px;'

# ---- SIDEBAR (Only controls) ----
with st.sidebar:
    st.markdown("## 🛠️ Configuration")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📁 Upload CSV dataset", type=['csv'])
    if uploaded_file is None:
        st.info("👈 Using demo grocery data")
    
    st.markdown("### ⚙️ Parameters")
    
    min_support = st.slider(
        "Min. Support",
        0.01, 0.50, 0.05, 0.01,
        help="Minimum frequency of an itemset in the whole dataset"
    )
    
    min_confidence = st.slider(
        "Min. Confidence",
        0.1, 1.0, 0.3, 0.05,
        help="Probability that the consequent is bought when the antecedent is bought"
    )
    
    min_lift = st.slider(
        "Min. Lift",
        0.5, 5.0, 1.0, 0.1,
        help="Strength of association (lift >1 means positive correlation)"
    )
    
    st.markdown("---")
    st.caption("Made with ❤️ by Varda Kunde")
    st.caption("[GitHub Repo](https://github.com/varda24/association-rule-mining-dashboard)")

# ---- MAIN CONTENT AREA ----
st.markdown('<h1 class="header-title">🛒 Market Basket Analyzer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a0a0a0;">Uncover hidden buying patterns using Association Rule Mining</p>', unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('dataset/groceries.csv')
    except:
        return None

dataset_loaded = False

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        rows = []
        for line in content:
            line = line.strip()
            if line:
                rows.append(line)
        df = pd.DataFrame(rows)
        st.success("✅ Custom dataset loaded successfully!")
        dataset_loaded = True
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
else:
    df = load_default_data()
    if df is not None:
        st.info("📦 Using built‑in grocery dataset. Upload your own CSV in the sidebar.")
        dataset_loaded = True
    else:
        st.warning("⚠️ Upload a CSV dataset to start analysis.")
        df = pd.DataFrame()
        dataset_loaded = False

# ---- PROCESS DATA ----
@st.cache_data
def process_data(df, min_support, min_confidence, min_lift):
    transactions = df.values.tolist()
    clean_transactions = []
    for t in transactions:
        if isinstance(t[0], str):
            items = [item.strip() for item in t[0].split(',')]
        else:
            items = [str(item).strip() for item in t if str(item) != 'nan']
        clean_transactions.append(items)

    te = TransactionEncoder()
    te_ary = te.fit(clean_transactions).transform(clean_transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_confidence]
        rules['strength'] = rules.apply(get_strength, axis=1)
        rules = rules.sort_values('confidence', ascending=False)
    else:
        rules = pd.DataFrame()
    return rules, df_encoded, frequent_itemsets

if dataset_loaded:
    rules, df_encoded, frequent_itemsets = process_data(df, min_support, min_confidence, min_lift)
else:
    rules = pd.DataFrame()
    df_encoded = pd.DataFrame()
    frequent_itemsets = pd.DataFrame()

def make_rules_display(df_rules):
    display_df = df_rules.copy()
    for column in ['antecedents', 'consequents']:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda x: ', '.join(sorted(str(i) for i in x)))
    return display_df

rules_display = make_rules_display(rules) if not rules.empty else pd.DataFrame()

# ---- RENDER LEARNING SECTION ----
def render_learn_section():
    st.markdown('<div class="section-header">📘 Understanding Association Rule Metrics</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Learn what each parameter means and how to choose the right values for your analysis</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="learn-card">
            <div class="learn-icon">📏</div>
            <div class="learn-title">Support</div>
            <p style="color: #c0c0c0; line-height: 1.6; margin-bottom: 15px;">
                Measures how <strong>frequently</strong> an itemset appears across all transactions.
            </p>
            <div class="learn-formula">
                Support(A→B) = <br>
                Transactions(A & B) / Total Transactions
            </div>
            <div class="learn-highlight">
                <div class="learn-highlight-item">
                    🟢 <strong>Low (0.01–0.03)</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">Finds rare patterns, generates many rules</span>
                </div>
                <div class="learn-highlight-item" style="margin-top: 8px;">
                    🔵 <strong>High (0.10+)</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">Only very frequent items, fewer rules</span>
                </div>
            </div>
            <p class="learn-tip">💡 Start low, increase to filter noise</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="learn-card">
            <div class="learn-icon">🎯</div>
            <div class="learn-title">Confidence</div>
            <p style="color: #c0c0c0; line-height: 1.6; margin-bottom: 15px;">
                The <strong>probability</strong> that customers who buy A also buy B.
            </p>
            <div class="learn-formula">
                Confidence(A→B) = <br>
                Support(A & B) / Support(A)
            </div>
            <div class="learn-highlight">
                <div class="learn-highlight-item">
                    🟡 <strong>Low (0.2–0.4)</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">More rules, some may be unreliable</span>
                </div>
                <div class="learn-highlight-item" style="margin-top: 8px;">
                    🟢 <strong>High (0.7+)</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">Very reliable but fewer recommendations</span>
                </div>
            </div>
            <p class="learn-tip">💡 ~0.5 gives balanced, useful results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="learn-card">
            <div class="learn-icon">📈</div>
            <div class="learn-title">Lift</div>
            <p style="color: #c0c0c0; line-height: 1.6; margin-bottom: 15px;">
                Measures the <strong>strength</strong> of association between products.
            </p>
            <div class="learn-formula">
                Lift(A→B) = <br>
                Confidence(A→B) / Support(B)
            </div>
            <div class="learn-highlight">
                <div class="learn-highlight-item">
                    🟢 <strong>Lift > 1</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">Positive association (buy together)</span>
                </div>
                <div class="learn-highlight-item" style="margin-top: 8px;">
                    ⚪ <strong>Lift = 1</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">No relationship (independent)</span>
                </div>
                <div class="learn-highlight-item" style="margin-top: 8px;">
                    🔴 <strong>Lift < 1</strong><br>
                    <span style="font-size: 0.9em; color: #a0a0a0;">Negative association (substitutes)</span>
                </div>
            </div>
            <p class="learn-tip">💡 Keep lift ≥ 1.2 for meaningful rules</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommended Settings
    st.markdown('<hr class="learn-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⭐ Recommended Parameter Settings</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Choose based on your analysis goals</p>', unsafe_allow_html=True)
    
    rec1, rec2, rec3 = st.columns(3)
    
    with rec1:
        st.markdown("""
        <div class="success-card" style="padding: 20px; border-radius: 12px;">
            <h4 style="color: #38ef7d; margin-bottom: 12px;">🔬 Exploratory Analysis</h4>
            <p style="margin: 5px 0;"><strong>Support:</strong> 0.01 – 0.03</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> 0.2 – 0.3</p>
            <p style="margin: 5px 0;"><strong>Lift:</strong> 1.0</p>
            <p style="color: #a0a0a0; font-size: 0.85em; margin-top: 10px;">Find as many potential patterns as possible</p>
        </div>
        """, unsafe_allow_html=True)
    
    with rec2:
        st.markdown("""
        <div class="warning-card" style="padding: 20px; border-radius: 12px;">
            <h4 style="color: #ffc107; margin-bottom: 12px;">⚖️ Balanced Analysis</h4>
            <p style="margin: 5px 0;"><strong>Support:</strong> 0.03 – 0.08</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> 0.4 – 0.6</p>
            <p style="margin: 5px 0;"><strong>Lift:</strong> 1.5</p>
            <p style="color: #a0a0a0; font-size: 0.85em; margin-top: 10px;">Good mix of quantity and quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    with rec3:
        st.markdown("""
        <div class="info-card" style="padding: 20px; border-radius: 12px;">
            <h4 style="color: #00d2ff; margin-bottom: 12px;">🎯 High Precision</h4>
            <p style="margin: 5px 0;"><strong>Support:</strong> 0.08 – 0.15</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> 0.6 – 0.8</p>
            <p style="margin: 5px 0;"><strong>Lift:</strong> 2.0+</p>
            <p style="color: #a0a0a0; font-size: 0.85em; margin-top: 10px;">Only the strongest, most reliable patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Formulas
    st.markdown('<hr class="learn-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📐 Quick Formula Reference</div>', unsafe_allow_html=True)
    
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown("""
        <div style="background: rgba(0,210,255,0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.15);">
            <h4 style="color: #00d2ff;">Support</h4>
            <p style="color: #c0c0c0; font-family: monospace;">P(A ∩ B)</p>
            <p style="color: #888; font-size: 0.85em;">Transactions with both items<br>÷ Total transactions</p>
        </div>
        """, unsafe_allow_html=True)
    with fcol2:
        st.markdown("""
        <div style="background: rgba(56,239,125,0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(56,239,125,0.15);">
            <h4 style="color: #38ef7d;">Confidence</h4>
            <p style="color: #c0c0c0; font-family: monospace;">P(B | A)</p>
            <p style="color: #888; font-size: 0.85em;">Support(A & B)<br>÷ Support(A)</p>
        </div>
        """, unsafe_allow_html=True)
    with fcol3:
        st.markdown("""
        <div style="background: rgba(255,193,7,0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,193,7,0.15);">
            <h4 style="color: #ffc107;">Lift</h4>
            <p style="color: #c0c0c0; font-family: monospace;">P(B|A) / P(B)</p>
            <p style="color: #888; font-size: 0.85em;">Confidence(A→B)<br>÷ Support(B)</p>
        </div>
        """, unsafe_allow_html=True)

# ---- EDUCATIONAL SECTION ----
if uploaded_file is None:
    # Before dataset upload → Show directly
    render_learn_section()
else:
    # After dataset upload → Show in expander
    with st.expander("📘 Learn: What are Support, Confidence & Lift?", expanded=False):
        render_learn_section()

# ---- DATASET PREVIEW ----
if dataset_loaded:
    with st.expander("📄 View Uploaded Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows out of {len(df)} transactions")

# ---- RESULTS ----
if not rules.empty:
    st.markdown("## 📊 Analysis Results")
    
    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><h3>Total Rules</h3><div class="kpi-value">{len(rules)}</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><h3>Avg. Lift</h3><div class="kpi-value">{rules["lift"].mean():.2f}</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card"><h3>Avg. Confidence</h3><div class="kpi-value">{rules["confidence"].mean():.1%}</div></div>', unsafe_allow_html=True)
    with kpi4:
        strong = len(rules[rules['strength'] == 'Strong'])
        st.markdown(f'<div class="kpi-card"><h3>Strong Rules</h3><div class="kpi-value">{strong}</div></div>', unsafe_allow_html=True)

    # Top Product Recommendations
    st.markdown("## 🛒 Top Product Recommendations")
    st.markdown("*If a customer buys the left product, they are likely to also buy the right product. Sorted by reliability (confidence).*")
    
    for _, row in rules.head(15).iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        confidence = row['confidence']
        strength = row['strength']
        badge_class = f"badge-{strength.lower()}"
        st.markdown(f"""
        <div class="glass-card" style="padding: 15px; margin: 8px 0;">
            <span style="font-size: 1.1em;">
                If a customer buys <strong style="color:#00d2ff;">{ant}</strong>, they are likely to also buy 
                <strong style="color:#3a7bd5;">{con}</strong>
            </span>
            <span style="float: right;">
                <span class="{badge_class}">{strength}</span>
                <strong style="font-size: 1.2em; color: #38ef7d;"> {confidence:.0%}</strong> confidence
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Rules", "📈 Visualizations", "🔍 Search by Product", "📥 Export"])

    with tab1:
        col_s, col_f = st.columns([2,1])
        with col_s:
            search = st.text_input("🔍 Search by product")
        with col_f:
            strength_sel = st.multiselect("Strength", ['Strong','Medium','Weak'], default=['Strong','Medium','Weak'])
        
        filtered = rules.copy()
        if search:
            mask = filtered['antecedents'].apply(lambda x: any(search.lower() in str(i).lower() for i in x))
            mask |= filtered['consequents'].apply(lambda x: any(search.lower() in str(i).lower() for i in x))
            filtered = filtered[mask]
        if strength_sel:
            filtered = filtered[filtered['strength'].isin(strength_sel)]
        
        filtered_display = make_rules_display(filtered)
        st.dataframe(filtered_display.style.map(style_strength, subset=['strength']), height=400)
        st.caption(f"Showing {len(filtered)} of {len(rules)} rules")

    with tab2:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig1 = px.scatter(rules_display, x='support', y='confidence', size='lift', color='strength',
                              color_discrete_map={'Strong':'#11998e','Medium':'#f2994a','Weak':'#eb5757'},
                              hover_data=['antecedents','consequents'], title="Support vs Confidence")
            fig1.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
        with col_v2:
            top10 = rules_display.nlargest(10, 'lift').copy()
            top10['label'] = top10.apply(lambda x: f"{x['antecedents']} → {x['consequents']}", axis=1)
            fig2 = px.bar(top10, x='lift', y='label', orientation='h', color='strength',
                          color_discrete_map={'Strong':'#11998e','Medium':'#f2994a','Weak':'#eb5757'},
                          title="Top 10 by Lift")
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        col_v3, col_v4 = st.columns(2)
        with col_v3:
            dist = rules['strength'].value_counts()
            fig3 = px.pie(values=dist.values, names=dist.index, hole=0.4,
                          color_discrete_map={'Strong':'#11998e','Medium':'#f2994a','Weak':'#eb5757'},
                          title="Strength Distribution")
            fig3.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)
        with col_v4:
            top_items = frequent_itemsets.nlargest(15, 'support')
            top_items['items'] = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))
            fig4 = px.bar(top_items, x='support', y='items', orientation='h', color='support',
                          color_continuous_scale='Viridis', title="Top 15 Itemsets")
            fig4.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        all_products = sorted(set([item for itemset in rules['antecedents'] for item in itemset] +
                                  [item for itemset in rules['consequents'] for item in itemset]))
        selected = st.selectbox("🛍️ Select a product for recommendations", all_products)
        if selected:
            recs = rules[rules['antecedents'].apply(lambda x: selected in x)]
            if not recs.empty:
                st.markdown(f"### If a customer buys **{selected}**, they may also buy:")
                for _, row in recs.iterrows():
                    conseq = ', '.join(list(row['consequents']))
                    st.markdown(f"""
                    <div class="glass-card">
                        <p>➡️ <strong style="color:#3a7bd5;">{conseq}</strong>
                        <span class="badge-{row['strength'].lower()}">{row['strength']}</span>
                        <small>Confidence: {row['confidence']:.1%} | Lift: {row['lift']:.2f}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recommendations found with current parameters.")

    with tab4:
        fmt = st.radio("Format", ["CSV", "JSON"], horizontal=True)
        if fmt == "CSV":
            data = rules_display.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data, f"rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        else:
            data = rules_display.to_json(orient='records')
            st.download_button("⬇️ Download JSON", data, f"rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
        
        st.markdown("### 📄 Analysis Summary")
        st.markdown(f"""
        - **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Total rules:** {len(rules)}
        - **Strong:** {len(rules[rules['strength']=='Strong'])} | **Medium:** {len(rules[rules['strength']=='Medium'])} | **Weak:** {len(rules[rules['strength']=='Weak'])}
        - **Avg lift:** {rules['lift'].mean():.2f} | **Avg confidence:** {rules['confidence'].mean():.2%}
        - **Parameters:** Support≥{min_support}, Confidence≥{min_confidence}, Lift≥{min_lift}
        """)

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#888;">Built with Streamlit & Association Rule Mining | © Varda Kunde</p>', unsafe_allow_html=True)