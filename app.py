import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import train_demo_model, predict_risk, make_demo_history

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "database", "virtual_patients.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

st.set_page_config(page_title="Virtual Patient AI | Clinical Lab", page_icon="✚", layout="wide", initial_sidebar_state="expanded")

# Language preferences are stored beside the app only when the user opts in.
PREFS_PATH = os.path.join(APP_DIR, "user_preferences.json")
def read_preferences():
    try:
        import json
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
def write_preferences(language):
    import json
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump({"language": language}, f, ensure_ascii=False, indent=2)
def forget_preferences():
    try: os.remove(PREFS_PATH)
    except FileNotFoundError: pass

PREFS = read_preferences()
if "language" not in st.session_state:
    st.session_state.language = PREFS.get("language", "")
    st.session_state.remember_language = bool(PREFS.get("language"))
if "welcome_complete" not in st.session_state:
    st.session_state.welcome_complete = bool(PREFS.get("language"))

# Full bilingual labels plus a safe phrase translator for the dashboard copy.
AR = {
"Overview":"نظرة عامة", "Virtual Patient":"المريض الافتراضي", "Digital Twin Simulator":"محاكي التوأم الرقمي",
"Patient Records":"سجلات المرضى", "Model & Validation":"النموذج والتحقق", "Settings":"الإعدادات",
"Navigate":"التنقل", "Workspace overview":"نظرة عامة على مساحة العمل", "Recent virtual patients":"أحدث المرضى الافتراضيين",
"Patient":"المريض", "Age":"العمر", "Created":"تاريخ الإنشاء", "GET STARTED":"ابدأ من هنا",
"No patient profiles yet":"لا توجد ملفات مرضى حتى الآن", "Research note":"ملاحظة بحثية", "PATIENT WORKSPACE":"مساحة عمل المريض",
"Virtual Patient Profile":"ملف المريض الافتراضي", "Patient information":"بيانات المريض", "Patient display name / ID":"اسم المريض / المعرّف",
"Age (years)":"العمر (بالسنوات)", "Body Mass Index (BMI)":"مؤشر كتلة الجسم (BMI)", "Glucose (mg/dL)":"الجلوكوز (mg/dL)",
"Systolic blood pressure (mmHg)":"ضغط الدم الانقباضي (mmHg)", "Family history of diabetes":"تاريخ عائلي لمرض السكري",
"Weekly physical activity":"النشاط البدني الأسبوعي", "No":"لا", "Yes":"نعم", "Very low":"منخفض جدًا", "Low":"منخفض", "Moderate":"متوسط", "High":"مرتفع",
"✦  Analyze virtual patient":"✦  تحليل المريض الافتراضي", "＋ Save patient to local records":"＋ حفظ المريض في السجلات المحلية",
"Virtual patient saved to this computer.":"تم حفظ المريض الافتراضي على هذا الكمبيوتر.", "Analysis snapshot":"ملخص التحليل",
"Demo model score":"نتيجة النموذج التجريبي", "MODEL RESPONSE":"استجابة النموذج", "Input profile":"بيانات الإدخال",
"SCENARIO LAB":"مختبر السيناريوهات", "Baseline profile":"الملف الأساسي", "Alternative profile":"الملف البديل",
"Set the starting values for comparison.":"حدد القيم الأساسية للمقارنة.", "Adjust the values to explore a second hypothetical case.":"غيّر القيم لاستكشاف حالة افتراضية ثانية.",
"Comparison results":"نتائج المقارنة", "Scenario A score":"نتيجة السيناريو A", "Scenario B score":"نتيجة السيناريو B",
"Difference (B − A)":"الفرق (B − A)", "Model output comparison":"مقارنة مخرجات النموذج", "LOCAL DATA":"البيانات المحلية",
"Review and export the sample virtual-patient profiles saved on this computer.":"راجع وصدّر ملفات المرضى الافتراضيين المحفوظة على هذا الكمبيوتر.",
"Your workspace is empty":"مساحة العمل فارغة", "Saved profiles":"الملفات المحفوظة", "Average demo score":"متوسط النتيجة التجريبية",
"Latest record":"أحدث سجل", "⇩ Export records as CSV":"⇩ تصدير السجلات بصيغة CSV", "Delete all saved demo records":"حذف كل السجلات التجريبية المحفوظة",
"TRANSPARENCY & METHODS":"الشفافية والمنهجية", "Model & Validation":"النموذج والتحقق", "Model card":"بطاقة النموذج",
"Intended use":"الاستخدام المقصود", "Input variables":"متغيرات الإدخال", "Limitations":"القيود", "To reach research quality":"للوصول إلى مستوى بحثي",
"Synthetic holdout metrics":"مقاييس الاختبار للبيانات الاصطناعية", "Accuracy":"الدقة", "Training data":"بيانات التدريب",
"Clinical validation":"التحقق السريري", "Not performed":"لم يُجرَ", "Algorithm":"الخوارزمية", "Synthetic":"اصطناعية",
"Remember my language":"تذكّر اللغة التي اخترتها", "Remember selected language":"تذكّر اللغة المختارة", "Interface language":"لغة الواجهة",
"Save language preference":"حفظ تفضيل اللغة", "Language preference saved.":"تم حفظ تفضيل اللغة.", "Language preference will not be saved.":"لن يتم حفظ تفضيل اللغة.",
"Welcome to the Future of Digital Healthcare":"مرحبًا بك في مستقبل الرعاية الصحية الرقمية", "Choose your language to continue":"اختر اللغة للمتابعة",
"Continue":"متابعة", "Welcome":"مرحبًا", "Local system":"النظام المحلي", "Your demo records stay on this computer.":"تبقى سجلات العرض على هذا الكمبيوتر.", "LOCAL DEMO":"عرض محلي", "AI LAB":"مختبر الذكاء الاصطناعي", "ACADEMIC PROTOTYPE":"نموذج أكاديمي",
"INTELLIGENT HEALTHCARE EXPERIENCE":"تجربة رعاية صحية ذكية", "Workspace overview":"نظرة عامة على مساحة العمل", "Patient profile":"ملف المريض",
"What-if simulation":"محاكاة ماذا لو؟", "Family history":"التاريخ العائلي", "Activity":"النشاط", "Month":"الشهر",
"Illustrative measurements timeline":"خط زمني توضيحي للقياسات", "Scenario A":"السيناريو A", "Scenario B":"السيناريو B",
"Danger zone — remove demonstration records":"منطقة خطرة — حذف السجلات التجريبية", "Export records as CSV":"تصدير السجلات بصيغة CSV",
}

def tr(value):
    if not isinstance(value, str) or st.session_state.get("language") != "العربية": return value
    # Longest phrases first to avoid partial replacements.
    for en in sorted(AR, key=len, reverse=True):
        value = value.replace(en, AR[en])
    return value

def to_english(value):
    if not isinstance(value, str): return value
    for en, ar in AR.items():
        if value == ar: return en
    return value

def translate_arg(value):
    if isinstance(value, str): return tr(value)
    if isinstance(value, list): return [translate_arg(x) for x in value]
    if isinstance(value, tuple): return tuple(translate_arg(x) for x in value)
    return value

# Translate Streamlit's visible text arguments without changing stored model values.
for _method in ("markdown", "title", "header", "subheader", "caption", "info", "success", "warning", "error", "write", "text", "metric", "radio", "selectbox", "slider", "text_input", "button", "form_submit_button", "download_button", "checkbox", "expander"):
    _original = getattr(st, _method, None)
    if _original:
        def _make_translator(fn):
            def _wrapped(*args, **kwargs):
                args = tuple(translate_arg(x) for x in args)
                kwargs = {k: translate_arg(v) if k in ("label", "options", "help", "placeholder") else v for k,v in kwargs.items()}
                return fn(*args, **kwargs)
            return _wrapped
        setattr(st, _method, _make_translator(_original))

if not st.session_state.welcome_complete:
    st.markdown("""<style>
    .stApp {background:radial-gradient(ellipse at 50% 0%,#143755 0%,#07111f 58%,#060d18 100%);}
    .block-container {max-width:900px;padding-top:8vh;}
    .welcome-card {background:linear-gradient(145deg,#102a42,#0b1b2c);border:1px solid #28506a;border-radius:28px;padding:42px 44px;text-align:center;box-shadow:0 24px 80px #0005;}
    .welcome-logo {font-size:2.4rem;font-weight:850;letter-spacing:-.05em;color:#f0f7ff;}
    .welcome-accent {color:#43d9e8;} .welcome-tag {color:#52dce9;letter-spacing:.18em;font-size:.75rem;font-weight:800;margin-top:15px;}
    .welcome-sub {color:#a9bfd4;font-size:1.05rem;margin:14px 0 26px;}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='welcome-card'><div class='welcome-logo'>✚ VIRTUAL<span class='welcome-accent'>PATIENT</span> AI</div><div class='welcome-tag'>DIGITAL HEALTHCARE EXPERIENCE</div><div class='welcome-sub'>Welcome to the Future of Digital Healthcare<br>مرحبًا بك في مستقبل الرعاية الصحية الرقمية</div></div>", unsafe_allow_html=True)
    with st.form("welcome_language_form"):
        chosen = st.radio("Choose your language to continue / اختر اللغة للمتابعة", ["العربية", "English"], horizontal=True, label_visibility="visible")
        remember = st.checkbox("تذكّر اللغة التي اخترتها — Remember my language", value=False)
        go = st.form_submit_button("متابعة — Continue", type="primary", use_container_width=True)
    if go:
        st.session_state.language = chosen
        st.session_state.remember_language = remember
        st.session_state.welcome_complete = True
        if remember: write_preferences(chosen)
        else: forget_preferences()
        st.rerun()
    st.stop()

if not st.session_state.get("language"):
    st.session_state.language = "English"


# Visual system: dark clinical dashboard, cyan/blue accents, rounded cards.
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#07111f; --panel:#0d1b2c; --panel2:#102238; --line:#1e3550; --muted:#8da4bd; --text:#e8f1fb; --cyan:#43d9e8; --blue:#5689ff; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.stApp { background: radial-gradient(ellipse at 70% 0%, #112c48 0%, #07111f 48%, #060d18 100%); color:var(--text); }
.block-container { padding-top:1.3rem; padding-bottom:2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1a2b 0%,#07111f 100%); border-right:1px solid #1c3148; }
[data-testid="stSidebar"] * { color:#dce9f7; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:10px; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#10253a,#0c1b2b); border:1px solid #1c3851; padding:17px 18px; border-radius:16px; box-shadow:0 8px 26px #00000018; }
[data-testid="stMetricLabel"] { color:#91a9c2 !important; font-size:.83rem !important; }
[data-testid="stMetricValue"] { color:#f0f7ff !important; font-weight:750; }
[data-testid="stMetricDelta"] { color:var(--cyan) !important; }
.stButton>button, .stDownloadButton>button, button[kind="primary"] { border-radius:10px; min-height:2.7rem; border:1px solid #2b6681; background:linear-gradient(100deg,#146b86,#2454a2); color:white; font-weight:650; }
.stButton>button:hover { border-color:#55dbe8; color:white; }
input, textarea, [data-baseweb="select"] > div { background-color:#0c1b2b !important; color:#e8f1fb !important; border-color:#24425e !important; border-radius:10px !important; }
label, .stMarkdown p, .stMarkdown li { color:#c5d4e5; }
h1,h2,h3 { color:#f0f7ff !important; letter-spacing:-.03em; }
hr { border-color:#20374e; }
.hero { padding:26px 28px; border:1px solid #21445f; border-radius:22px; background:linear-gradient(115deg,#102a42ed,#0b1b2cf2 65%,#102e40); position:relative; overflow:hidden; margin-bottom:20px; }
.hero:after { content:'✚'; position:absolute; right:35px; top:-36px; font-size:190px; color:#3ad5e8; opacity:.07; font-weight:800; }
.eyebrow { color:#52dce9; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; }
.hero-title { color:#f3f8ff; font-size:2.2rem; line-height:1.15; font-weight:800; margin:8px 0 10px; }
.hero-sub { color:#9db4ca; max-width:760px; font-size:.98rem; line-height:1.6; }
.pill { display:inline-block; border:1px solid #28506a; color:#8ce9ef; background:#0c2838; padding:5px 10px; border-radius:99px; font-size:.75rem; margin-right:6px; }
.section-head { display:flex; align-items:center; gap:10px; margin:24px 0 12px; color:#f0f7ff; font-size:1.12rem; font-weight:750; }
.section-head span { color:#49d8e6; }
.panel { background:linear-gradient(145deg,#0e2033,#0b1928); border:1px solid #1e3851; border-radius:17px; padding:18px 20px; margin-bottom:14px; }
.panel-title { font-weight:750; color:#e9f3ff; margin-bottom:6px; }
.panel-muted { color:#8da7c0; font-size:.87rem; line-height:1.55; }
.big-number { color:#f2f8ff; font-size:1.85rem; font-weight:800; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#40d6b0; margin-right:7px; }
div[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #1e3851; border-radius:12px; overflow:hidden; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("language") == "العربية":
    st.markdown("""<style>
    html, body, [class*="css"] {direction:rtl; text-align:right;}
    [data-testid="stSidebar"] {border-right:0;border-left:1px solid #1c3148;}
    .hero:after {right:auto;left:35px;}
    .pill {margin-right:0;margin-left:6px;}
    .section-head {flex-direction:row;}
    </style>""", unsafe_allow_html=True)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, age INTEGER NOT NULL,
            bmi REAL NOT NULL, glucose INTEGER NOT NULL, systolic_bp INTEGER NOT NULL,
            family_history TEXT NOT NULL, activity TEXT NOT NULL, demo_score REAL NOT NULL,
            created_at TEXT NOT NULL)""")
        conn.commit()


def save_patient(p, score):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO patients
        (label, age, bmi, glucose, systolic_bp, family_history, activity, demo_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (p["patient_id"], p["age"], p["bmi"], p["glucose"], p["systolic_bp"],
         p["family_history"], p["activity"], round(score, 2), datetime.now().isoformat(timespec="seconds")))
        conn.commit()


def load_patients():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)


@st.cache_resource
def get_model():
    return train_demo_model()


def hero(kicker, title, subtitle):
    st.markdown(f'''<div class="hero"><div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div><div style="margin-top:17px"><span class="pill">LOCAL DEMO</span><span class="pill">AI LAB</span><span class="pill">ACADEMIC PROTOTYPE</span></div></div>''', unsafe_allow_html=True)


def chart_layout(fig, height=310):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a9bfd4", family="Inter"), margin=dict(l=12,r=12,t=50,b=12),
        title_font=dict(color="#e8f2fc", size=15), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1b3045", zerolinecolor="#1b3045", linecolor="#1b3045")
    fig.update_yaxes(gridcolor="#1b3045", zerolinecolor="#1b3045", linecolor="#1b3045")
    return fig

init_db()
model, model_metrics = get_model()

with st.sidebar:
    st.markdown("<div style='padding:12px 4px 20px'><div style='font-size:1.8rem;font-weight:800;letter-spacing:-.06em;color:#f0f7ff'>✚ VIRTUAL<span style='color:#45d9e7'>PATIENT</span></div><div style='font-size:.76rem;color:#7f9ab4;letter-spacing:.12em;margin-top:5px'>AI CLINICAL SIMULATION LAB</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.7rem;color:#63819d;font-weight:800;letter-spacing:.15em;margin-bottom:8px'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Virtual Patient", "Digital Twin Simulator", "Patient Records", "Model & Validation", "Settings"], label_visibility="collapsed")
    page = {AR.get(k, k): k for k in ["Overview", "Virtual Patient", "Digital Twin Simulator", "Patient Records", "Model & Validation", "Settings"]}.get(page, page)
    st.markdown("<br><div class='panel'><div class='panel-title'><span class='status-dot'></span>Local system</div><div class='panel-muted'>Your demo records stay on this computer.</div></div>", unsafe_allow_html=True)
    

if page == "Overview":
    hero("INTELLIGENT HEALTHCARE EXPERIENCE", "Virtual Patient AI", "An interactive digital-patient workspace for exploring synthetic health data, machine-learning demo scores, and what-if scenarios.")
    a,b,c,d = st.columns(4)
    a.metric("AI engine", "Random Forest", "Demo model")
    b.metric("Patient records", len(load_patients()), "Saved locally")
    c.metric("Input features", "6", "Patient variables")
    d.metric("System status", "Ready", "Local prototype")
    st.markdown("<div class='section-head'><span>▦</span> Workspace overview</div>", unsafe_allow_html=True)
    left, right = st.columns([1.35,1])
    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Explore a virtual patient</div><div class='panel-muted'>Create a sample profile, inspect the selected measurements, and run the demonstration model. Use the simulator to compare two hypothetical sets of values.</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='eyebrow'>MODULE 01</div><div class='panel-title'>Patient profile</div><div class='panel-muted'>Age, BMI, glucose, blood pressure, family history and activity.</div></div>", unsafe_allow_html=True)
    with right:
        recent = load_patients().head(5)
        if not recent.empty:
            st.markdown("<div class='section-head'><span>◷</span> Recent virtual patients</div>", unsafe_allow_html=True)
            st.dataframe(recent[["label","age","demo_score","created_at"]].rename(columns={"label":"Patient","age":"Age","demo_score":"Demo score %","created_at":"Created"}), use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='panel'><div class='eyebrow'>GET STARTED</div><div class='panel-title'>No patient profiles yet</div><div class='panel-muted'>Open Virtual Patient to create the first sample record.</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-title'>Research note</div><div class='panel-muted'>The current model uses synthetic generated data. Its scores and test metrics are for software demonstration only and are not clinically validated.</div></div>", unsafe_allow_html=True)

elif page == "Virtual Patient":
    hero("PATIENT WORKSPACE", "Virtual Patient Profile", "Build a sample digital patient profile and explore the model's demonstration output.")
    with st.form("patient_form"):
        st.markdown("<div class='section-head'><span>◉</span> Patient information</div>", unsafe_allow_html=True)
        label = st.text_input("Patient display name / ID", "Virtual Patient 001")
        col1,col2,col3 = st.columns(3)
        with col1:
            age = st.slider("Age (years)",18,90,45)
            bmi = st.slider("Body Mass Index (BMI)",15.0,50.0,29.0,0.5)
        with col2:
            glucose = st.slider("Glucose (mg/dL)",60,300,115)
            bp = st.slider("Systolic blood pressure (mmHg)",80,220,125)
        with col3:
            family = st.selectbox("Family history of diabetes",["No","Yes"])
            activity = st.selectbox("Weekly physical activity",["Very low","Low","Moderate","High"],index=1)
        analyze = st.form_submit_button("✦  Analyze virtual patient",type="primary",use_container_width=True)
    family = to_english(family)
    activity = to_english(activity)
    if analyze:
        patient = {"patient_id":label.strip() or "Virtual Patient","age":age,"bmi":bmi,"glucose":glucose,"systolic_bp":bp,"family_history":family,"activity":activity}
        st.session_state["last_patient"] = patient
        st.session_state["last_result"] = predict_risk(model,patient)
    if "last_patient" in st.session_state:
        patient = st.session_state["last_patient"]
        result = st.session_state["last_result"]
        st.markdown(f"<div class='section-head'><span>▣</span> Analysis snapshot — {patient['patient_id']}</div>",unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Age",f"{patient['age']} years")
        m2.metric("BMI",f"{patient['bmi']:.1f}")
        m3.metric("Glucose",f"{patient['glucose']} mg/dL")
        m4.metric("Demo model score",f"{result['risk']:.1f}%")
        score_fig = go.Figure(go.Indicator(mode="gauge+number",value=result["risk"],number={"suffix":"%","font":{"color":"#eff7ff","size":34}},title={"text":"MODEL DEMO SCORE","font":{"color":"#9db4ca","size":12}},gauge={"axis":{"range":[0,100],"tickcolor":"#8099b2"},"bar":{"color":"#43d9e8"},"bgcolor":"#0b1928","bordercolor":"#24435e","steps":[{"range":[0,30],"color":"#102b38"},{"range":[30,60],"color":"#183047"},{"range":[60,100],"color":"#342638"}]}))
        score_fig.update_layout(height=245,paper_bgcolor="rgba(0,0,0,0)",font_color="#e8f1fb",margin=dict(l=18,r=18,t=35,b=5))
        chartcol, detailscol = st.columns([1,1.2])
        with chartcol: st.plotly_chart(score_fig,use_container_width=True,config={"displayModeBar":False})
        with detailscol:
            st.markdown(f"<div class='panel'><div class='eyebrow'>MODEL RESPONSE</div><div class='big-number'>{result['category']}</div><div class='panel-muted'>This label describes only the synthetic demonstration model's output. It is not a real-world clinical risk category.</div></div>",unsafe_allow_html=True)
            st.markdown("<div class='panel'><div class='panel-title'>Input profile</div><div class='panel-muted'>Family history: <b>"+patient["family_history"]+"</b><br>Weekly activity: <b>"+patient["activity"]+"</b><br>Systolic pressure: <b>"+str(patient["systolic_bp"])+" mmHg</b></div></div>",unsafe_allow_html=True)
        if st.button("＋ Save patient to local records",type="primary",use_container_width=True):
            save_patient(patient,result["risk"])
            st.success("Virtual patient saved to this computer.")
        hist = make_demo_history(patient)
        fig = px.line(hist,x="Month",y=["Glucose (illustrative)","BMI (illustrative)"],markers=True,title="Illustrative measurements timeline")
        fig.update_traces(line_width=3)
        st.plotly_chart(chart_layout(fig,300),use_container_width=True)
        st.caption("The timeline is randomly generated for interface demonstration; it is not a forecast or actual patient history.")

elif page == "Digital Twin Simulator":
    hero("SCENARIO LAB", "Digital Twin Simulator", "Compare two hypothetical virtual-patient profiles and see how the synthetic demo model responds to each input set.")
    st.info("This is a what-if interface demonstration. It does not model human physiology or prove that changing a measurement causes an outcome.")
    ca,cb = st.columns(2)
    with ca:
        st.markdown("<div class='panel'><div class='eyebrow'>SCENARIO A</div><div class='panel-title'>Baseline profile</div><div class='panel-muted'>Set the starting values for comparison.</div></div>",unsafe_allow_html=True)
        a_age=st.slider("Age",18,90,45,key="aa")
        a_bmi=st.slider("BMI",15.0,50.0,29.0,0.5,key="ab")
        a_glucose=st.slider("Glucose (mg/dL)",60,300,115,key="ag")
        a_bp=st.slider("Systolic BP (mmHg)",80,220,125,key="ap")
        a_fam=st.selectbox("Family history",["No","Yes"],key="af")
        a_act=st.selectbox("Activity",["Very low","Low","Moderate","High"],index=1,key="at")
    with cb:
        st.markdown("<div class='panel'><div class='eyebrow'>SCENARIO B</div><div class='panel-title'>Alternative profile</div><div class='panel-muted'>Adjust the values to explore a second hypothetical case.</div></div>",unsafe_allow_html=True)
        b_age=st.slider("Age",18,90,45,key="ba")
        b_bmi=st.slider("BMI",15.0,50.0,26.0,0.5,key="bb")
        b_glucose=st.slider("Glucose (mg/dL)",60,300,100,key="bg")
        b_bp=st.slider("Systolic BP (mmHg)",80,220,120,key="bp")
        b_fam=st.selectbox("Family history",["No","Yes"],key="bf")
        b_act=st.selectbox("Activity",["Very low","Low","Moderate","High"],index=2,key="bt")
    a_fam, b_fam = to_english(a_fam), to_english(b_fam)
    a_act, b_act = to_english(a_act), to_english(b_act)
    pa={"age":a_age,"bmi":a_bmi,"glucose":a_glucose,"systolic_bp":a_bp,"family_history":a_fam,"activity":a_act}
    pb={"age":b_age,"bmi":b_bmi,"glucose":b_glucose,"systolic_bp":b_bp,"family_history":b_fam,"activity":b_act}
    ra,rb=predict_risk(model,pa),predict_risk(model,pb)
    st.markdown("<div class='section-head'><span>⇄</span> Comparison results</div>",unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.metric("Scenario A score",f"{ra['risk']:.1f}%")
    y.metric("Scenario B score",f"{rb['risk']:.1f}%")
    z.metric("Difference (B − A)",f"{rb['risk']-ra['risk']:+.1f} pts")
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Scenario A",x=["Demo score"],y=[ra["risk"]],marker_color="#4d8cff",text=[f"{ra['risk']:.1f}%"],textposition="outside"))
    fig.add_trace(go.Bar(name="Scenario B",x=["Demo score"],y=[rb["risk"]],marker_color="#43d9e8",text=[f"{rb['risk']:.1f}%"],textposition="outside"))
    fig.update_layout(barmode="group",title="Model output comparison",yaxis=dict(range=[0,110],title="Synthetic demo score (%)"))
    st.plotly_chart(chart_layout(fig,350),use_container_width=True,config={"displayModeBar":False})
    comp=pd.DataFrame({"Measurement":["Age","BMI","Glucose","Systolic BP","Family history","Activity","Demo score"],"Scenario A":[a_age,a_bmi,a_glucose,a_bp,a_fam,a_act,round(ra["risk"],1)],"Scenario B":[b_age,b_bmi,b_glucose,b_bp,b_fam,b_act,round(rb["risk"],1)]})
    st.dataframe(comp,use_container_width=True,hide_index=True)

elif page == "Patient Records":
    hero("LOCAL DATA", "Patient Records", "Review and export the sample virtual-patient profiles saved on this computer.")
    df=load_patients()
    if df.empty:
        st.markdown("<div class='panel'><div class='panel-title'>Your workspace is empty</div><div class='panel-muted'>Analyze a virtual patient and save the profile to populate this table.</div></div>",unsafe_allow_html=True)
    else:
        m1,m2,m3=st.columns(3)
        m1.metric("Saved profiles",len(df))
        m2.metric("Average demo score",f"{df['demo_score'].mean():.1f}%")
        m3.metric("Latest record",str(df.iloc[0]["created_at"])[:10])
        st.dataframe(df,use_container_width=True,hide_index=True)
        csv=df.to_csv(index=False).encode("utf-8")
        st.download_button("⇩ Export records as CSV",csv,file_name="virtual_patients.csv",mime="text/csv")
        with st.expander("Danger zone — remove demonstration records"):
            st.warning("This permanently deletes all saved demo records from the local database.")
            if st.button("Delete all saved demo records"):
                with sqlite3.connect(DB_PATH) as conn: conn.execute("DELETE FROM patients"); conn.commit()
                st.success("All demonstration records were deleted. Refresh the page to update the table.")

elif page == "Settings":
    hero("PREFERENCES", "Settings", "Customize your interface language and decide whether to remember it next time.")
    st.markdown("<div class='panel'><div class='panel-title'>Interface Language / لغة الواجهة</div><div class='panel-muted'>The interface updates immediately. You can change this any time from the sidebar.</div></div>", unsafe_allow_html=True)
    current = st.session_state.get("language", "English")
    language_choice = st.radio("Interface language", ["العربية", "English"], index=0 if current == "العربية" else 1, horizontal=True, key="settings_language_choice")
    remember_choice = st.checkbox("تذكّر اللغة المختارة — Remember selected language", value=st.session_state.get("remember_language", False), key="settings_remember_choice")
    if language_choice != current:
        st.session_state.language = language_choice
        st.session_state.remember_language = remember_choice
        if remember_choice: write_preferences(language_choice)
        else: forget_preferences()
        st.rerun()
    if remember_choice != st.session_state.get("remember_language", False):
        st.session_state.remember_language = remember_choice
        if remember_choice: write_preferences(st.session_state.get("language", "English"))
        else: forget_preferences()
        st.success("Language preference saved." if remember_choice else "Language preference will not be saved.")
else:
    hero("TRANSPARENCY & METHODS", "Model & Validation", "Understand the current model, the data used to train it, and what would be required for a research-grade medical AI system.")
    c1,c2,c3=st.columns(3)
    c1.metric("Algorithm","Random Forest")
    c2.metric("Training data","Synthetic")
    c3.metric("Clinical validation","Not performed")
    st.markdown("<div class='section-head'><span>⌘</span> Model card</div>",unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Intended use</div><div class='panel-muted'>Demonstrate how a local Python application can collect synthetic profile data and pass it to a machine-learning model.</div></div>",unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-title'>Input variables</div><div class='panel-muted'>Age, body mass index, glucose, systolic blood pressure, family history, and self-reported activity level.</div></div>",unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'><div class='panel-title'>Limitations</div><div class='panel-muted'>Artificial data are not representative clinical evidence. Output scores are not calibrated medical probabilities, and no diagnostic or treatment decisions should be made from them.</div></div>",unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-title'>To reach research quality</div><div class='panel-muted'>Use a legally usable documented dataset, define intended use, evaluate on an independent test set, assess calibration and subgroup performance, and seek clinical/ethics oversight where appropriate.</div></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-head'><span>▤</span> Synthetic holdout metrics</div>",unsafe_allow_html=True)
    m1,m2=st.columns(2)
    m1.metric("Accuracy",f"{model_metrics['accuracy']:.3f}")
    m2.metric("ROC-AUC",f"{model_metrics['roc_auc']:.3f}")
    st.caption("These metrics describe an artificial synthetic-data test split only. They do not indicate real-world medical performance.")

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;color:#67839e;font-size:.75rem'><span>VIRTUAL PATIENT AI · ACADEMIC DEMONSTRATION</span><span>LOCAL FIRST · SYNTHETIC DATA · NOT FOR CLINICAL USE</span></div>",unsafe_allow_html=True)
