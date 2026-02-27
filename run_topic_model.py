#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Topic Modelling — İtalyanca Edebiyat Metinleri
===================================================
Lemma-tabanlı, deterministik, tekrar edilebilir analiz.
DH Metodoloji Öğretim Aracı felsefesine uygun: öğrenci algoritmayı
çalışırken görür, parametreleri değiştirir, sonuçları gözlemler.

Kullanım:
    streamlit run run_topic_model.py
"""
from __future__ import annotations

import sys
import io
import subprocess
import zipfile
import platform
from datetime import datetime

from collections import Counter

import numpy as np
import pandas as pd
import spacy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Dosya formatları
from odf.opendocument import load as odf_load
from odf.text import P as odf_P
from odf import teletype
import pdfplumber

# Coherence
from gensim.corpora import Dictionary as GensimDictionary
from gensim.models.coherencemodel import CoherenceModel

# ── SABİT PARAMETRELER ──────────────────────────────────────────────────────
SEED = 42
DEFAULT_N_TOPICS = 5
DEFAULT_MAX_ITER = 20
DEFAULT_CHUNK_SIZE = 200
MIN_DF = 2
MAX_DF = 0.85
MIN_TOKEN_LEN = 3
TOP_N_WORDS = 10

LANG_MODELS = {
    "İtalyanca": "it_core_news_sm",
    "İngilizce": "en_core_web_sm",
}

ALL_POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
DEFAULT_POS = ["NOUN", "VERB", "ADJ"]

np.random.seed(SEED)


# ── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────

def get_environment_report(spacy_model: str) -> str:
    """Çalışma ortamı bilgilerini raporla."""
    import gensim
    nlp_meta = spacy.load(spacy_model).meta
    lines = [
        f"Tarih/Saat          : {datetime.now().isoformat()}",
        f"Platform            : {platform.platform()}",
        f"Python              : {sys.version}",
        f"numpy               : {np.__version__}",
        f"scikit-learn        : {sklearn.__version__}",
        f"pandas              : {pd.__version__}",
        f"spaCy               : {spacy.__version__}",
        f"gensim              : {gensim.__version__}",
        f"spaCy model         : {nlp_meta['lang']}_{nlp_meta['name']} v{nlp_meta['version']}",
        f"random_state (seed) : {SEED}",
    ]
    return "\n".join(lines)


def get_model_parameters(
    spacy_model: str,
    effective_min_df: int,
    effective_max_df: float,
    actual_topics: int,
    max_iter: int,
    chunk_size: int,
    pos_tags: list[str],
    custom_sw: set[str],
) -> str:
    """Model ve ön-işleme parametrelerini raporla."""
    lines = [
        "═══ ÖN-İŞLEME ═══",
        f"spaCy model           : {spacy_model}",
        f"Küçük harfe çevirme   : Evet",
        f"Noktalama çıkarma     : Evet",
        f"Sayı çıkarma          : Evet",
        f"Stopword temizleme    : Evet (spaCy + özel)",
        f"Özel stopword         : {', '.join(sorted(custom_sw)) if custom_sw else 'Yok'}",
        f"POS filtresi          : {', '.join(pos_tags)}",
        f"Lemmatizasyon         : Evet",
        f"Min. token uzunluğu   : {MIN_TOKEN_LEN}",
        "",
        "═══ BELGE PARÇALAMA ═══",
        f"Parça boyutu (kelime) : {chunk_size}",
        f"Min. parça uzunluğu   : 20 kelime",
        "",
        "═══ VEKTÖRİZASYON (CountVectorizer) ═══",
        f"min_df                : {effective_min_df}",
        f"max_df                : {effective_max_df}",
        f"ngram_range           : (1, 1)",
        "",
        "═══ MODEL (LatentDirichletAllocation) ═══",
        f"n_components          : {actual_topics}",
        f"random_state          : {SEED}",
        f"learning_method       : batch",
        f"max_iter              : {max_iter}",
    ]
    return "\n".join(lines)


# ── Dosya okuma ──────────────────────────────────────────────────

def read_odt(file_bytes: bytes) -> str:
    doc = odf_load(io.BytesIO(file_bytes))
    paragraphs = doc.getElementsByType(odf_P)
    return "\n".join(teletype.extractText(p) for p in paragraphs)


def read_pdf(file_bytes: bytes) -> str:
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as exc:
        raise ValueError(f"PDF okunamadı (dosya bozuk veya şifreli olabilir): {exc}")


def read_uploaded_file(uploaded_file) -> tuple[str, str | None]:
    """Dosyayı oku. (metin, hata_mesajı) döndürür."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    try:
        if name.endswith(".odt"):
            return read_odt(raw), None
        if name.endswith(".pdf"):
            return read_pdf(raw), None
        for enc in ("utf-8", "latin-1", "iso-8859-9"):
            try:
                return raw.decode(enc), None
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace"), None
    except Exception as exc:
        return "", str(exc)


# ── NLP ──────────────────────────────────────────────────────────

@st.cache_resource
def load_spacy_model(model_name: str):
    """spaCy modelini yükle; bulunmazsa otomatik indir."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name]
        )
        nlp = spacy.load(model_name)
    nlp.max_length = 2_000_000
    return nlp


def preprocess(
    text: str,
    nlp,
    allowed_pos: set[str],
    custom_stopwords: set[str],
) -> list[str]:
    """Tek bir belgeyi ön-işle, temiz token listesi döndür."""
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop:
            continue
        if token.like_num or token.text.isdigit():
            continue
        if token.pos_ not in allowed_pos:
            continue
        lemma = token.lemma_.strip()
        if lemma in custom_stopwords:
            continue
        if len(lemma) >= MIN_TOKEN_LEN and lemma.isalpha():
            tokens.append(lemma)
    return tokens


def preprocess_with_trace(
    text: str,
    nlp,
    allowed_pos: set[str],
    custom_stopwords: set[str],
) -> pd.DataFrame:
    """Ön-işleme adımlarını satır satır gösteren tablo üret."""
    doc = nlp(text.lower())
    rows = []
    for token in doc:
        if token.is_space:
            continue
        lemma = token.lemma_.strip()
        is_punct = token.is_punct
        is_stop = token.is_stop or lemma in custom_stopwords
        is_num = token.like_num or token.text.isdigit()
        is_short = len(lemma) < MIN_TOKEN_LEN
        pos_ok = token.pos_ in allowed_pos
        kept = (
            not is_punct and not is_stop and not is_num
            and not is_short and pos_ok and lemma.isalpha()
        )
        rows.append({
            "Token": token.text,
            "Lemma": lemma,
            "POS": token.pos_,
            "Stopword": is_stop,
            "Noktalama": is_punct,
            "Sayı": is_num,
            "Kısa (<3)": is_short,
            "POS Filtre": not pos_ok,
            "Dahil": kept,
        })
    return pd.DataFrame(rows)


def split_into_chunks(tokens: list[str], chunk_size: int = 200) -> list[list[str]]:
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        if len(chunk) >= 20:
            chunks.append(chunk)
    return chunks


# ── Analiz yardımcıları ──────────────────────────────────────────

def build_topics_text(model, feature_names, n_words=TOP_N_WORDS):
    lines = []
    for idx, topic_dist in enumerate(model.components_):
        top_indices = topic_dist.argsort()[: -n_words - 1 : -1]
        top_words = [feature_names[i] for i in top_indices]
        lines.append(f"Topic {idx}: {', '.join(top_words)}")
    return "\n".join(lines)


def compute_coherence(lda_model, feature_names, token_lists, n_words=TOP_N_WORDS):
    """C_v coherence skoru hesapla (gensim)."""
    try:
        topics_words = []
        for topic_dist in lda_model.components_:
            top_idx = topic_dist.argsort()[: -n_words - 1 : -1]
            topics_words.append([feature_names[i] for i in top_idx])
        dictionary = GensimDictionary(token_lists)
        cm = CoherenceModel(
            topics=topics_words,
            texts=token_lists,
            dictionary=dictionary,
            coherence="c_v",
        )
        return cm.get_coherence()
    except Exception:
        return None


def compute_corpus_stats(
    raw_texts: list[str],
    all_token_lists: list[list[str]],
    nlp,
) -> dict:
    """Derlem istatistiklerini hesapla."""
    # Ham metin istatistikleri
    raw_combined = " ".join(raw_texts)
    raw_words = raw_combined.split()
    raw_word_count = len(raw_words)
    raw_char_count = len(raw_combined)

    # Temizlenmiş token istatistikleri
    all_tokens_flat = [t for tl in all_token_lists for t in tl]
    clean_token_count = len(all_tokens_flat)
    clean_type_count = len(set(all_tokens_flat))

    # Type-Token Ratio
    ttr = clean_type_count / clean_token_count if clean_token_count > 0 else 0.0

    # Frekans dağılımı
    freq = Counter(all_tokens_flat)
    top_50 = freq.most_common(50)

    # Hapax legomena (yalnızca 1 kez geçen)
    hapax = sum(1 for w, c in freq.items() if c == 1)

    # Ortalama parça uzunluğu
    avg_chunk_len = np.mean([len(tl) for tl in all_token_lists])

    return {
        "raw_word_count": raw_word_count,
        "raw_char_count": raw_char_count,
        "clean_token_count": clean_token_count,
        "clean_type_count": clean_type_count,
        "ttr": ttr,
        "top_50": top_50,
        "hapax_count": hapax,
        "avg_chunk_len": avg_chunk_len,
        "freq": freq,
    }


def generate_wordcloud(words_weights: dict[str, float]) -> plt.Figure:
    """Kelime bulutu üret ve matplotlib Figure döndür."""
    wc = WordCloud(
        width=600,
        height=300,
        background_color="white",
        colormap="viridis",
        max_words=50,
        random_state=SEED,
    ).generate_from_frequencies(words_weights)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def create_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files.items():
            zf.writestr(f"outputs/{fname}", content)
    return buf.getvalue()


def render_pyldavis(lda_model, dtm, vectorizer) -> str | None:
    try:
        import pyLDAvis
        import pyLDAvis.lda_model as lda_vis
        vis_data = lda_vis.prepare(lda_model, dtm, vectorizer, sort_topics=False)
        return pyLDAvis.prepared_data_to_html(vis_data)
    except ImportError:
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# STREAMLIT ARAYÜZÜ
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LDA Topic Modelling — DH Öğretim Aracı",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("📚 LDA Topic Modelling")
st.sidebar.caption("Ankara Üniversitesi — Dijital Beşeri Bilimler")
st.sidebar.markdown(
    '<span style="font-size:0.82em; color:#64748b;">'
    'Öğr. Gör. Dr. Oğuz KORAN<br>'
    'Dr. Öğr. Üyesi Barış YÜCESAN</span>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

# Dil
lang_label = st.sidebar.selectbox(
    "🌐 Metin Dili",
    list(LANG_MODELS.keys()),
    help="Metnin yazıldığı dili seçin. spaCy dil modeli buna göre belirlenir: "
         "tokenizasyon, lemmatizasyon ve stopword listesi seçilen dile göre değişir.\n\n"
         "**Neden dil seçimi kritik?**\n\n"
         "Lemmatizasyon dile özgüdür: İtalyanca 'andavo' → 'andare' dönüşümü, "
         "ancak İtalyanca modelle doğru yapılır. Yanlış dil modeli = yanlış lemmalar = "
         "anlamsız topic'ler.\n\n"
         "**Model boyutu:** Bu araç `sm` (small) modelleri kullanır. Daha büyük modeller "
         "(lg) daha doğru lemmatizasyon yapar ancak yükleme süresi ve bellek kullanımı artar.",
)
spacy_model = LANG_MODELS[lang_label]

# POS filtresi
st.sidebar.markdown("### 🏷️ POS Filtresi")
pos_tags = st.sidebar.multiselect(
    "Dahil edilecek sözcük türleri",
    ALL_POS_TAGS,
    default=DEFAULT_POS,
    help="Yalnızca seçili sözcük türlerindeki kelimeler analize dahil edilir.\n\n"
         "• **NOUN** (İsim): Topic modelling'de en bilgilendirici tür. "
         "Akademik çalışmaların çoğu yalnızca isimlerle başlar.\n\n"
         "• **VERB** (Fiil): Eylem temaları yakalar (ör. 'combattere', 'amare'). "
         "Anlatı analizi için önemlidir.\n\n"
         "• **ADJ** (Sıfat): Duygusal ve tanımlayıcı temaları güçlendirir. "
         "Sentiment-yakın analiz için faydalı.\n\n"
         "• **ADV** (Zarf): Genellikle gürültü ekler, dikkatli kullanın. "
         "Akademik çalışmalarda nadiren dahil edilir.\n\n"
         "• **PROPN** (Özel İsim): Karakter/yer adlarını dahil eder. "
         "NER (Named Entity Recognition) tarzı analizlerde yararlı.\n\n"
         "**Akademik tercih:** Schofield ve Mimno (2016) çalışması, "
         "POS filtresinin topic kalitesini önemli ölçüde artırdığını göstermiştir. "
         "**Sadece NOUN** ile başlamak akademik standarttır. "
         "Ardından NOUN+ADJ, sonra NOUN+ADJ+VERB deneyin. "
         "Her seferinde coherence skorunun nasıl değiştiğini gözlemleyin.\n\n"
         "> Schofield, A. & Mimno, D. (2016). 'Comparing Apples to Apple.' "
         "ACL Workshop on NLP+CSS.",
)
if not pos_tags:
    pos_tags = DEFAULT_POS
    st.sidebar.warning("En az bir POS türü gerekli. Varsayılan kullanılıyor.")
allowed_pos_set = set(pos_tags)

# Özel stopword
st.sidebar.markdown("### 🚫 Özel Stopword")
custom_sw_input = st.sidebar.text_input(
    "Ek stopword (virgülle ayırın)",
    placeholder="ör: dire, fare, cosa, essere",
    help="spaCy'nin varsayılan stopword listesine ek olarak çıkarmak istediğiniz kelimeler.\n\n"
         "**Neden gerekli?** spaCy'nin genel stopword listesi günlük dil için tasarlanmıştır. "
         "Edebiyat metinlerinde çok sık geçen ama tematik anlam taşımayan kelimeler "
         "(ör. İtalyancada 'dire', 'fare', 'cosa', 'essere', 'molto', 'ancora') "
         "ek olarak çıkarılmalıdır.\n\n"
         "**Yöntem (iteratif stopword refinement):**\n"
         "1. Analizi ilk kez stopword eklemeden çalıştırın.\n"
         "2. Topic kelimelerine bakın → birden fazla topic'te tekrar eden anlamsız kelimeler var mı?\n"
         "3. Bu kelimeleri buraya ekleyin, tekrar çalıştırın.\n"
         "4. Coherence skoru yükseldiyse doğru yoldasınız.\n\n"
         "**Dikkat:** Tematik anlam taşıyan kelimeleri çıkarmayın. "
         "Örneğin 'morte' (ölüm) sık geçebilir ama tematik bir sinyal taşır.",
)
custom_stopwords = {w.strip().lower() for w in custom_sw_input.split(",") if w.strip()}

# Parametreler
st.sidebar.markdown("### ⚙️ Parametreler")

n_topics = st.sidebar.slider(
    "Konu sayısı (K)", 2, 15, DEFAULT_N_TOPICS, key="k",
    help="LDA'nın kaç farklı konu (topic) arayacağını belirler. "
         "Bu, modelin en kritik hiperparametresidir.\n\n"
         "• **K çok düşükse** → farklı konular tek topic altında birleşir "
         "(under-fitting). Topic'ler çok genel ve belirsiz olur.\n\n"
         "• **K çok yüksekse** → konular anlamsız parçalara bölünür "
         "(over-fitting). Benzer topic'ler çoğalır.\n\n"
         "**Optimal K nasıl bulunur?**\n\n"
         "1. **Coherence (C_v) yöntemi:** Farklı K değerleri (3, 5, 7, 10) deneyin. "
         "En yüksek coherence skorunu veren K'yı seçin. Bu yöntem Röder et al. (2015) "
         "tarafından önerilmiştir.\n\n"
         "2. **İnsani değerlendirme:** Topic kelimelerine bakın — her topic'e anlamlı "
         "bir etiket verebiliyor musunuz? Veremiyorsanız K'yı değiştirin.\n\n"
         "3. **pyLDAvis kontrolü:** Daireler net biçimde ayrışıyorsa K uygun. "
         "Üst üste biniyorsa K'yı düşürün.\n\n"
         "**Akademik standart:** Çoğu DH çalışması, farklı K değerleriyle yapılan "
         "deneyleri ve seçim gerekçelerini raporlar. Tek bir K ile sonuç çıkarmak "
         "metodolojik olarak zayıf kabul edilir.",
)

n_words = st.sidebar.slider(
    "Konu başına kelime", 5, 20, TOP_N_WORDS, key="w",
    help="Her topic için gösterilecek en ağırlıklı kelime sayısı.\n\n"
         "Bu parametre modeli **değiştirmez**, yalnızca gösterim derinliğini ayarlar.\n\n"
         "• **10 kelime** (varsayılan): Topic'in ana temasını görmek için yeterli.\n\n"
         "• **15–20 kelime**: Daha ince ayrımları görmek için. Alt temalar görünür hale gelir.\n\n"
         "**Akademik not:** Çoğu akademik yayında topic'ler 10–15 kelimeyle raporlanır.",
)

chunk_size = st.sidebar.slider(
    "Parça boyutu (kelime)", 50, 500, DEFAULT_CHUNK_SIZE, key="chunk",
    help="Metin kaç kelimelik parçalara bölünsün? "
         "Bu parametre özellikle tek belge yüklendiğinde kritiktir.\n\n"
         "• **Küçük parça (50–100):** Mikro tematik değişimleri yakalar ama gürültülü olabilir. "
         "Kısa şiirler veya paragraf düzeyinde analiz için uygundur.\n\n"
         "• **Orta parça (150–250):** Roman ve uzun denemeler için önerilen aralık. "
         "Tematik ayrımları yakalarken yeterli istatistiksel güç sağlar.\n\n"
         "• **Büyük parça (300–500):** Genel temaları yakalar ama ince ayrımlar kaybolur. "
         "Çok uzun metinlerde veya birden fazla eserde kullanılır.\n\n"
         "**Akademik bağlam:** Parça boyutu, LDA'nın 'belge' olarak neyi gördüğünü belirler. "
         "Jockers (2013) *Macroanalysis* kitabında parça boyutunun sonuçları önemli ölçüde "
         "etkilediğini göstermiştir. Farklı boyutlarla deney yapmanız önerilir.\n\n"
         "**Minimum parça:** 20 kelimeden kısa parçalar otomatik olarak atlanır "
         "(istatistiksel olarak anlamsızdır).",
)

max_iter = st.sidebar.slider(
    "LDA iterasyon", 5, 50, DEFAULT_MAX_ITER, key="iter",
    help="Modelin eğitim döngüsü sayısı (kaç kez veri üzerinden geçer).\n\n"
         "• **10–20 iterasyon:** Küçük ve orta korpuslar için genellikle yeterli.\n\n"
         "• **30–50 iterasyon:** Çok büyük veya karmaşık korpuslarda gerekebilir.\n\n"
         "**Yakınsama (convergence):** Model her iterasyonda biraz daha iyi hale gelir. "
         "Eğer sonuçlar iterasyon artırıldığında değişmiyorsa, model yakınsamış demektir. "
         "Gereksiz yere yüksek iterasyon çalışma süresini uzatır.\n\n"
         "**Not:** Bu araç `batch` öğrenme kullanır — tüm veri her iterasyonda bir kez "
         "işlenir. Bu yöntem deterministiktir (online'a kıyasla) ve akademik tekrar "
         "edilebilirlik için tercih edilir.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Seed:** `{SEED}` · **Model:** `{spacy_model}`")
st.sidebar.caption("Aynı veri + aynı parametreler = aynı sonuç.")


# ── Ana Alan ─────────────────────────────────────────────────────

st.title("📚 LDA Topic Modelling")
st.markdown("**İtalyanca edebiyat metinleri için lemma-tabanlı, deterministik analiz**")

with st.expander("📖 LDA Nedir?", expanded=False):
    st.markdown("""
**Latent Dirichlet Allocation (LDA)**, metinlerdeki gizli *konuları* (topics)
bulmaya çalışan bir olasılıksal modeldir. İlk kez **Blei, Ng ve Jordan (2003)**
tarafından önerilmiştir ve Dijital Beşeri Bilimler'de (*Digital Humanities*) en
yaygın kullanılan *topic modelling* tekniğidir.

**Temel varsayımlar:**
- Her **belge** = konuların bir karışımı (ör. %40 aşk, %30 savaş, %30 doğa)
- Her **konu** = kelimelerin bir olasılık dağılımı (ör. "aşk" konusu: *cuore, amore, dolore, ...*)
- Model, kelime birliktelik örüntülerinden bu gizli yapıyı **istatistiksel olarak** çıkarır

**Ne yapar?**
LDA, "bu kelimeler birlikte sık geçiyorsa muhtemelen aynı konuya aittir"
mantığıyla çalışır. *cuore*, *amore* ve *passione* kelimeleri birlikte görülüyorsa,
bunları bir "konu kümesi" olarak gruplar.

**Generative (üretici) model:**
LDA şu soruyu sorar: "Bu metinleri üreten gizli konu yapısı ne olabilir?"
Bu nedenle **üretici** (generative) bir modeldir — metni olasılıksal olarak
yeniden üretmeye çalışır ve gerçek metne en yakın konu dağılımını bulur.

**Bag-of-Words (BoW) varsayımı:**
LDA, kelime **sırasını** dikkate almaz. Yalnızca kelimelerin belgede **kaç kez**
geçtiğine bakar. Bu sınırlama bilinçli bir tercihtir: kelimelerin konumsal
ilişkisini göz ardı etmek, büyük ölçekli tematik örüntüleri yakalamayı
kolaylaştırır.

**Tek belge yüklediğinizde:** Metin otomatik olarak parçalara bölünür
(her parça bir "pseudo-belge" olur) ve LDA bu parçalar üzerinde çalışır.
Bu yöntem *distant reading* (uzak okuma) yaklaşımıyla uyumludur:
metnin mikro-bölümlerindeki tematik değişimleri gözlemlemenizi sağlar.

> **Referans:** Blei, D.M., Ng, A.Y. & Jordan, M.I. (2003). "Latent Dirichlet Allocation."
> *Journal of Machine Learning Research*, 3, 993–1022.
""")

with st.expander("🔬 Ön-İşleme Adımları", expanded=False):
    st.markdown("""
```
Ham Metin
   ↓  Küçük harfe çevirme
   ↓  Tokenizasyon (spaCy)
   ↓  Noktalama ve sayıları çıkarma
   ↓  Stopword temizleme (spaCy + özel liste)
   ↓  POS filtresi (yalnızca seçili türler)
   ↓  Lemmatizasyon
   ↓  Min. 3 karakter filtresi
Temiz Token Listesi
```

**Neden her adım gerekli?**

| Adım | Neden? | Örnek |
|------|--------|-------|
| **Küçük harf** | "Roma" ve "roma" aynı kelime olarak sayılsın | Roma → roma |
| **Tokenizasyon** | Metni anlamlı birimlere ayırma | "l'amore" → "l'" + "amore" |
| **Noktalama çıkarma** | Virgül, nokta vb. konu bilgisi taşımaz | "casa," → "casa" |
| **Sayı çıkarma** | Rakamlar tematik değildir | "1943", "25" → çıkarılır |
| **Stopword** | Çok sık ama anlamsız kelimeler | "il", "di", "che", "non" |
| **POS filtresi** | Yalnızca içerik kelimelerini tut | İsim, Fiil, Sıfat |
| **Lemmatizasyon** | Farklı çekimleri birleştir | "andavo", "andare", "andata" → "andare" |
| **Min. 3 karakter** | Çok kısa tokenları ele | "di", "a", "in" → çıkarılır |

**Akademik bağlam:** Ön-işleme, topic modelling'in en kritik aşamasıdır.
Schofield, Magnusson ve Mimno (2017) göstermiştir ki kötü ön-işleme,
model parametrelerinden daha fazla sonucu etkiler. Bu nedenle her adımı
kontrol altında tutmak önemlidir — "Adım Adım" tablosu tam da bunu yapmanızı sağlar.
""")

with st.expander("🎯 Akademik Metodoloji: LDA Nasıl Kullanılır?", expanded=False):
    st.markdown("""
**LDA, Dijital Beşeri Bilimler'de nasıl kullanılır?**

Topic modelling, edebiyat araştırmalarında şu amaçlarla kullanılır:

1. **Tematik keşif (*distant reading*):** Büyük bir korpusta hangi konuların
   baskın olduğunu otomatik tespit etme. Franco Moretti'nin *distant reading*
   kavramıyla uyumludur: metni yakından okumadan önce genel yapıyı kavramak.

2. **Karşılaştırmalı analiz:** Farklı yazarların, dönemlerin veya eserlerin
   tematik profillerini karşılaştırma. Örn. Pavese ile Sciascia'nın hangi
   konularda ayrıştığını görmek.

3. **Tematik dönüşüm:** Bir eserin bölümleri arasında konuların nasıl
   değiştiğini takip etme. Roman boyunca hangi temaların yükselip düştüğünü görmek.

**Akademik iş akışı (önerilen):**

```
1. Korpus hazırlığı → metin temizleme, format birliği
2. İlk analiz       → varsayılan parametrelerle çalıştır
3. İteratif iyileştirme:
   a. Topic kelimelerine bak → anlamlı mı?
   b. Anlamsız kelimeler → özel stopword'e ekle
   c. Coherence skoru düşükse → K'yı veya POS'u değiştir
   d. pyLDAvis'te daireler üst üsteyse → K'yı düşür
4. Sonuçları yorumla → topic'lere etiket ver
5. Raporla → parametreler + metrikler + görselleştirmeler
```

**Önemli not:** LDA bir keşif aracıdır (*exploratory tool*). Sonuçlar kesin "doğrular"
değil, yorumlanması gereken *olasılıksal örüntülerdir*. Topic'lerin ne anlama geldiğine
araştırmacı karar verir — model yalnızca kelime kümelerini sunar.
""")

with st.expander("⚠️ Sık Yapılan Hatalar", expanded=False):
    st.markdown("""
| Hata | Neden sorunlu? | Çözüm |
|------|----------------|-------|
| K'yı sabitleyip hiç değiştirmemek | Optimal K veriye bağlıdır | En az 3 farklı K deneyin, coherence'ı karşılaştırın |
| Stopword eklememek | "dire", "fare" gibi kelimeler topic'leri kirletir | İlk çalıştırmada topic kelimelerini inceleyin, tekrar eden anlamsızları ekleyin |
| Yalnızca perplexity'e bakmak | Düşük perplexity ≠ anlamlı topic'ler | Coherence (C_v) + insani yorumlama birlikte kullanın |
| Çok küçük parça boyutu | Topic'ler gürültülü ve tutarsız olur | Tek kitap için en az 150 kelime |
| Çok büyük parça boyutu | İnce tematik ayrımlar kaybolur | 500'den büyük nadiren gerekli |
| POS filtresiz çalışmak | Edatlar, bağlaçlar topic'leri domine eder | En az NOUN filtresi kullanın |
| Tek bir çalıştırma ile sonuç çıkarmak | LDA parametrelere duyarlıdır | İteratif deneme yapın, farklı ayarları karşılaştırın |
| Sonuçları "kanıt" olarak sunmak | LDA keşif aracıdır, ispat aracı değil | "LDA şunu gösteriyor" yerine "LDA şuna işaret ediyor" kullanın |
""")

st.divider()

# Dosya yükleme
uploaded_files = st.file_uploader(
    "Belge yükleyin (.txt, .odt veya .pdf)",
    type=["txt", "odt", "pdf"],
    accept_multiple_files=True,
    help="Her dosya ayrı bir belge olarak değerlendirilir. "
         "Tek dosya yüklerseniz metin otomatik parçalanır.",
)

if uploaded_files:
    st.info(f"📄 {len(uploaded_files)} dosya yüklendi.")

    if st.button("▶ Analizi Başlat", type="primary", use_container_width=True):

        # ── 1. Dosyaları oku ─────────────────────────────────────
        with st.spinner("Dosyalar okunuyor…"):
            filenames = []
            raw_texts = []
            for f in uploaded_files:
                f.seek(0)
                text, err = read_uploaded_file(f)
                if err:
                    st.warning(f"⚠️ {f.name}: {err} — atlandı.")
                    continue
                if not text.strip():
                    st.warning(f"⚠️ {f.name}: Dosya boş — atlandı.")
                    continue
                filenames.append(f.name)
                raw_texts.append(text)

        if not raw_texts:
            st.error("Hiçbir dosyadan metin okunamadı.")
            st.stop()
        st.success(f"✅ {len(raw_texts)} belge okundu.")

        # ── 2. spaCy ön-işleme ──────────────────────────────────
        with st.spinner(f"spaCy ön-işleme ({spacy_model})…"):
            nlp = load_spacy_model(spacy_model)

            all_token_lists = []
            chunk_labels = []
            trace_dfs = {}

            for fname, raw in zip(filenames, raw_texts):
                tokens = preprocess(raw, nlp, allowed_pos_set, custom_stopwords)

                if len(trace_dfs) < 3:
                    preview_text = " ".join(raw.split()[:500])
                    trace_dfs[fname] = preprocess_with_trace(
                        preview_text, nlp, allowed_pos_set, custom_stopwords
                    )

                if len(tokens) > chunk_size:
                    chunks = split_into_chunks(tokens, chunk_size)
                    for ci, chunk in enumerate(chunks):
                        all_token_lists.append(chunk)
                        chunk_labels.append(f"{fname}_p{ci+1}")
                elif len(tokens) >= 20:
                    all_token_lists.append(tokens)
                    chunk_labels.append(fname)
                else:
                    st.warning(
                        f"⚠️ {fname}: ön-işleme sonrası çok az token "
                        f"({len(tokens)}), atlandı."
                    )

        if not all_token_lists:
            st.error("Hiçbir belgede yeterli token bulunamadı.")
            st.stop()

        n_chunks = len(all_token_lists)
        st.success(
            f"✅ {n_chunks} metin parçası hazır "
            f"({len(filenames)} belge → {n_chunks} parça)"
        )

        # Pedagojik: ön-işleme tabloları
        if trace_dfs:
            st.subheader(
                "🔍 Ön-İşleme: Adım Adım",
                help="Bu tablo metnin LDA'ya verilmeden önce nasıl dönüştürüldüğünü gösterir.\n\n"
                     "• **Token:** Orijinal kelime.\n\n"
                     "• **Lemma:** Sözlük formu (ör. 'andavo' → 'andare').\n\n"
                     "• **POS:** Sözcük türü (NOUN, VERB, ADJ vb.).\n\n"
                     "• **POS Filtre:** Seçili olmayan sözcük türleri.\n\n"
                     "• 🟢 Yeşil = analize dahil | 🔴 Kırmızı = çıkarılan.",
            )
            for fname, tdf in trace_dfs.items():
                with st.expander(f"{fname} — ilk 500 kelime"):
                    def _hl(row):
                        color = "#dcfce7" if row.get("Dahil") else "#fee2e2"
                        return [f"background-color: {color}"] * len(row)
                    st.dataframe(
                        tdf.style.apply(_hl, axis=1),
                        use_container_width=True,
                        height=350,
                    )
                    n_kept = int(tdf["Dahil"].sum())
                    st.caption(
                        f"🟢 Dahil: {n_kept} | 🔴 Çıkarılan: {len(tdf) - n_kept}"
                    )

        # ── 2b. Derlem İstatistikleri ─────────────────────────────
        with st.spinner("Derlem istatistikleri hesaplanıyor…"):
            corpus_stats = compute_corpus_stats(raw_texts, all_token_lists, nlp)

        st.subheader(
            "📊 Derlem İstatistikleri",
            help="Analiz öncesi derlem hakkında temel nicel bilgiler.\n\n"
                 "Bu veriler, LDA sonuçlarını yorumlamak için bağlam sağlar. "
                 "Akademik makalenizin veri tanıtım bölümünde bu istatistikleri raporlayın.\n\n"
                 "**Type-Token Ratio (TTR):** Sözcüksel çeşitliliği ölçer. "
                 "Yüksek TTR → metin zengin ve çeşitli bir söz varlığı kullanıyor. "
                 "Düşük TTR → metin daha tekrarlayıcı.\n\n"
                 "**Hapax Legomena:** Yalnızca 1 kez geçen kelimeler. "
                 "Dilbilimde Zipf yasasına göre, herhangi bir korpusta kelimelerin büyük "
                 "bir oranı yalnızca bir kez geçer.",
        )

        # Özet metrikler
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "Ham Sözcük",
            f"{corpus_stats['raw_word_count']:,}".replace(",", "."),
            help="Ön-işleme öncesi, ham metindeki toplam sözcük sayısı (whitespace split).",
        )
        s2.metric(
            "Temiz Token",
            f"{corpus_stats['clean_token_count']:,}".replace(",", "."),
            help="Lemmatizasyon, POS filtresi ve stopword temizliği sonrası kalan token sayısı. "
                 "Bu sayı, LDA'ya girdi olarak verilen toplam kelime miktarıdır.",
        )
        s3.metric(
            "Benzersiz Kelime (Type)",
            f"{corpus_stats['clean_type_count']:,}".replace(",", "."),
            help="Temizlenmiş tokenlar arasındaki farklı kelime sayısı. "
                 "LDA'nın sözlük boyutunu etkiler.",
        )
        s4.metric(
            "TTR",
            f"{corpus_stats['ttr']:.3f}",
            help="Type-Token Ratio = benzersiz kelime / toplam token. "
                 "Sözcüksel çeşitliliğin temel göstergesi.\n\n"
                 "• **> 0.20** → Çeşitli söz varlığı\n\n"
                 "• **0.05 – 0.20** → Orta düzey\n\n"
                 "• **< 0.05** → Tekrarlayıcı metin\n\n"
                 "⚠️ TTR, metin uzunluğuna duyarlıdır: uzun metinlerde doğal olarak düşer.",
        )

        s5, s6, s7, s8 = st.columns(4)
        s5.metric(
            "Karakter Sayısı",
            f"{corpus_stats['raw_char_count']:,}".replace(",", "."),
            help="Ham metindeki toplam karakter sayısı (boşluklar dahil).",
        )
        s6.metric(
            "Hapax Legomena",
            f"{corpus_stats['hapax_count']:,}".replace(",", "."),
            help="Yalnızca 1 kez geçen kelime sayısı. "
                 "Zipf yasasına göre, doğal dil metinlerinde kelimelerin yaklaşık "
                 "%40–60'ı hapax legomena'dır.",
        )
        s7.metric(
            "Hapax Oranı",
            f"{corpus_stats['hapax_count'] / corpus_stats['clean_type_count'] * 100:.1f}%"
            if corpus_stats['clean_type_count'] > 0 else "—",
            help="Hapax legomena / benzersiz kelime sayısı. "
                 "Doğal dil metinlerinde %40–60 arası normaldir.",
        )
        s8.metric(
            "Ort. Parça Uzunluğu",
            f"{corpus_stats['avg_chunk_len']:.0f} token",
            help="LDA'ya verilen parçaların ortalama token sayısı.",
        )

        # En sık 50 kelime — bar chart
        with st.expander("📈 En Sık 50 Kelime (Frekans Dağılımı)", expanded=True):
            st.markdown(
                "Ön-işleme sonrası **en sık geçen 50 kelime**. "
                "Bu liste, LDA'ya girdi olan temizlenmiş sözlükteki dağılımı gösterir."
            )
            top_50_df = pd.DataFrame(
                corpus_stats["top_50"], columns=["Kelime", "Frekans"]
            )
            if not top_50_df.empty:
                freq_chart = (
                    alt.Chart(top_50_df)
                    .mark_bar(color="#2563EB")
                    .encode(
                        x=alt.X("Frekans:Q", title="Frekans"),
                        y=alt.Y("Kelime:N", sort="-x", title="Kelime"),
                        tooltip=["Kelime", "Frekans"],
                    )
                    .properties(height=max(400, len(top_50_df) * 18))
                )
                st.altair_chart(freq_chart, use_container_width=True)

                st.dataframe(
                    top_50_df.style.format({"Frekans": "{:,}"}),
                    use_container_width=True,
                    height=300,
                )

            st.info(
                "💡 **İpucu:** Bu listede birden fazla topic'te tekrar eden anlamsız kelimeler "
                "görüyorsanız, onları sol paneldeki **özel stopword** alanına ekleyin ve "
                "analizi yeniden çalıştırın."
            )

        with st.expander("📖 Derlem İstatistikleri Ne Anlama Geliyor?"):
            st.markdown(f"""
**Derlem profili özeti:**

| İstatistik | Değer | Açıklama |
|---|---|---|
| Ham sözcük sayısı | {corpus_stats['raw_word_count']:,} | Ön-işleme öncesi toplam sözcük |
| Temiz token sayısı | {corpus_stats['clean_token_count']:,} | LDA'ya girdi olan kelime sayısı |
| Benzersiz kelime (type) | {corpus_stats['clean_type_count']:,} | Farklı kelime sayısı |
| Type-Token Ratio | {corpus_stats['ttr']:.4f} | Sözcüksel çeşitlilik |
| Hapax legomena | {corpus_stats['hapax_count']:,} | Yalnızca 1 kez geçen kelimeler |
| Filtreleme oranı | {(1 - corpus_stats['clean_token_count'] / corpus_stats['raw_word_count']) * 100:.1f}% | Ham metinden ne kadarı elendi |

**Bu sayıları nasıl yorumlamalı?**

1. **Filtreleme oranı:** Ön-işleme, ham metnin yaklaşık %{(1 - corpus_stats['clean_token_count'] / corpus_stats['raw_word_count']) * 100:.0f}'ini eledi.
   Bu normaldir — İtalyanca metinlerde stopword, noktalama ve fonksiyon kelimeleri
   metnin %60–80'ini oluşturur.

2. **TTR ({corpus_stats['ttr']:.3f}):** {"Yüksek sözcüksel çeşitlilik — metin zengin bir söz varlığı kullanıyor." if corpus_stats['ttr'] > 0.20 else "Orta/düşük TTR — uzun metinlerde bu normaldir (tekrar eden kelimeler artar)." }

3. **Hapax oranı:** Benzersiz kelimelerin %{corpus_stats['hapax_count'] / corpus_stats['clean_type_count'] * 100:.0f}'i yalnızca bir kez geçiyor.
   Bu, doğal dil metinlerinde beklenen bir Zipf dağılımıdır.

**Akademik bağlam:** Derlem istatistikleri, sonuçlarınızı kontekstüalize eder.
Çok küçük bir derlem (< 1.000 token) üzerinde LDA sonuçları güvenilir olmayabilir.
Genel kural: en az **2.000–5.000 temiz token** ile çalışmak önerilir.

> **Referans:** Zipf, G.K. (1949). *Human Behavior and the Principle of Least Effort*.
> Addison-Wesley.
""")

        # ── 3. Vektörizasyon ─────────────────────────────────────
        with st.spinner("CountVectorizer…"):
            text_chunks = [" ".join(tl) for tl in all_token_lists]

            effective_min_df = 1 if n_chunks < 3 else MIN_DF
            effective_max_df = 1.0 if n_chunks < 3 else MAX_DF

            vectorizer = CountVectorizer(
                min_df=effective_min_df,
                max_df=effective_max_df,
                ngram_range=(1, 1),
            )
            dtm = vectorizer.fit_transform(text_chunks)
            feature_names = vectorizer.get_feature_names_out().tolist()

        if len(feature_names) == 0:
            st.error(
                "Vektörizasyon sonrası sözlük boş kaldı. "
                "Parça boyutunu küçültün veya POS filtresini genişletin."
            )
            st.stop()

        st.success(f"✅ Sözlük: {len(feature_names)} terim | DTM: {dtm.shape}")

        # ── 4. LDA ──────────────────────────────────────────────
        actual_topics = min(n_topics, n_chunks, len(feature_names))
        if actual_topics < n_topics:
            st.warning(
                f"Parça/terim sayısı yetersiz → K={actual_topics} olarak ayarlandı."
            )

        with st.spinner(f"LDA eğitiliyor (K={actual_topics}, iter={max_iter})…"):
            lda = LatentDirichletAllocation(
                n_components=actual_topics,
                random_state=SEED,
                learning_method="batch",
                max_iter=max_iter,
            )
            doc_topic = lda.fit_transform(dtm)

        perplexity_val = lda.perplexity(dtm)
        log_likelihood_val = lda.score(dtm)

        # Coherence
        with st.spinner("Coherence skoru hesaplanıyor…"):
            coherence_val = compute_coherence(
                lda, feature_names, all_token_lists, n_words
            )

        # ── 5. Çıktıları hazırla ────────────────────────────────
        topics_txt = build_topics_text(lda, feature_names, n_words)

        topic_cols = [f"Topic_{i}" for i in range(actual_topics)]
        df_dist = pd.DataFrame(doc_topic, columns=topic_cols)
        df_dist.insert(0, "Parça", chunk_labels)

        coherence_str = f"{coherence_val:.4f}" if coherence_val else "hesaplanamadı"
        metrics_txt = (
            f"Perplexity     : {perplexity_val:.4f}\n"
            f"Log-likelihood : {log_likelihood_val:.4f}\n"
            f"Coherence (C_v): {coherence_str}"
        )
        params_txt = get_model_parameters(
            spacy_model, effective_min_df, effective_max_df,
            actual_topics, max_iter, chunk_size, pos_tags, custom_stopwords,
        )
        env_txt = get_environment_report(spacy_model)
        env_txt += (
            f"\nToplam belge sayısı : {len(filenames)}"
            f"\nToplam parça sayısı : {n_chunks}"
        )

        # Terminal
        print("\n" + "=" * 60)
        print("LDA TOPIC MODELLING — ORTAM BİLGİLERİ")
        print("=" * 60)
        print(env_txt)
        print("=" * 60 + "\n")

        # ── 6. Sonuçları göster ──────────────────────────────────
        st.divider()
        st.header("📊 Sonuçlar")

        # Metrikler
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Belgeler", len(filenames), help="Yüklenen dosya sayısı.")
        m2.metric(
            "Parçalar", n_chunks,
            help="Ön-işleme ve parçalama sonrası LDA'ya verilen toplam metin birimi.",
        )
        m3.metric(
            "Perplexity", f"{perplexity_val:.2f}",
            help="Modelin metni ne kadar iyi tahmin ettiğini ölçer.\n\n"
                 "• Düşük perplexity → model metne daha iyi uyum sağlamış.\n\n"
                 "• Farklı K değerleriyle karşılaştırın.\n\n"
                 "⚠️ Tek başına yeterli değildir; coherence ile birlikte değerlendirin.",
        )
        m4.metric(
            "Log-Likelihood", f"{log_likelihood_val:.2f}",
            help="Modelin veriye atadığı olasılığın logaritması. "
                 "Yüksek (sıfıra yakın) → daha iyi uyum.",
        )
        m5.metric(
            "Coherence (C_v)",
            f"{coherence_val:.3f}" if coherence_val else "—",
            help="Topic'lerin insan tarafından ne kadar anlamlı algılandığını ölçen standart metriktir.\n\n"
                 "**Değer aralığı:** 0 ile 1 arası (yüksek = daha iyi).\n\n"
                 "• **> 0.55** → Topic'ler genellikle anlamlı ve tutarlı.\n\n"
                 "• **0.40 – 0.55** → Kabul edilebilir, ama iyileştirme mümkün.\n\n"
                 "• **< 0.40** → Topic'ler muhtemelen anlamlı değil; K'yı, parça boyutunu "
                 "veya POS filtresini değiştirmeyi deneyin.\n\n"
                 "**Perplexity ile farkı:** Perplexity modelin istatistiksel uyumunu, "
                 "coherence ise topic'lerin insani yorumlanabilirliğini ölçer. "
                 "Düşük perplexity ama düşük coherence mümkündür — her ikisine de bakın.\n\n"
                 "**Nasıl iyileştirilir?**\n"
                 "1. Farklı K değerleri deneyin.\n"
                 "2. POS filtresini daraltın (sadece NOUN).\n"
                 "3. Özel stopword ekleyin.\n"
                 "4. Parça boyutunu ayarlayın.",
        )

        # ── Metriklerin yorumu ─────────────────────────────────────
        with st.expander("📐 Bu Metrikler Ne Anlama Geliyor?"):
            coh_note = ""
            if coherence_val is not None:
                if coherence_val > 0.55:
                    coh_note = ("**Coherence sonucunuz ({:.3f}):** Yüksek. "
                                "Topic'leriniz muhtemelen anlamlı ve tutarlı. "
                                "Kelime listelerine bakarak etiketleme aşamasına geçebilirsiniz.").format(coherence_val)
                elif coherence_val > 0.40:
                    coh_note = ("**Coherence sonucunuz ({:.3f}):** Kabul edilebilir, "
                                "ama iyileştirme mümkün. Farklı K değerleri veya POS filtresi "
                                "deneyin. Özel stopword eklemeyi düşünün.").format(coherence_val)
                else:
                    coh_note = ("**Coherence sonucunuz ({:.3f}):** Düşük. "
                                "Topic'ler muhtemelen anlamlı değil. Şunları deneyin:\n"
                                "1. K'yı değiştirin (düşürün veya artırın)\n"
                                "2. POS filtresini daraltın (sadece NOUN)\n"
                                "3. Özel stopword ekleyin\n"
                                "4. Parça boyutunu ayarlayın").format(coherence_val)
            st.markdown(f"""
**Üç metriğin birlikte okunması:**

| Metrik | Ne ölçer? | İyi yönü |
|--------|-----------|----------|
| **Coherence (C_v)** | Topic'ler insana anlamlı mı? | Yüksek (> 0.55) |
| **Perplexity** | Model veriyi ne kadar iyi tahmin ediyor? | Düşük |
| **Log-likelihood** | Verinin model altında olasılığı | Sıfıra yakın |

**Hangisine bakmalıyım?** Öncelikli olarak **Coherence (C_v)** skoruna bakın.
Perplexity ve log-likelihood istatistiksel uyumu ölçer ama düşük perplexity
her zaman anlamlı topic'ler demek değildir (Chang et al., 2009).

{coh_note}

> Chang, J. et al. (2009). "Reading Tea Leaves: How Humans Interpret Topic Models."
> *NeurIPS.*
""")

        # ── Topic bar chart + word cloud ─────────────────────────
        st.subheader(
            f"Konular (K={actual_topics})",
            help="Her topic, o konuyu en çok temsil eden kelimelerin ağırlıklı listesidir.\n\n"
                 "• Kelimelerine bakarak topic'e etiket vermeye çalışın.\n\n"
                 "• Benzer kelimeler içeren topic'ler varsa K'yı düşürün.\n\n"
                 "• Bir topic çok farklı temaları karıştırıyorsa K'yı artırın.",
        )

        with st.expander("🏷️ Topic'leri Nasıl Etiketleriz?"):
            st.markdown("""
**Etiketleme, topic modelling'in en önemli insani adımıdır.**

Model size kelime listeleri verir, ama bu kelimelerin ne anlama geldiğine
siz karar verirsiniz. Örnek:

| Topic Kelimeleri | Olası Etiket |
|-----------------|--------------|
| cuore, amore, donna, dolore, occhi | Aşk ve duygu |
| guerra, soldato, morte, sangue, nemico | Savaş ve şiddet |
| casa, famiglia, madre, figlio, porta | Aile ve ev |
| mare, cielo, sole, terra, vento | Doğa ve manzara |

**Etiketleme ipuçları:**
1. İlk 5 kelimeye odaklanın — bunlar topic'in "çekirdeğidir".
2. Kelime bulutu'nda büyük görünen kelimelere bakın.
3. Birden fazla tema görüyorsanız → K'yı artırmayı deneyin.
4. Etiket bulamıyorsanız → topic muhtemelen anlamsız ("junk topic").
   Bu durumda özel stopword ekleyin veya K'yı değiştirin.

**Akademik raporlamada:** Topic etiketleri araştırmacının yorumudur.
"Topic 0: Aşk ve duygu (cuore, amore, donna, dolore, occhi)"
şeklinde hem etiket hem kelimeler birlikte verilir.
""")

        for idx, topic_dist in enumerate(lda.components_):
            top_indices = topic_dist.argsort()[: -n_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]
            weights = [topic_dist[i] for i in top_indices]

            st.markdown(f"**Topic {idx}**")
            col_bar, col_wc = st.columns([3, 2])

            with col_bar:
                bar_df = pd.DataFrame({"Kelime": top_words, "Ağırlık": weights})
                st.bar_chart(bar_df, x="Kelime", y="Ağırlık", height=220)

            with col_wc:
                wc_indices = topic_dist.argsort()[:-51:-1]
                ww = {feature_names[i]: float(topic_dist[i]) for i in wc_indices}
                fig_wc = generate_wordcloud(ww)
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)

        # ── pyLDAvis ─────────────────────────────────────────────
        st.subheader("🗺️ pyLDAvis İnteraktif Harita")
        with st.expander("ℹ️ pyLDAvis nasıl yorumlanır?", expanded=True):
            st.markdown("""
**pyLDAvis**, Sievert ve Shirley (2014) tarafından geliştirilen interaktif
topic model görselleştirme aracıdır. İki panelden oluşur:

---

**Sol panel — Topic daireleri (Intertopic Distance Map):**
- Dairenin **büyüklüğü** = topic'in korpustaki genel ağırlığı (sıklığı).
  Büyük daire = o konu metinlerde daha baskın.
- Daireler arası **mesafe** = topic'ler arasındaki benzerlik.
  - **Birbirinden uzak daireler** → topic'ler birbirinden ayrışıyor (iyi!).
  - **Üst üste binen daireler** → topic'ler benzer kelimeler paylaşıyor.
    Bu durumda **K'yı düşürmeyi** deneyin.
- Koordinatlar PCA (Principal Component Analysis) ile hesaplanır;
  yüksek boyutlu topic-kelime dağılımlarını 2 boyuta indirger.

**Sağ panel — Kelime çubukları:**
- Bir daireye **tıklayın** → o topic'in en ağırlıklı kelimeleri görünür.
- **Kırmızı çubuk** = kelimenin **o topic'teki** sıklığı (topic-spesifik).
- **Mavi çubuk** = kelimenin **genel korpustaki** sıklığı (overall).
- **Kırmızı ≫ mavi** → o kelime bu topic'e **özgüdür** (ayırt edici).
- **Kırmızı ≈ mavi** → kelime her yerde geçiyor, topic'e özgü değil.
  Bu tür kelimeleri özel stopword'e eklemeyi düşünün.

---

**λ (lambda) slider'ı — en önemli kontrol:**

| λ değeri | Ne gösterir? | Ne zaman? |
|----------|-------------|-----------|
| **λ = 1** | Mutlak sıklığa göre sıralar | Topic'in en sık kelimelerini görmek için |
| **λ = 0** | Ayırt ediciliğe göre sıralar | Topic'e özgü kelimeleri bulmak için |
| **λ ≈ 0.6** | Sıklık ve ayırt edicilik dengesi | **Önerilen başlangıç** (Sievert & Shirley, 2014) |

**Pratik yöntem:** λ'yı 1'den 0'a doğru kaydırın. Kelimelerin sıralaması değişecektir.
λ=1'de üst sıradaki genel kelimeler, λ=0'a yaklaştıkça yerini o topic'e özgü
kelimelere bırakır. Bu değişimi gözlemlemek topic'in karakterini anlamanızı sağlar.

> Sievert, C. & Shirley, K. (2014). "LDAvis: A method for visualizing and
> interpreting topics." *ACL Workshop on Interactive Language Learning,
> Visualization, and Interfaces.*
""")

        with st.spinner("pyLDAvis hazırlanıyor…"):
            vis_html = render_pyldavis(lda, dtm, vectorizer)
        if vis_html:
            st.components.v1.html(vis_html, height=800, scrolling=True)
        else:
            st.info("pyLDAvis yüklenemedi. `pip install pyLDAvis`")

        # ── Isı haritası ─────────────────────────────────────────
        st.subheader(
            "Isı Haritası",
            help="Her hücre, bir metin parçasının belirli bir topic ile ilişki olasılığını gösterir.\n\n"
                 "• **Koyu mavi** = yüksek olasılık (güçlü ilişki).\n\n"
                 "• **Açık mavi / beyaz** = düşük olasılık (zayıf ilişki).\n\n"
                 "**Nasıl yorumlanır?**\n\n"
                 "• Bir satırda **tek koyu hücre** → o parça net bir konuya ait (monotematik).\n\n"
                 "• Bir satırda **birden fazla koyu hücre** → o parça birden fazla konu içeriyor (çoklu tema). "
                 "Bu, edebiyat metinlerinde normaldir — bir paragraf hem aşk hem doğa teması taşıyabilir.\n\n"
                 "• Bir **sütun tamamen koyu** → o topic metinlerin tümüne yayılmış (dominant topic). "
                 "Çok baskın bir topic varsa, o topic'in kelimelerini stopword'e eklemeyi düşünün.\n\n"
                 "**Akademik kullanım:** Isı haritası, metnin bölümleri boyunca tematik dönüşümü "
                 "görselleştirmek için kullanılır. Roman analizlerinde bölümler arası tema geçişleri "
                 "bu haritadan okunabilir.",
        )
        heat_data = df_dist.melt(
            id_vars="Parça", var_name="Topic", value_name="Olasılık"
        )
        heatmap = (
            alt.Chart(heat_data)
            .mark_rect()
            .encode(
                x=alt.X("Topic:N", title="Topic"),
                y=alt.Y("Parça:N", title="Parça", sort=None),
                color=alt.Color("Olasılık:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Parça", "Topic", alt.Tooltip("Olasılık:Q", format=".4f")],
            )
            .properties(height=max(250, n_chunks * 22))
        )
        st.altair_chart(heatmap, use_container_width=True)

        # ── Dağılım tablosu ──────────────────────────────────────
        st.subheader(
            "Parça — Topic Dağılımı",
            help="Her satır bir metin parçasının topic olasılık dağılımını gösterir.\n\n"
                 "• Her satırın toplamı **1.0** (= %100)'dir.\n\n"
                 "• **0.82** = o parçanın %82'si bu konuyla ilişkili.\n\n"
                 "• **0.05** = neredeyse hiç ilişki yok.\n\n"
                 "Bu tablo, ısı haritasının sayısal karşılığıdır. "
                 "CSV olarak indirip kendi analizlerinizde (ör. R veya Excel) kullanabilirsiniz.",
        )
        st.dataframe(
            df_dist.style.format({c: "{:.4f}" for c in topic_cols}),
            use_container_width=True,
        )

        # ── Baskın topic ─────────────────────────────────────────
        st.subheader(
            "Baskın Topic (Parça Başına)",
            help="Her parça için en yüksek olasılığa sahip topic ve güven skoru.\n\n"
                 "**Güven skoru yorumlama:**\n\n"
                 "| Güven | Yorum |\n"
                 "|-------|-------|\n"
                 "| **> 0.70** | Parça net bir konuya odaklı (monotematik) |\n"
                 "| **0.40 – 0.70** | Çoklu tema var ama bir tanesi baskın |\n"
                 "| **< 0.40** | Parça tematik olarak dağınık — birden fazla konu eşit ağırlıkta |\n\n"
                 "**Edebiyat analizi için:** Düşük güven skorları her zaman kötü değildir. "
                 "Bir roman bölümü hem savaş hem aşk temasını eşit ağırlıkta işleyebilir. "
                 "Önemli olan, bu dağılımın metnin içeriğiyle tutarlı olup olmadığıdır — "
                 "yakın okuma (*close reading*) ile doğrulayın.",
        )
        dominant = df_dist.copy()
        dominant["Baskın Topic"] = doc_topic.argmax(axis=1)
        dominant["Güven"] = doc_topic.max(axis=1)
        st.dataframe(
            dominant[["Parça", "Baskın Topic", "Güven"]].style.format({"Güven": "{:.4f}"}),
            use_container_width=True,
        )

        # ── Akademik raporlama rehberi ─────────────────────────────
        with st.expander("📝 Sonuçları Akademik Makalede Nasıl Raporlarım?"):
            st.markdown(f"""
**Bir DH makalesinde topic modelling sonuçlarını raporlarken şu bilgileri mutlaka verin:**

**1. Veri ve ön-işleme:**
- Korpus büyüklüğü: {len(filenames)} belge, {n_chunks} parça
- Dil modeli: `{spacy_model}`
- POS filtresi: {', '.join(pos_tags)}
- Özel stopword: {', '.join(sorted(custom_stopwords)) if custom_stopwords else 'yok'}
- Parça boyutu: {chunk_size} kelime
- Sözlük boyutu: {len(feature_names)} terim

**2. Model parametreleri:**
- Algoritma: LDA (scikit-learn LatentDirichletAllocation)
- K (topic sayısı): {actual_topics}
- Öğrenme yöntemi: batch
- Iterasyon: {max_iter}
- Seed: {SEED}

**3. Değerlendirme metrikleri:**
- Coherence (C_v): {f"{coherence_val:.4f}" if coherence_val else "hesaplanamadı"}
- Perplexity: {perplexity_val:.4f}
- Log-likelihood: {log_likelihood_val:.4f}

**4. Farklı K denemeleri:**
Akademik standartta en az 3 farklı K denenmeli ve sonuçları karşılaştırılmalıdır.
Neden bu K'yı seçtiğinizi gerekçelendirin (coherence skoru + insani değerlendirme).

**5. Topic etiketleri:**
Her topic'e araştırmacı olarak verdiğiniz etiketler ve gerekçeler.

**6. Tekrar edilebilirlik:**
"Tüm analizler seed=42 ile yapılmıştır. Paket sürümleri ve ortam bilgisi
indirilebilir çıktılarda mevcuttur."

**Örnek paragraf:**
> "Korpus, LDA topic modelling ile analiz edilmiştir (K={actual_topics}, batch learning,
> seed=42). Ön-işleme aşamasında spaCy `{spacy_model}` modeli ile lemmatizasyon
> yapılmış, yalnızca {', '.join(pos_tags)} sözcük türleri dahil edilmiştir.
> Coherence (C_v) skoru {f'{coherence_val:.3f}' if coherence_val else '—'} olarak
> hesaplanmıştır (Röder et al., 2015)."
""")

        # ── Ortam & parametre bilgisi ────────────────────────────
        with st.expander("🖥️ Ortam Bilgisi"):
            st.code(env_txt, language="text")
        with st.expander("⚙️ Model Parametreleri"):
            st.code(params_txt, language="text")

        # ── İndirme ──────────────────────────────────────────────
        st.divider()
        st.subheader("📥 Çıktıları İndir")

        csv_buf = io.StringIO()
        df_dist.to_csv(csv_buf, index=False)

        # Derlem istatistikleri raporu
        corpus_stats_lines = [
            "═══ DERLEM İSTATİSTİKLERİ ═══",
            f"Ham sözcük sayısı       : {corpus_stats['raw_word_count']:,}",
            f"Ham karakter sayısı     : {corpus_stats['raw_char_count']:,}",
            f"Temiz token sayısı      : {corpus_stats['clean_token_count']:,}",
            f"Benzersiz kelime (type) : {corpus_stats['clean_type_count']:,}",
            f"Type-Token Ratio (TTR)  : {corpus_stats['ttr']:.4f}",
            f"Hapax legomena          : {corpus_stats['hapax_count']:,}",
            f"Ort. parça uzunluğu     : {corpus_stats['avg_chunk_len']:.0f} token",
            f"Filtreleme oranı        : {(1 - corpus_stats['clean_token_count'] / corpus_stats['raw_word_count']) * 100:.1f}%",
            "",
            "═══ EN SIK 50 KELİME ═══",
        ]
        for rank, (word, count) in enumerate(corpus_stats["top_50"], 1):
            corpus_stats_lines.append(f"{rank:>3}. {word:<25} {count:>6}")
        corpus_stats_txt = "\n".join(corpus_stats_lines)

        # Frekans CSV
        freq_csv_buf = io.StringIO()
        freq_df_all = pd.DataFrame(
            sorted(corpus_stats["freq"].items(), key=lambda x: -x[1]),
            columns=["Kelime", "Frekans"],
        )
        freq_df_all.to_csv(freq_csv_buf, index=False)

        output_files = {
            "topics.txt": topics_txt,
            "doc_topic_distribution.csv": csv_buf.getvalue(),
            "corpus_statistics.txt": corpus_stats_txt,
            "word_frequencies.csv": freq_csv_buf.getvalue(),
            "metrics.txt": metrics_txt,
            "model_parameters.txt": params_txt,
            "environment_report.txt": env_txt,
        }

        zip_bytes = create_zip(output_files)
        st.download_button(
            label="⬇ Tüm çıktıları indir (ZIP)",
            data=zip_bytes,
            file_name="lda_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )

        with st.expander("Dosyaları ayrı ayrı indir"):
            for fname, content in output_files.items():
                st.download_button(
                    label=fname,
                    data=content,
                    file_name=fname,
                    mime="text/plain" if fname.endswith(".txt") else "text/csv",
                )

else:
    # Hoşgeldin ekranı
    st.markdown("""
### Nasıl Kullanılır

1. Sol panelden **dil**, **POS filtresi** ve **parametreleri** ayarlayın.
2. Yukarıdaki kutuya `.txt`, `.odt` veya `.pdf` dosyalarınızı sürükleyin.
3. **▶ Analizi Başlat** düğmesine tıklayın.
4. Sonuçları ekranda inceleyin, ZIP olarak indirin.

| Adım | Ne olur? |
|------|----------|
| **Ön-işleme** | Metin → token → POS filtre → lemma → temiz kelime listesi |
| **Parçalama** | Tek uzun metin otomatik parçalara bölünür |
| **Vektörizasyon** | Kelimeler → sayısal matris (CountVectorizer) |
| **LDA** | Gizli konular (topics) hesaplanır |
| **Görselleştirme** | Bar chart, kelime bulutu, ısı haritası, pyLDAvis |
""")

    st.info(
        "💡 Tek kitap da yüklenebilir. Metin otomatik olarak parçalara "
        "bölünür ve her parça bir belge gibi analiz edilir."
    )

    with st.expander("🎓 Parametre Rehberi: Nereden Başlamalı?"):
        st.markdown("""
**1. Konu sayısı (K) — en kritik karar:**

| Durum | Öneri | Gerekçe |
|---|---|---|
| Kısa metin (< 5.000 kelime) | K = 2–3 | Az veri, fazla topic anlamsız olur |
| Orta metin (5.000–50.000 kelime) | K = 3–7 | Standart aralık |
| Uzun metin / birden fazla eser | K = 5–12 | Daha fazla tematik çeşitlilik beklenir |
| Büyük korpus (10+ eser) | K = 8–15 | Geniş tematik yelpaze |

**Akademik yöntem:** En az **3 farklı K** deneyin (ör. 3, 5, 7) ve Coherence
skorlarını karşılaştırın. En yüksek C_v → en anlamlı topic yapısı. Bunu
insani değerlendirme ile doğrulayın: "Topic kelimelerine bakarak anlamlı
bir etiket verebiliyor muyum?"

**2. POS Filtresi — akademik tercihler:**

| Konfigürasyon | Ne zaman? | Akademik referans |
|---|---|---|
| Sadece **NOUN** | İlk deney, en temiz sonuçlar | Schofield & Mimno (2016) |
| **NOUN + ADJ** | Duygusal/tanımlayıcı temalar gerektiğinde | Sentiment-yakın analiz |
| **NOUN + ADJ + VERB** | Anlatı ve eylem analizi | Narratoloji çalışmaları |
| **NOUN + PROPN** | Karakter/yer adı analizi | NER tarzı çalışmalar |

**Öneri:** Sadece NOUN ile başlayın. Her POS eklediğinizde coherence'ın nasıl
değiştiğini gözlemleyin.

**3. Parça boyutu — veri yapısına göre:**

| Boyut | Ne zaman? | Dikkat |
|---|---|---|
| 50–100 kelime | Şiir, kısa metin, paragraf analizi | Gürültülü olabilir |
| 150–250 kelime | Roman, uzun deneme (**önerilen**) | Denge noktası |
| 300–500 kelime | Çok uzun metin, genel tema taraması | İnce ayrımlar kaybolur |

**4. İterasyon:**
- **20** (varsayılan) çoğu durum için yeterlidir.
- Sonuçlar 20 ile 30 arasında değişmiyorsa model yakınsamıştır.
- Artırmak çalışma süresini uzatır ama sonucu iyileştirmeyebilir.

**5. Değerlendirme kontrol listesi:**
- [ ] En az 3 farklı K deneyin (3, 5, 7)
- [ ] Her denemede Coherence (C_v) skorunu kaydedin
- [ ] En yüksek coherence veren K'yı seçin
- [ ] Topic kelimelerini okuyun: anlamlı temalar oluşuyor mu?
- [ ] Anlamsız/tekrar eden kelimeler varsa özel stopword'e ekleyin
- [ ] pyLDAvis'te dairelerin birbirinden ayrışıp ayrışmadığını kontrol edin
- [ ] POS filtresini değiştirip etkisini gözlemleyin
- [ ] Sonuçları close reading ile doğrulayın
""")

    with st.expander("🧪 Deney Tasarımı: Adım Adım Analiz Yöntemi"):
        st.markdown("""
**Bir edebiyat metni üzerinde sistematik LDA analizi nasıl yapılır?**

```
AŞAMA 1: Keşif (Exploratory)
├─ Varsayılan parametrelerle ilk çalıştırma
├─ Topic kelimelerine göz atma
├─ Anlamsız kelimeleri not alma
└─ Genel izlenim edinme

AŞAMA 2: Parametre Optimizasyonu
├─ K = 3 → Coherence skorunu kaydet
├─ K = 5 → Coherence skorunu kaydet
├─ K = 7 → Coherence skorunu kaydet
├─ K = 10 → Coherence skorunu kaydet
├─ En iyi K'yı seç (yüksek coherence + anlamlı topic'ler)
├─ Özel stopword ekle → tekrar çalıştır
└─ POS filtresini dene (NOUN → NOUN+ADJ → NOUN+ADJ+VERB)

AŞAMA 3: Yorumlama
├─ Her topic'e etiket ver
├─ pyLDAvis ile topic ilişkilerini incele
├─ Isı haritasında tematik dönüşümleri gözlemle
├─ Baskın topic tablosu ile metin yapısını analiz et
└─ Close reading ile doğrula

AŞAMA 4: Raporlama
├─ Tüm parametreleri belge
├─ Farklı K denemelerini raporla
├─ Seçim gerekçelerini açıkla
├─ Görselleştirmeleri ekle
└─ Tekrar edilebilirlik bilgisini ver (seed, paket sürümleri)
```

**Önemli:** Her aşamada sonuçları **not alın**. "K=5'te coherence 0.52,
K=7'de 0.48 — K=5 tercih edildi çünkü topic'ler daha anlamlı etiketlenebiliyordu."
Bu tür notlar akademik makalenizin metodoloji bölümünü oluşturur.
""")

    with st.expander("📚 Akademik Kaynaklar"):
        st.markdown("""
**Topic Modelling temel kaynakları:**

| Kaynak | Katkısı |
|--------|---------|
| Blei, Ng & Jordan (2003). "Latent Dirichlet Allocation." *JMLR* | LDA'nın orijinal makalesi |
| Blei (2012). "Probabilistic Topic Models." *CACM* | Erişilebilir genel bakış |
| Röder, Both & Hinneburg (2015). "Exploring the Space of Topic Coherence Measures." *WSDM* | C_v coherence metriği |
| Chang et al. (2009). "Reading Tea Leaves." *NeurIPS* | Perplexity ≠ insani yorum |
| Sievert & Shirley (2014). "LDAvis." *ACL Workshop* | pyLDAvis görselleştirmesi |
| Schofield & Mimno (2016). "Comparing Apples to Apple." *ACL Workshop* | POS filtreleme etkisi |
| Schofield, Magnusson & Mimno (2017). "Pulling Out the Stops." *EACL* | Stopword etkisi |

**Dijital Beşeri Bilimler kaynakları:**

| Kaynak | Katkısı |
|--------|---------|
| Moretti (2013). *Distant Reading*. Verso | Uzak okuma kavramı |
| Jockers (2013). *Macroanalysis*. UIUC Press | Edebiyatta hesaplamalı yöntemler |
| Underwood (2019). *Distant Horizons*. Chicago UP | Topic modelling + edebiyat tarihi |
| Graham, Weingart & Milligan (2015). *Exploring Big Historical Data*. Imperial College Press | DH'da topic modelling pratikleri |

**Faydalı çevrimiçi kaynaklar:**
- [Programming Historian — Topic Modeling](https://programminghistorian.org/) — adım adım rehber
- [DARIAH-DE Topic Modelling](https://dariah-de.github.io/Topics/) — DH odaklı araç
""")

    with st.expander("🔑 Temel Kavramlar Sözlüğü"):
        st.markdown("""
| Kavram | Açıklama |
|--------|----------|
| **Topic** | LDA'nın bulduğu gizli konu — kelimelerin olasılık dağılımı |
| **Lemma** | Kelimenin sözlük formu: "andavo" → "andare" |
| **Tokenizasyon** | Metni kelimelere ayırma işlemi |
| **POS (Part-of-Speech)** | Sözcük türü: isim, fiil, sıfat vb. |
| **Stopword** | Çok sık ama anlamsız kelimeler: "il", "di", "che" |
| **Coherence (C_v)** | Topic'lerin insani anlamlılığını ölçen metrik (0–1) |
| **Perplexity** | Modelin istatistiksel uyumunu ölçen metrik (düşük = iyi) |
| **Bag-of-Words (BoW)** | Kelime sırasını göz ardı eden temsil yöntemi |
| **DTM (Document-Term Matrix)** | Satır = belge, sütun = kelime, değer = sayı |
| **min_df / max_df** | Çok nadir veya çok yaygın kelimeleri filtreleme eşikleri |
| **Batch learning** | Tüm veriyi her iterasyonda işleyen öğrenme yöntemi |
| **Seed (random_state)** | Rastgeleliği sabitleyen sayı — tekrar edilebilirlik için |
| **Distant reading** | Metni yakından okumadan hesaplamalı yöntemlerle analiz etme |
| **Close reading** | Metni dikkatli, yakından okuma — DH'da distant reading'i doğrular |
| **Pseudo-belge** | Tek bir metnin parçalara bölünmesiyle oluşan yapay belge birimi |
| **λ (lambda)** | pyLDAvis'te sıklık ve ayırt edicilik arasındaki denge kontrolü |
""")
