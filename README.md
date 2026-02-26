# LDA Topic Modelling — Edebiyat Metinleri (DH Öğretim Aracı)

Lemma-tabanlı, deterministik ve tekrar edilebilir bir LDA topic modelling aracı.
Tarayıcıda çalışır; Python bilgisi gerektirmez.

**Desteklenen diller:** İtalyanca, İngilizce

---

## Canlı Uygulama (Öğrenciler İçin)

Uygulamaya doğrudan erişmek için:

👉 **https://lda-topic-modelling-ita.streamlit.app** *(deploy sonrası aktif olacak)*

Kurulum gerekmez. Tarayıcıdan açın, dosya yükleyin, sonuçları indirin.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| **Çoklu dosya formatı** | `.txt`, `.odt` ve `.pdf` dosyaları desteklenir |
| **Tek / çoklu belge** | Tek kitap yüklendiğinde otomatik parçalama yapılır |
| **POS filtresi** | İsim, Fiil, Sıfat, Zarf, Özel İsim — seçilebilir |
| **Özel stopword** | Kullanıcı ek stopword ekleyebilir |
| **Coherence (C_v)** | Topic kalitesini ölçen standart DH metriği |
| **Kelime bulutu** | Her topic için görsel kelime bulutu |
| **pyLDAvis** | İnteraktif topic haritası |
| **Isı haritası** | Belge × topic olasılık matrisi |
| **Pedagojik arayüz** | Tüm parametrelerde ℹ️ açıklamaları, ön-işleme adım tablosu |
| **Deterministik** | seed=42, batch learning → aynı veri = aynı sonuç |
| **ZIP çıktı** | Tüm sonuçlar tek tıkla indirilir |

---

## Deploy: Uygulamayı Yayınlama (Eğitimci İçin)

### Adım 1 — GitHub Hesabı

[github.com](https://github.com) adresinde ücretsiz bir hesap açın (varsa atlayın).

### Adım 2 — Yeni Repo Oluşturun

1. GitHub'da sağ üstte **"+"** → **"New repository"** tıklayın.
2. Repo adı: `lda-topic-modelling-ita` (veya istediğiniz bir isim).
3. **Public** seçin (Streamlit Cloud ücretsiz plan için gerekli).
4. **"Create repository"** tıklayın.

### Adım 3 — Dosyaları Yükleyin

Repo sayfasında **"uploading an existing file"** bağlantısına tıklayın ve şu dosyaları sürükleyin:

```
run_topic_model.py
requirements.txt
.streamlit/config.toml     ← klasör yapısıyla birlikte
```

> **Not:** `.streamlit` klasörünü yüklemek için: bilgisayarınızda bu klasörü içeren
> tüm proje dosyalarını ZIP yapıp GitHub'da "Upload files" ile yükleyebilir,
> veya aşağıdaki terminal yöntemini kullanabilirsiniz.

**Terminal ile (isteğe bağlı):**

```bash
cd /proje/klasörünüz
git init
git add run_topic_model.py requirements.txt .streamlit/config.toml README.md
git commit -m "LDA topic modelling app"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADINIZ/lda-topic-modelling-ita.git
git push -u origin main
```

### Adım 4 — Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) adresine gidin.
2. **"Sign in with GitHub"** ile giriş yapın.
3. **"New app"** tıklayın.
4. Ayarları doldurun:
   - **Repository:** `KULLANICI_ADINIZ/lda-topic-modelling-ita`
   - **Branch:** `main`
   - **Main file path:** `run_topic_model.py`
5. **"Deploy!"** tıklayın.

İlk deploy 3–5 dakika sürer (spaCy modeli ve gensim indirilir). Tamamlanınca size şuna benzer bir URL verilir:

```
https://lda-topic-modelling-ita.streamlit.app
```

Bu bağlantıyı öğrencilerinizle paylaşın. Herkes tarayıcıdan erişir, kurulum gerekmez.

---

## Yerel Kullanım (İsteğe Bağlı)

### 1. Python Kurulumu

Python 3.9+ gerekir: [python.org/downloads](https://www.python.org/downloads/)

```bash
python3 --version
```

### 2. Bağımlılıkları Kurun

```bash
cd /dosyalarınızın/bulunduğu/klasör
pip install -r requirements.txt
```

spaCy İtalyanca modeli `requirements.txt` içinde tanımlıdır, otomatik kurulur.
İngilizce modeli seçildiğinde uygulama ilk çalıştırmada otomatik indirir.

### 3. Çalıştırın

```bash
streamlit run run_topic_model.py
```

Tarayıcı `http://localhost:8501` adresinde açılır.

---

## Kullanım

1. Sol panelden **dil**, **POS filtresi**, **özel stopword** ve **parametreleri** ayarlayın.
2. Sayfadaki kutuya `.txt`, `.odt` veya `.pdf` dosyalarınızı sürükleyin.
   - Tek dosya veya birden fazla dosya yüklenebilir.
   - Her dosya ayrı bir belge olarak değerlendirilir.
   - Tek dosya yüklendiğinde metin otomatik parçalara bölünür.
3. **▶ Analizi Başlat** düğmesine tıklayın.
4. Sonuçları ekranda inceleyin:
   - Ön-işleme adım tablosu (her kelime ne oldu?)
   - Topic bar chart + kelime bulutu
   - Coherence (C_v), Perplexity, Log-likelihood metrikleri
   - pyLDAvis interaktif harita
   - Isı haritası ve dağılım tablosu
5. **⬇ Tüm çıktıları indir (ZIP)** ile sonuçları kaydedin.

---

## Çıktılar

| Dosya | İçerik |
|---|---|
| `topics.txt` | Her topic için en ağırlıklı kelimeler |
| `doc_topic_distribution.csv` | Parça × topic olasılık matrisi |
| `metrics.txt` | Perplexity, Log-likelihood, Coherence (C_v) |
| `model_parameters.txt` | Ön-işleme ve model parametreleri |
| `environment_report.txt` | Python, paket sürümleri, seed bilgisi |

---

## Tekrar Edilebilirlik

Bu analiz deterministiktir:

- `random_state = 42` tüm rastgele süreçleri sabitler.
- `learning_method = 'batch'` online öğrenmedeki sıra etkisini ortadan kaldırır.
- Aynı veri + aynı seed + aynı parametreler = aynı sonuç.

Sonuçları doğrulamak için `environment_report.txt` dosyasındaki sürüm bilgilerini kullanın.

---

## Parametreler

### Ön-İşleme (spaCy)

| Parametre | Değer |
|---|---|
| Model | `it_core_news_sm` / `en_core_web_sm` |
| nlp.max_length | 2.000.000 karakter |
| Küçük harf | Evet |
| Noktalama çıkarma | Evet |
| Sayı çıkarma | Evet |
| Stopword | spaCy dil stopwords + özel liste |
| POS filtresi | NOUN, VERB, ADJ (varsayılan; ayarlanabilir) |
| Lemmatizasyon | Evet |
| Min. token uzunluğu | 3 |

### Vektörizasyon (CountVectorizer)

| Parametre | Değer |
|---|---|
| min_df | 2 (< 3 parça: 1) |
| max_df | 0.85 (< 3 parça: 1.0) |
| ngram_range | (1, 1) |

### Model (LDA)

| Parametre | Değer |
|---|---|
| n_components | 5 (varsayılan; 2–15 arası ayarlanabilir) |
| random_state | 42 |
| learning_method | batch |
| max_iter | 20 (varsayılan; 5–50 arası ayarlanabilir) |

### Değerlendirme Metrikleri

| Metrik | Açıklama |
|---|---|
| **Coherence (C_v)** | Topic'lerin insani yorumlanabilirliği. > 0.55 iyi, < 0.40 zayıf |
| **Perplexity** | Modelin metne uyumu. Düşük = daha iyi |
| **Log-likelihood** | Olasılık skoru. Sıfıra yakın = daha iyi |

---

## Sorun Giderme

**"ModuleNotFoundError"** → `pip install -r requirements.txt` komutunu tekrar çalıştırın.

**"Can't find model 'it_core_news_sm'"** → Uygulama ilk çalıştırmada otomatik indirir. Sorun devam ederse: `python -m spacy download it_core_news_sm`

**"PDF okunamadı"** → Dosyanın şifreli veya bozuk olmadığından emin olun. Metin tabanlı (taranmış olmayan) PDF'ler desteklenir.

**Boş sonuç** → Belgelerinizin seçili dilde metin içerdiğinden emin olun. POS filtresini genişletin veya parça boyutunu küçültün.

**Port meşgul** → `streamlit run run_topic_model.py --server.port 8502`

**Streamlit Cloud'da yavaş** → İlk deploy'da gensim ve spaCy modeli indirilir (3-5 dk). Sonraki erişimler daha hızlıdır.
