# 🧠 Implementasi Arsitektur Transformer dari Nol dengan NumPy

Repositori ini berisi implementasi forward pass dari arsitektur **decoder-only Transformer (GPT-style)** yang dibangun sepenuhnya dari nol menggunakan NumPy.  

---

## ⚙️ Fitur

Implementasi ini mencakup semua komponen wajib dari arsitektur Transformer, yaitu:

- ✅ **Token Embedding**  
- ✅ **Positional Encoding (Sinusoidal dan RoPE)**  
- ✅ **Scaled Dot-Product Attention dengan Causal Masking**  
- ✅ **Multi-Head Attention**  
- ✅ **Feed-Forward Network**  
- ✅ **Residual Connection dengan Layer Normalization (Pre-Norm)**  
- ✅ **Lapisan Output dengan Softmax**

Selain itu, implementasi ini juga menyertakan beberapa **fitur bonus** untuk meningkatkan performa dan analisis:

- 🌀 **Rotary Positional Embedding (RoPE)** — alternatif positional encoding yang lebih modern.  
- 🔗 **Weight Tying** — antara lapisan embedding dan lapisan output untuk efisiensi parameter.  
- 🔍 **Visualisasi Attention Map** — menggunakan Matplotlib dan Seaborn untuk menganalisis bagaimana model memfokuskan perhatiannya.

---

## 🧩 Dependensi

Proyek ini hanya memerlukan beberapa library Python standar.  
Pastikan Anda telah menginstal:

- `numpy` — untuk semua operasi matematis dan matriks  
- `matplotlib` & `seaborn` — untuk visualisasi attention heatmap

---

## ▶️ Cara Menjalankan Program 

1️⃣ Clone Repository

```bash
git clone https://github.com/namakamu/nama-repo.git
```

2️⃣ Masuk ke Direktori Proyek

```bash
git clone https://github.com/namakamu/nama-repo.git
```

3️⃣ Instal semua dependensi dengan menjalankan perintah berikut di terminal Anda:

```bash
pip install numpy matplotlib seaborn
```

4️⃣ Jalankan Skrip Python

```bash
python transformer.py
```

