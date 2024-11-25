import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Mengatur judul dan tata letak
st.set_page_config(
    page_title="Aplikasi Deteksi Tingkat Depresi",
    layout="centered"
)

# Membaca dataset dari file CSV
@st.cache_data
def load_data():
    df = pd.read_csv('Deepression.csv')
    return df

# Memuat data
df = load_data()

# Label encoding untuk kolom Depression State
label_encoder = LabelEncoder()
df['Depression State'] = label_encoder.fit_transform(df['Depression State'])

# Fitur dan label
X = df.drop(columns=["Number", "Depression State"])
y = df["Depression State"]

# Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=2, weights='distance')
knn.fit(X_train, y_train)

# Menilai akurasi
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Judul halaman
st.title("Aplikasi Deteksi Tingkat Depresi")
st.write("""
Selamat datang di aplikasi untuk mendeteksi tingkat depresi.

### Anggota Kelompok:
- **140810220044** - Candra Wibawa
- **140810220046** - Muhammad Adzikra Dhiya Alfauzan
- **140810220052** - Ivan Arsy Himawan
""")

# Menampilkan akurasi
st.write(f"Akurasi model KNN: {accuracy * 100:.2f}%")

# Tingkatan depresi
depression_levels = {
    "No depression": "Tidak ada tanda-tanda depresi.",
    "Mild": "Tanda-tanda depresi ringan. Disarankan untuk memperhatikan kesehatan mental Anda.",
    "Moderate": "Gejala depresi sedang. Sebaiknya konsultasikan ke ahli kesehatan mental.",
    "Severe": "Depresi berat. Segera cari bantuan dari profesional kesehatan mental."
}

# Form input untuk pengguna
st.subheader("Isi kuesioner berikut untuk mengetahui tingkat depresi Anda")

with st.form("depression_form"):
    sleep = st.radio("Seberapa sering Anda merasa memiliki kualitas tidur yang buruk?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    appetite = st.radio("Seberapa sering Anda kehilangan nafsu makan?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    interest = st.radio("Seberapa sering Anda merasa kurang tertarik pada kegiatan yang biasanya Anda nikmati?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    fatigue = st.radio("Seberapa sering Anda merasa sangat lelah atau kehabisan energi?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    worthlessness = st.radio("Seberapa sering Anda merasa tidak berharga?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    concentration = st.radio("Seberapa sering Anda merasa kesulitan berkonsentrasi?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    agitation = st.radio("Seberapa sering Anda merasa mudah tersinggung?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    suicidal = st.radio("Seberapa sering Anda berpikir untuk menyakiti diri sendiri?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    sleep_disturbance = st.radio("Seberapa sering Anda mengalami gangguan saat tidur?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    aggression = st.radio("Seberapa sering Anda merasa agresif atau mudah marah?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    panic = st.radio("Seberapa sering Anda mengalami serangan panik?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    hopelessness = st.radio("Seberapa sering Anda merasa kehilangan harapan?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    restlessness = st.radio("Seberapa sering Anda merasa gelisah atau tidak tenang?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    low_energy = st.radio("Seberapa sering Anda merasa kurang energi atau malas melakukan aktivitas?", ["Tidak Pernah", "Jarang", "Kadang-kadang", "Sering", "Selalu"])
    submitted = st.form_submit_button("Submit")

# Fungsi untuk memprediksi tingkat depresi  
if submitted:
    # Mengubah jawaban ke dalam format angka untuk pemodelan
    mapping = {"Tidak Pernah": 1, "Jarang": 2, "Kadang-kadang": 3, "Sering": 4, "Selalu": 5}
    input_data = [[mapping[sleep], mapping[appetite], mapping[interest], mapping[fatigue], mapping[worthlessness], 
                   mapping[concentration], mapping[agitation], mapping[suicidal], mapping[sleep_disturbance], 
                   mapping[aggression], mapping[panic], mapping[hopelessness], mapping[restlessness], mapping[low_energy]]]
    
    prediction = knn.predict(input_data)
    depression_state = label_encoder.inverse_transform(prediction)[0]
    level_description = depression_levels.get(depression_state, "Deskripsi tidak tersedia.")

    st.subheader("Hasil Anda:")
    st.write(f"Berdasarkan input Anda, tingkat depresi Anda adalah: **{depression_state}**")
    st.write(f"**Deskripsi:** {level_description}")
    st.info("Jika Anda merasa membutuhkan bantuan, segera konsultasikan dengan profesional kesehatan mental.")
    