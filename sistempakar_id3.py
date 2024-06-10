import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from imagetree import show_tree_image

# Membaca data dari CSV
df = pd.read_csv('data.csv')

# Mengonversi fitur kategori ke numerik
le_nama = LabelEncoder()
le_jurusan_sma = LabelEncoder()
le_jurusan_pilihan = LabelEncoder()
le_daerah_asal = LabelEncoder()
le_prioritas = LabelEncoder()
le_hasil = LabelEncoder()

df['Nama'] = le_nama.fit_transform(df['Nama'])
df['Jurusan_SMA'] = le_jurusan_sma.fit_transform(df['Jurusan_SMA'])
df['Jurusan_Pilihan'] = le_jurusan_pilihan.fit_transform(df['Jurusan_Pilihan'])
df['Daerah_Asal'] = le_daerah_asal.fit_transform(df['Daerah_Asal'])
df['Prioritas'] = le_prioritas.fit_transform(df['Prioritas'])
df['Hasil'] = le_hasil.fit_transform(df['Hasil'])

# Pisahkan fitur dan label
X = df[['Nama', 'Jurusan_SMA', 'Jurusan_Pilihan', 'Daerah_Asal', 'Rata-rata_NA', 'Prioritas', 'Skor_Ujian']]
y = df['Hasil']

# Inisialisasi dan latih DecisionTreeClassifier dengan kriteria 'entropy' (ID3)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Fungsi untuk mendapatkan input manual
def get_user_input():
    nama = input("Nama: ")
    jurusan_sma = input("Jurusan Sewaktu SMA (IPA/IPS): ")
    jurusan_pilihan = input("Jurusan Pilihan (IPA/IPS/Kejuruan): ")
    daerah_asal = input("Daerah Asal (Banten/DKI Jakarta/Jawa Barat/Jawa Tengah/Jawa Timur): ")
    rata_rata_na = float(input("Rata-rata Nilai Akhir: "))
    prioritas = input("Prioritas Pilihan Prodi (Pilihan 1/Pilihan 2/Pilihan 3): ")
    skor_ujian = int(input("Skor Ujian: "))

    # Konversi input pengguna ke numerik
    try:
        nama_num = le_nama.transform([nama])[0]
    except ValueError:
        print(f"Label {nama} tidak dikenali dalam Nama.")
        return None
    try:
        jurusan_sma_num = le_jurusan_sma.transform([jurusan_sma])[0]
    except ValueError:
        print(f"Label {jurusan_sma} tidak dikenali dalam Jurusan_SMA.")
        return None
    try:
        jurusan_pilihan_num = le_jurusan_pilihan.transform([jurusan_pilihan])[0]
    except ValueError:
        print(f"Label {jurusan_pilihan} tidak dikenali dalam Jurusan_Pilihan.")
        return None
    try:
        daerah_asal_num = le_daerah_asal.transform([daerah_asal])[0]
    except ValueError:
        print(f"Label {daerah_asal} tidak dikenali dalam Daerah_Asal.")
        return None
    try:
        prioritas_num = le_prioritas.transform([prioritas])[0]
    except ValueError:
        print(f"Label {prioritas} tidak dikenali dalam Prioritas.")
        return None

    return [[nama_num, jurusan_sma_num, jurusan_pilihan_num, daerah_asal_num, rata_rata_na, prioritas_num, skor_ujian]]

# Meminta input dari pengguna
user_input = get_user_input()
if user_input is not None:
    # Prediksi menggunakan model decision tree
    prediksi_num = clf.predict(user_input)
    prediksi = le_hasil.inverse_transform(prediksi_num)[0]

    print(f"Prediksi penerimaan: {prediksi}")

    # Tampilkan decision tree
    show_tree_image(clf, feature_names=['Nama', 'Jurusan_SMA', 'Jurusan_Pilihan', 'Daerah_Asal', 'Rata-rata_NA', 'Prioritas', 'Skor_Ujian'], class_names=le_hasil.classes_)
