import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Selamat datang di Pengenalan Angka Tulisan Tangan oleh Kelompok EkaVirMalaTia")

# Memutuskan apakah akan memuat model yang ada atau melatih model baru
train_new_model = True

if train_new_model:
    # Memuat set data MNIST dengan sampel dan membaginya
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalisasi data (membuat panjang = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Membuat model jaringan saraf
    # Menambahkan satu lapisan input yang diratakan untuk piksel
    # Menambahkan dua lapisan tersembunyi yang padat
    # Menambahkan satu lapisan output yang padat untuk 10 digit
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Mengompilasi dan mengoptimalkan model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Melatih model
    model.fit(X_train, y_train, epochs=3)

    # Mengevaluasi model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Menyimpan model
    model.save('handwritten_digits.model')
else:
    # Memuat model
    model = tf.keras.models.load_model('handwritten_digits.model')

# Memuat gambar kustom dan memprediksinya
nomor_gambar = 1
while os.path.isfile('digits/digit{}.png'.format(nomor_gambar)):
    try:
        # Membaca gambar untuk angka pertama
        img1 = cv2.imread('digits/digit{}.png'.format(nomor_gambar))[:,:,0]
        img1 = np.invert(np.array([img1]))
        prediksi1 = model.predict(img1)
        angka_pertama = np.argmax(prediksi1)
        print("Angka pertama yang saya lihat adalah {}".format(angka_pertama))

        # Membaca gambar untuk angka kedua dari file lainnya
        nomor_gambar_angka_kedua = nomor_gambar + 1
        img2 = cv2.imread('digits/digit{}.png'.format(nomor_gambar_angka_kedua))[:,:,0]
        img2 = np.invert(np.array([img2]))
        prediksi2 = model.predict(img2)
        angka_kedua = np.argmax(prediksi2)
        print("Angka kedua yang saya lihat adalah {}".format(angka_kedua))

        # Menambahkan program untuk menambahkan dua bilangan
        hasil_penambahan = angka_pertama + angka_kedua
        print("Hasil penambahan {} + {} adalah {}".format(angka_pertama, angka_kedua, hasil_penambahan))

        # Untuk menampilkan hasil dari penjumlahan bilangan satu dengan lainnya
        plt.imshow(img1[0], cmap=plt.cm.binary)
        plt.show()
        plt.imshow(img2[0], cmap=plt.cm.binary)
        plt.show()
        nomor_gambar += 2  # Melompat dua gambar untuk mengambil angka berikutnya
    except:
        print("Error membaca gambar! Melanjutkan dengan gambar berikutnya...")
        nomor_gambar += 1
