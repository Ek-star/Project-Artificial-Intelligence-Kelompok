import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Analisis Gambar Angka dan Operasi Hitung sebagai Media Perkembangan Anak menggunakan metode CNN
# Mengimpor library yang diperlukan, termasuk OpenCV (cv2) untuk manipulasi gambar, NumPy untuk operasi array, TensorFlow untuk pembuatan dan pelatihan model jaringan saraf, dan Matplotlib untuk menampilkan gambar.

print("Selamat datang di Pengenalan Angka Tulisan Tangan oleh Kelompok EkaVirMalaTia")

train_new_model = True

# Memutuskan apakah akan memuat model yang ada atau melatih model baru berdasarkan nilai variabel train_new_model.
if train_new_model:
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    class PrintMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch {}: Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(epoch + 1, logs['val_loss'], logs['val_accuracy']))

    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[PrintMetricsCallback()])

    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    model.save('handwritten_digits.model')
    # Jika train_new_model adalah True, maka model akan dilatih menggunakan dataset MNIST. Model yang dibangun memiliki dua lapisan tersembunyi dengan fungsi aktivasi ReLU dan satu lapisan output dengan fungsi aktivasi softmax. Model kemudian dikompilasi dengan optimizer 'adam' dan fungsi loss 'sparse_categorical_crossentropy'. Model yang dilatih dievaluasi pada set data uji, dan jika model baru, model tersebut disimpan sebagai file 'handwritten_digits.model'.

else:
    model = tf.keras.models.load_model('handwritten_digits.model')
    # Jika train_new_model adalah False, model yang sudah dilatih sebelumnya akan dimuat.

nomor_gambar = 1
while os.path.isfile('digits/digit{}.png'.format(nomor_gambar)):
    try:
        img1 = cv2.imread('digits/digit{}.png'.format(nomor_gambar))[:,:,0]
        img1 = np.invert(np.array([img1]))
        prediksi1 = model.predict(img1)
        angka_pertama = np.argmax(prediksi1)
        print("Angka pertama yang saya lihat adalah {}".format(angka_pertama))

        nomor_gambar_angka_kedua = nomor_gambar + 1
        img2 = cv2.imread('digits/digit{}.png'.format(nomor_gambar_angka_kedua))[:,:,0]
        img2 = np.invert(np.array([img2]))
        prediksi2 = model.predict(img2)
        angka_kedua = np.argmax(prediksi2)
        print("Angka kedua yang saya lihat adalah {}".format(angka_kedua))

        hasil_penambahan = angka_pertama + angka_kedua
        print("Hasil penambahan {} + {} adalah {}".format(angka_pertama, angka_kedua, hasil_penambahan))
        # Menggunakan loop while, program memproses gambar angka secara berurutan (digit1.png, digit2.png, dst.) dari direktori 'digits'. Setiap gambar dibaca menggunakan OpenCV, diproses, dan hasilnya diprediksi menggunakan model jaringan saraf. Hasil prediksi digunakan untuk menampilkan angka pertama dan kedua, dan hasil penjumlahannya.



        plt.imshow(img1[0], cmap=plt.cm.binary)
        plt.show()
        plt.imshow(img2[0], cmap=plt.cm.binary)
        plt.show()
        # Menampilkan gambar angka pertama dan angka kedua menggunakan Matplotlib. Gambar angka ditampilkan dalam skala abu-abu biner. Selanjutnya, hasil penjumlahan dari kedua angka ditampilkan.


        nomor_gambar += 2
        # Program melompat dua gambar untuk mengambil angka berikutnya. Hal ini dilakukan karena setiap iterasi loop while memproses dua gambar sekaligus.
    except:
        print("Error membaca gambar! Melanjutkan dengan gambar berikutnya...")
        nomor_gambar += 1
        # Menangani error yang mungkin terjadi saat membaca gambar. Program tetap melanjutkan ke gambar berikutnya jika terjadi error.

        class PrintMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.train_loss = []
                self.train_acc = []
                self.val_loss = []
                self.val_acc = []

            def on_epoch_end(self, epoch, logs=None):
                print(
                    "Epoch {}: Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(epoch + 1, logs['val_loss'],
                                                                                            logs['val_accuracy']))
                self.train_loss.append(logs['loss'])
                self.train_acc.append(logs['accuracy'])
                self.val_loss.append(logs['val_loss'])
                self.val_acc.append(logs['val_accuracy'])


        # Ganti callback saat memanggil model.fit
        print_metrics_callback = PrintMetricsCallback()
        model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[print_metrics_callback])

        # Setelah pelatihan selesai, gunakan nilai yang telah disimpan untuk membuat grafik
        epochs = range(1, len(print_metrics_callback.train_loss) + 1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, print_metrics_callback.train_loss, 'r', label='Training Loss')
        plt.plot(epochs, print_metrics_callback.val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, print_metrics_callback.train_acc, 'r', label='Training Accuracy')
        plt.plot(epochs, print_metrics_callback.val_acc, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()