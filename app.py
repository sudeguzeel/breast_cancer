import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
# SAYFA AYARLARI

st.set_page_config(
    page_title="Meme Kanseri Tespiti",
    layout="centered"
)


# RENK TASARIMI


st.markdown("""
<style>
h1, h2, h3 {
    color: #355872;
    text-align: center;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.benign {
    background-color: #2E7D32;
    color: white;
}
.malignant {
    background-color: #C62828;
    color: white;
}
</style>
""", unsafe_allow_html=True)



# MODEL YÜKLE

@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()


# MENÜ

menu = st.sidebar.selectbox(
    "Menü",
    ["Tahmin", "Analiz", "Hakkında"]
)

# 1. PREDICTION SAYFASI

if menu == "Tahmin":

    st.title("Meme Kanseri Tespiti")

    uploaded_file = st.file_uploader("Ultrason resmi yükleyin", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # preprocessing
        img = image.resize((128, 128))
        img = np.array(img) / 255.0
        img = img.reshape(1, 128, 128, 3)

        prediction = model.predict(img)[0][0]

        confidence = float(prediction) * 100

        if prediction > 0.5:
            st.markdown(f'<div class="result-box malignant">MALIGNANt<br>Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box benign">BENIGN<br>Confidence: {100-confidence:.2f}%</div>', unsafe_allow_html=True)


# 2. ANALYSIS SAYFASI

elif menu == "Analiz":

    st.title("Model Analizi")

    st.subheader("Training Performance")
    if os.path.exists("accuracy_loss.png"):
        st.image("accuracy_loss.png")

    st.subheader("Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png")

    st.subheader("ROC Curve")
    if os.path.exists("roc_curve.png"):
        st.image("roc_curve.png")


# 3. ABOUT SAYFASI

else:

    st.title("Proje Hakkında")

    st.write("""
PROBLEM VE PROJE ÖZETİ
Bu çalışmada ele alınan problem, meme ultrason görüntülerinin makine öğrenmesi yöntemleri kullanılarak sınıflandırılmasıdır. Meme kanseri teşhisi, özellikle ultrason görüntülerinde uzmanlık gerektiren ve hata payı içerebilen bir süreçtir. Bu nedenle, görüntü işleme ve derin öğrenme teknikleri kullanılarak bu sürecin otomatik hale getirilmesi hedeflenmektedir.
Bu kapsamda, ultrason görüntülerinden elde edilen veriler kullanılarak görüntülerin benign (iyi huylu) ve malignant (kötü huylu) olarak sınıflandırılması amaçlanmaktadır. Problem, bir görüntü sınıflandırma problemi olup, giriş verisi olarak görüntüler alınmakta ve çıkış olarak bu görüntünün ait olduğu sınıf tahmin edilmektedir.
Elde edilen sonuçlara göre modelin test verisi üzerindeki doğruluk oranı yaklaşık %95.9 olarak hesaplanmıştır. Bu değer, modelin meme ultrason görüntülerini başarılı bir şekilde sınıflandırabildiğini göstermektedir.

PROJENİN AMACI
Bu projenin amacı, meme ultrason görüntülerini kullanarak derin öğrenme tabanlı bir sınıflandırma modeli geliştirmektir. Geliştirilen modelin hedefi, verilen bir ultrason görüntüsünü analiz ederek görüntünün benign (iyi huylu) veya malignant (kötü huylu) olduğunu doğru bir şekilde tahmin etmektir.
Bu doğrultuda, görüntü işleme ve konvolüsyonel sinir ağları (CNN) kullanılarak otomatik bir karar destek sistemi oluşturulması amaçlanmaktadır. Proje kapsamında elde edilen modelin, tıbbi görüntü analizinde uzmanlara yardımcı olabilecek bir araç olarak kullanılması hedeflenmektedir.

ÇALIŞMANIN ÖNEMİ VE KULLANIM ALANLARI

Bu projenin amacı, meme ultrason görüntülerini kullanarak derin öğrenme tabanlı bir sınıflandırma modeli geliştirmektir. Geliştirilen modelin hedefi, verilen bir ultrason görüntüsünü analiz ederek görüntünün benign (iyi huylu) veya malignant (kötü huylu) olduğunu doğru bir şekilde tahmin etmektir.
Bu doğrultuda, görüntü işleme ve konvolüsyonel sinir ağları (CNN) kullanılarak otomatik bir karar destek sistemi oluşturulması amaçlanmaktadır. Proje kapsamında elde edilen modelin, tıbbi görüntü analizinde uzmanlara yardımcı olabilecek bir araç olarak kullanılması hedeflenmektedir.


VERİSETİNİN KAYNAĞI
Bu çalışmada kullanılan veri seti, Kaggle platformu üzerinden temin edilmiştir. Veri seti, meme ultrason görüntülerinden oluşmakta olup, “Breast Ultrasound Images Dataset” adıyla paylaşılmaktadır  (Vuppala Adithya Sairam).
Veri setine aşağıdaki bağlantı ulaşılabilir:
https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer

VERİSETİ İÇERİĞİ
Bu çalışmada kullanılan veri seti, meme ultrason görüntülerinden oluşmaktadır. Veri seti iki sınıftan oluşmakta olup, benign (iyi huylu) ve malignant (kötü huylu) olarak etiketlenmiştir.
Veri setinde toplam 697 görüntü bulunmaktadır. Bu görüntülerin 487 tanesi benign, 210 tanesi ise malignant sınıfına aittir.
Görüntüler gri tonlu olup farklı boyutlarda bulunmakta ve modelde kullanılmadan önce yeniden boyutlandırma işlemi uygulanmıştır. Veri seti, gerçek hasta verilerinden elde edilmiş tıbbi görüntülerden oluşmaktadır.

PREPROCESSİNG
Modelin daha verimli öğrenebilmesi için veri setine bazı ön işleme adımları uygulanmıştır. Öncelikle tüm görüntüler model girişine uygun hale getirmek amacıyla 128x128 boyutuna yeniden boyutlandırılmıştır.
Daha sonra, piksel değerleri 0-255 aralığından 0-1 aralığına normalize edilmiştir. Bu işlem, modelin daha hızlı ve stabil öğrenmesini sağlamaktadır.
Veri seti, ayrıca eğitim ve doğrulama olmak üzere ikiye ayrılmış ve model bu ayrım üzerinden eğitilmiştir. Bu ön işleme adımları, model performansını artırmak ve daha sağlıklı sonuçlar elde etmek amacıyla uygulanmıştır.

VERİ BÖLME
Veri seti eğitim (train), doğrulama (validation) ve test (test) olmak üzere üç ayrı bölüme ayrılmıştır. Eğitim verisi modelin öğrenmesi için kullanılırken, doğrulama verisi eğitim sürecinde modelin performansını izlemek ve aşırı öğrenmeyi (overfitting) kontrol etmek amacıyla kullanılmıştır. Test verisi ise modelin daha önce görmediği veriler üzerindeki performansını değerlendirmek için kullanılmıştır. Bu sayede modelin genelleme yeteneği objektif bir şekilde ölçülmüştür. Veri seti yaklaşık olarak %70 eğitim, %20 doğrulama ve %10 test olarak dağıtılmıştır.
MODEL SEÇİMİ
Bu çalışmada model olarak Konvolüsyonel Sinir Ağı (Convolutional Neural Network - CNN) tercih edilmiştir. CNN modelleri, görüntü verileri üzerinde yüksek başarı gösteren ve görüntü içerisindeki önemli özellikleri otomatik olarak öğrenebilen derin öğrenme modelleridir.
Meme ultrason görüntüleri gibi görsel verilerde, kenar, doku ve şekil gibi özelliklerin çıkarılması büyük önem taşımaktadır. CNN yapısı, bu tür özellikleri manuel müdahaleye gerek kalmadan öğrenebildiği için bu problem için uygun bir model olarak seçilmiştir.
Ayrıca CNN modellerinin görüntü sınıflandırma problemlerinde yaygın olarak kullanılması ve başarılı sonuçlar vermesi, bu modelin tercih edilmesinde etkili olmuştur.


MODEL MİMARİSİ

Bu çalışmada kullanılan model, konvolüsyonel sinir ağı (CNN) tabanlı bir yapıya sahiptir. Model, giriş katmanı, konvolüsyon katmanları, havuzlama (pooling) katmanları ve tam bağlı (dense) katmanlardan oluşmaktadır.
İlk aşamada konvolüsyon katmanları kullanılarak görüntülerden anlamlı özellikler çıkarılmıştır. Bu katmanları takiben kullanılan max pooling katmanları ile boyut küçültme işlemi gerçekleştirilerek modelin daha verimli çalışması sağlanmıştır.
Sonraki aşamada flatten katmanı ile elde edilen özellikler düzleştirilmiş ve tam bağlı katmanlara aktarılmıştır. Son katmanda ise sigmoid aktivasyon fonksiyonu kullanılarak görüntünün benign veya malignant sınıfına ait olduğu tahmin edilmiştir.

HİPERPARAMETRELER
Model eğitimi sırasında çeşitli hiperparametreler belirlenmiş ve bu parametreler doğrultusunda eğitim gerçekleştirilmiştir. Eğitim sürecinde epoch sayısı 5 olarak seçilmiştir. Bu değer, modelin kısa sürede öğrenmesini sağlarken aşırı öğrenmenin (overfitting) önüne geçmek amacıyla düşük tutulmuştur.
Batch size değeri 16 olarak belirlenmiş olup, bu değer hem eğitim süresini optimize etmek hem de modelin stabil öğrenmesini sağlamak açısından uygun bir seçimdir. Modelin optimizasyonu için Adam optimizer kullanılmıştır. Adam, adaptif öğrenme oranı sayesinde hızlı ve etkili bir öğrenme süreci sağlamaktadır.
Kayıp fonksiyonu olarak ikili sınıflandırma problemi için uygun olan binary crossentropy kullanılmıştır. Model performansını değerlendirmek için ise accuracy metriği tercih edilmiştir. Bu hiperparametreler, modelin dengeli ve verimli bir şekilde eğitilmesini sağlamak amacıyla seçilmiştir.

Model, eğitim veri seti kullanılarak belirlenen hiperparametreler doğrultusunda eğitilmiştir. Eğitim sürecinde model, her epoch boyunca eğitim verileri üzerinde öğrenme gerçekleştirmiş ve aynı zamanda doğrulama verisi kullanılarak performansı izlenmiştir.
Eğitim sırasında modelin doğruluk (accuracy) değerinin artması ve kayıp (loss) değerinin azalması, modelin başarılı bir şekilde öğrenme gerçekleştirdiğini göstermektedir. Doğrulama verisi ile yapılan değerlendirme sayesinde modelin aşırı öğrenme (overfitting) durumu kontrol altında tutulmuştur.
Eğitim süreci sonunda model, test veri seti üzerinde değerlendirilerek nihai performansı ölçülmüştür.

PERFORMANS METRİKLERİ
Model performansını kapsamlı bir şekilde değerlendirebilmek amacıyla birden fazla performans metriği kullanılmıştır. Bu çalışmada accuracy, precision, recall, F1-score ve ROC eğrisi (AUC) gibi metrikler tercih edilmiştir.
Accuracy, modelin doğru tahmin oranını göstererek genel performans hakkında bilgi vermektedir. Ancak tek başına yeterli olmadığı için precision ve recall metrikleri de değerlendirilmiştir. Precision, modelin pozitif tahminlerinin ne kadarının doğru olduğunu gösterirken; recall, gerçek pozitif örneklerin ne kadarının doğru şekilde tespit edildiğini ifade etmektedir.
F1-score, precision ve recall değerlerinin dengeli bir ölçüsü olup, modelin genel başarısını daha doğru bir şekilde yansıtmaktadır. Özellikle veri setinin dengesiz olduğu durumlarda F1-score önemli bir metriktir.
Ayrıca model performansını farklı eşik değerlerinde incelemek amacıyla ROC eğrisi ve bu eğrinin altında kalan alanı ifade eden AUC değeri kullanılmıştır. Bu metrikler sayesinde modelin sınıflandırma performansı çok yönlü olarak değerlendirilmiştir.

PERFORMANS SONUÇLARI
Eğitim süreci sonunda modelin performansı doğrulama ve test veri setleri üzerinde değerlendirilmiştir. Modelin test veri seti üzerindeki doğruluk (accuracy) değeri yaklaşık %95.9 olarak elde edilmiştir.
Ayrıca model performansını daha detaylı incelemek amacıyla confusion matrix, ROC eğrisi ve classification report gibi değerlendirme yöntemleri kullanılmıştır. Elde edilen sonuçlar, modelin her iki sınıf için de yüksek doğruluk oranı ile tahmin yapabildiğini göstermektedir.
Precision, recall ve F1-score değerleri incelendiğinde modelin dengeli bir performans sergilediği ve hem yanlış pozitif hem de yanlış negatif oranlarının düşük olduğu görülmektedir.

    """)
