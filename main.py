# %% [markdown]
# # Müzik Duygularının Sınıflandırılması
# ## Turkish Music Emotion Dataset ile Veri Analizi
#
# Bu projede, UCI Machine Learning Repository'den alınan **Turkish Music Emotion** veri seti kullanılarak müzik parçalarının taşıdığı duygular sınıflandırılmıştır. Veri seti 400 örnek ve 50 farklı akustik özelliği içermektedir. Amaç, müziklerin `happy`, `sad`, `relax`, `angry` gibi etiketlere doğru şekilde atanmasını sağlamaktır.
#
# Bu çalışma kapsamında:
# - Veriler ön işlenmiştir (ölçekleme, etiketleme),
# - İki farklı makine öğrenmesi algoritması uygulanmıştır: **KNN** ve **SVM**,
# - Sonuçlar sayısal ve grafiksel olarak karşılaştırılmıştır.

# %%
# Gerekki kütüphaneleri içe aktar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE

# %%
# Veri setini yükle
df = pd.read_csv("dataset/Acoustic_Features.csv")
df.head()

# %%
# Veri setini önizle
print("Satır ve sütun sayısı:", df.shape)
print("Sütun isimleri:", df.columns)
print("Eksik veri kontrolü:", df.isnull().sum())
df.describe()  # İstatistiksel özet

# %% [markdown]
# ## Veri Ön İşleme
#
# - **LabelEncoder** ile sınıflar sayıya çevrilmiştir.
# - **StandardScaler** kullanılarak tüm özellikler normalize edilmiştir.
# - Veriler eğitim (%80) ve test (%20) olmak üzere ikiye ayrılmıştır.

# %%
# Veri seti ön işleme
# Label sütunu varsa 'emotion' veya benzeri
label_col = "Class"  # Gerçek etiket sütun ismini kontrol et!
X = df.drop(label_col, axis=1)
y = df[label_col]

# Etiketleri sayıya çevir
le = LabelEncoder()
y = le.fit_transform(y)

class_names = le.classes_

# Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/test ayırımı
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# %% [markdown]
# ## Kullanılan Modeller
#
# ### 1. KNN (K-En Yakın Komşu)
# - Parametre: `k = 5`
# - Basit ve yorumlanabilir bir algoritmadır.
# - Özellikle küçük veri setlerinde hızlı sonuç verir.
#
# ### 2. SVM (Destek Vektör Makineleri)
# - `rbf` çekirdeği ile kullanılmıştır.
# - Sınıflar arası maksimum ayrımı hedefler.
# - Genelde daha yüksek doğruluk sağlar.

# %%
# Modelleme (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# %%
# Modelleme (SVM)
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# %% [markdown]
# ## Sayısal Karşılaştırma
#
# - Accuracy (Doğruluk):
#   - KNN: %62.0
#   - SVM: **%81.0**
# - `classification_report` ile Precision, Recall ve F1-Score detayları incelenmiştir.

# %%
# Doğruluk Oranları
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"KNN Doğruluk Oranı: {accuracy_knn:.2f}")
print(f"SVM Doğruluk Oranı: {accuracy_svm:.2f}")

# Sınıflandırma Raporları
print("\nKNN Sınıflandırma Raporu:\n")
print(classification_report(y_test, y_pred_knn, target_names=class_names))

print("\nSVM Sınıflandırma Raporu:\n")
print(classification_report(y_test, y_pred_svm, target_names=class_names))

# %% [markdown]
# ## Görselleştirme
#
# ### 1. Confusion Matrix (Karışıklık Matrisi)
# Her modelin tahmin başarısı, gerçek sınıflarla kıyaslanarak görselleştirilmiştir.

# %%
# Confusion Matrix – KNN
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_knn),
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("KNN Confusion Matrix")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.tight_layout()
plt.show()

# Confusion Matrix – SVM
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_svm),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("SVM Confusion Matrix")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2. t-SNE ile Scatter Plot
# Yüksek boyutlu veri, 2 boyuta indirgenerek sınıfların birbirinden ayrımı görselleştirilmiştir.

# %%
# t-SNE ile görselleştirme
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# t-SNE sonuçlarını DataFrame'e çevir
tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
tsne_df["Label"] = le.inverse_transform(y)  # orijinal etiketlerle görselleştirme

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="Label", palette="Set2", s=70)
plt.title("t-SNE ile Scatter Plot (Müzik Duygu Sınıfları)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Duygu")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sonuç
#
# - **SVM modeli**, `relax` ve `happy` sınıflarında yüksek başarı göstermiştir.
# - **KNN modeli**, özellikle `sad` sınıfında zayıf kalmıştır.
# - Genel olarak, **SVM modeli bu veri setinde daha etkili sonuçlar vermektedir.**
