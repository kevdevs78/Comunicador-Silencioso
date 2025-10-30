import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class ModelTrainer:
    def __init__(self):
        self.data_path = "data/raw"
        self.model_path = "data/trained_models"
        os.makedirs(self.model_path, exist_ok=True)
        self.scaler = StandardScaler()
    
    def load_data(self):
        features_list = []
        labels = []
        label_names = []

        print("Cargando características de landmarks...")

        max_len = 0
        raw_data = []

        for filename in os.listdir(self.data_path):
            if filename.endswith('.npy'):
                label = filename.split('_')[0]
                if label not in label_names:
                    label_names.append(label)
                features = np.load(os.path.join(self.data_path, filename))
                raw_data.append((label, features))
                if len(features) > max_len:
                    max_len = len(features)

        if not raw_data:
            return np.array([]), np.array([]), []

        # 🔹 Normalizar longitud de vectores
        for label, features in raw_data:
            if len(features) < max_len:
                padded = np.pad(features, (0, max_len - len(features)), mode='constant')
            else:
                padded = features
            features_list.append(padded)
            labels.append(label_names.index(label))

        X = np.array(features_list)
        y = np.array(labels)

        # Normalizar características
        X = self.scaler.fit_transform(X)
        print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Clases detectadas: {label_names}")
        return X, y, label_names

    def train_model(self):
        try:
            X, y, class_names = self.load_data()

            if len(X) == 0:
                print("No hay datos para entrenar. Captura algunas señas primero.")
                return

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Entrenar modelos
            print("Entrenando modelo SVM...")
            svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            svm_model.fit(X_train, y_train)

            print("Entrenando modelo Random Forest...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Evaluar modelos
            svm_acc = svm_model.score(X_test, y_test)
            rf_acc = rf_model.score(X_test, y_test)

            print(f"Precisión SVM: {svm_acc:.3f}")
            print(f"Precisión Random Forest: {rf_acc:.3f}")

            # Escoger el mejor
            best_model = svm_model if svm_acc >= rf_acc else rf_model
            print("Usando modelo:", "SVM" if svm_acc >= rf_acc else "Random Forest")

            # Guardar modelo y scaler
            joblib.dump(best_model, os.path.join(self.model_path, 'sign_language_model.pkl'))
            joblib.dump(class_names, os.path.join(self.model_path, 'class_names.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))

            print("✅ ¡Modelo entrenado y guardado exitosamente!")

        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
