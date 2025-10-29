#  Proyecto N.º 5: Comunicador Silencioso  
**Universidad Mariano Gálvez de Guatemala**  
**Curso:** Autómatas y Lenguajes Formales  



---

## 🎯 1. Objetivo General

Desarrollar una aplicación accesible e inteligente que permita a personas con discapacidad auditiva y/o verbal comunicarse en tiempo real mediante lenguaje de señas detectado por cámara, el cual es procesado, traducido automáticamente y convertido en texto y voz.  
El sistema aplica **técnicas de visión por computadora**, **procesamiento de lenguaje** y **fundamentos de autómatas y lenguajes formales**.

---

## 🧠 2. Fundamento en Autómatas y Lenguajes Formales

| Concepto del curso | Aplicación en el proyecto |
|--------------------|---------------------------|
| **Lenguaje formal (L)** | Cada seña representa un símbolo; las frases son cadenas válidas en el lenguaje. |
| **Gramática libre de contexto (CFG)** | Define la estructura sintáctica de frases formadas por señas. |
| **Autómata finito determinista (AFD)** | Permite validar paso a paso la secuencia de gestos reconocidos. |
| **Tokens léxicos** | Cada seña identificada se interpreta como un token del lenguaje. |
| **Análisis sintáctico** | Se utiliza para validar que la secuencia de señas cumpla una estructura gramatical válida. |

---

## ⚙️ 3. Descripción del sistema

El sistema está compuesto por **tres módulos principales:**

### 🖐️ A. Captura de señas (entrada por cámara)
- Usa una webcam para captar los gestos del usuario.  
- El modelo de visión artificial (MediaPipe + OpenCV) identifica cada seña y la convierte en un **símbolo o token léxico**.  
- Las muestras capturadas se almacenan en `data/raw/` para entrenamiento posterior.

### 🧩 B. Reconocimiento y traducción
- Cada seña reconocida se convierte en un token.  
- Un **parser** basado en una **gramática libre de contexto (CFG)** valida si la secuencia forma una frase válida.  
- Si la estructura es correcta, se genera la frase completa en texto.  
- El modelo usa algoritmos **SVM** y **Random Forest**, seleccionando el de mayor precisión.

### 🔊 C. Salida accesible (texto y voz)
- La frase reconocida se muestra en pantalla.  
- Se convierte automáticamente a voz mediante un motor TTS (`pyttsx3`).

---

## 🧩 4. Componentes del sistema

| Archivo | Descripción |
|----------|-------------|
| **hand_detector.py** | Detecta manos y extrae características (landmarks y ángulos). |
| **data_collection.py** | Captura muestras de entrenamiento para cada seña (una o dos manos). |
| **model_training.py** | Entrena modelos SVM y Random Forest; guarda el mejor modelo. |
| **prediction.py** | Traduce señas en tiempo real; genera texto y salida de voz. |
| **text_to_speech.py** | Módulo de texto a voz usando `pyttsx3`. |
| **gif_generator.py** | Crea GIFs de las capturas o predicciones para documentación visual. |

---

## 📂 5. Estructura de carpetas

```
ComunicadorSilencioso/
│
├── src/
│   ├── hand_detector.py
│   ├── data_collection.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── text_to_speech.py
│   └── gif_generator.py
│
├── data/
│   ├── raw/               # Muestras (.npy, .jpg)
│   ├── trained_models/    # Modelos entrenados (.pkl)
│   └── gifs/              # GIFs generados
│
├── requirements.txt
└── README.md
```

---

## 🧮 6. Diseño formal del lenguaje

### Ejemplo de gramática formal (CFG)

```
S → PRONOMBRE VERBO OBJETO
PRONOMBRE → YO | TÚ | ÉL
VERBO → QUIERO | NECESITO | VOY
OBJETO → AGUA | AYUDA | COMER | BAÑO
```

El sistema valida que la secuencia de señas reconocidas forme una frase sintácticamente válida según esta gramática.  
Por ejemplo:

✅ **YO QUIERO AGUA** → frase válida  
❌ **QUIERO YO AGUA** → frase inválida

---

## 🧰 7. Tecnologías utilizadas

| Componente | Herramientas |
|-------------|--------------|
| **Visión por computadora** | OpenCV + MediaPipe + TensorFlow |
| **Clasificación de gestos** | SVM / RandomForest (Scikit-learn) |
| **Parser (CFG / AFD)** | Implementado en Python |
| **Texto a voz (TTS)** | pyttsx3 |
| **Interfaz** | OpenCV (ventanas interactivas) |

---

## 🧩 8. Dependencias

Archivo `requirements.txt`:
```
opencv-python==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
Pillow==10.0.1
pyttsx3==2.90
tensorflow==2.13.0
matplotlib==3.9.4
```

Instalación rápida:
```bash
pip install -r requirements.txt
```

---

## 🚀 9. Instrucciones de ejecución

### 1️⃣ Capturar señas
```bash
python data_collection.py
```
- Muestra tus manos frente a la cámara.  
- Presiona `c` para capturar.  
- Presiona `q` para salir.

### 2️⃣ Entrenar el modelo
```bash
python model_training.py
```
- Entrena y guarda el mejor modelo en `data/trained_models/`.

### 3️⃣ Predicción en tiempo real
```bash
python prediction.py
```
Controles:
- `s` → reproducir voz.  
- `g` → guardar GIF.  
- `q` → salir.  

---

## 🎞️ 10. Video de presentación (máx. 10 min)

Debe incluir:
- Explicación de la arquitectura general del sistema.  
- Demostración del reconocimiento de señas y validación de frases.  
- Explicación del código clave (relación con autómatas y lenguajes formales).  
- Reflexión final sobre la utilidad social del proyecto.  

---



---

## 👥 11. Público beneficiado

- Personas con sordera profunda y/o mutismo.  
- Niños en alfabetización bilingüe (señas + español).  
- Familiares, docentes, médicos o personal de atención al público.

---

## 🧠 12. Conceptos aplicados

- Autómatas finitos deterministas (AFD).  
- Gramáticas libres de contexto (CFG).  
- Análisis léxico y sintáctico (tokens y parser).  
- Reconocimiento de patrones con aprendizaje automático.  
- Procesamiento de lenguaje natural (PLN) y síntesis de voz (TTS).

---

## 🏁 13. Licencia

Este proyecto se desarrolla con fines **académicos y educativos**.  
Se autoriza su uso, modificación y difusión reconociendo al autor original:


Estudiantes de Ingeniería en Sistemas  
Universidad Mariano Gálvez de Guatemala
