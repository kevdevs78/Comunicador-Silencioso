import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Configurar propiedades de voz
        self.engine.setProperty('rate', 150)  # Velocidad
        self.engine.setProperty('volume', 0.8)  # Volumen
    
    def speak(self, text):
        try:
            print(f"Reproduciendo: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error en síntesis de voz: {e}")