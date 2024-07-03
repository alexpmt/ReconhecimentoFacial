import tkinter.messagebox as messagebox
import speech_recognition as sr

def reconhecer_nome_por_voz():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        messagebox.showinfo("Reconhecimento de Voz", "Fale seu nome completo:.")
        audio = recognizer.listen(source)

    try:
        nome_pessoa_presenca = recognizer.recognize_google(audio, language='pt-BR')
        return nome_pessoa_presenca
    except sr.UnknownValueError:
        messagebox.showerror("Erro", "Não entendi o que você disse.")
        return None
    except sr.RequestError:
        messagebox.showerror("Erro", "Não foi possível acessar o serviço de reconhecimento de voz.")
        return None
