import speech_recognition as sr

def abrir_microfone():
    reconhecedor = sr.Recognizer()
    with sr.Microphone() as source:
        print("Diga alguma coisa:")
        reconhecedor.adjust_for_ambient_noise(source)
        audio = reconhecedor.listen(source)

    try:
        texto = reconhecedor.recognize_google(audio, language='pt-BR')
        print("Nome do aluno: " + texto)
        return texto
    except sr.UnknownValueError:
        print("Não consegui entender o áudio, tente novamente")
    except sr.RequestError as e:
        print("Erro ao acessar o serviço de reconhecimento de fala; {0}".format(e))

if __name__ == "__main__":
    escutar_microfone()
