import tkinter as tk
from tkinter import messagebox
from codigo_principal import DetectorDeRostos


def iniciar_reconhecimento():
    nome_pessoa = entry_nome.get()
    if not nome_pessoa:
        messagebox.showwarning("Aviso", "Por favor, insira um nome.")
        return

    detector = DetectorDeRostos(pasta_presenca="presença")
    detector.executar(nome_pessoa)


janela = tk.Tk()
janela.title("Sistema de Reconhecimento Facial")


label_nome = tk.Label(janela, text="Nome para registro de presença:")
label_nome.pack(pady=10)

entry_nome = tk.Entry(janela)
entry_nome.pack(pady=5)

botao_iniciar = tk.Button(janela, text="Iniciar Reconhecimento", command=iniciar_reconhecimento)
botao_iniciar.pack(pady=20)

janela.mainloop()