import json
import numpy as np
import random
import nltk
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pickle

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Dados de exemplo
with open('intents.json') as arquivo:
    dados = json.load(arquivo)

palavras = []
classes = []
documentos = []
palavras_ignoradas = ['?', '!', '.']

# Pré-processamento dos dados
for intent in dados['intents']:
    for pattern in intent['patterns']:
        lista_palavras = nltk.word_tokenize(pattern)
        palavras.extend(lista_palavras)
        documentos.append((lista_palavras, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

palavras = [palavra.lower() for palavra in palavras if palavra not in palavras_ignoradas]
palavras = sorted(list(set(palavras)))
classes = sorted(list(set(classes)))

treinamento = []
saida_vazia = [0] * len(classes)

for doc in documentos:
    bag = []
    padroes_palavras = doc[0]
    padroes_palavras = [palavra.lower() for palavra in padroes_palavras]
    bag = [1 if palavra in padroes_palavras else 0 for palavra in palavras]
    
    linha_saida = list(saida_vazia)
    linha_saida[classes.index(doc[1])] = 1
    treinamento.append([bag, linha_saida])

random.shuffle(treinamento)
treinamento = np.array(treinamento, dtype=object)
treino_x = np.array(list(treinamento[:, 0]))
treino_y = np.array(list(treinamento[:, 1]))

# Construção do modelo
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(treino_x[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(treino_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

modelo.fit(treino_x, treino_y, epochs=200, batch_size=5, verbose=1)

modelo.save('modelo_chatbot.h5')
pickle.dump(palavras, open('palavras.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

def limpar_sentenca(sentenca):
    palavras_sentenca = nltk.word_tokenize(sentenca)
    return palavras_sentenca

# Processamento do input do usuário
def bag_of_words(sentenca, palavras):
    palavras_sentenca = limpar_sentenca(sentenca)
    bag = [0] * len(palavras)
    for s in palavras_sentenca:
        for i, palavra in enumerate(palavras):
            if palavra.lower() == s.lower():
                bag[i] = 1
    return np.array(bag)

def prever_intencao(sentenca, modelo):
    p = bag_of_words(sentenca, palavras)
    res = modelo.predict(np.array([p]))[0]
    LIMIAR_ERRO = 0.25
    resultados = [[i, r] for i, r in enumerate(res) if r > LIMIAR_ERRO]
    resultados.sort(key=lambda x: x[1], reverse=True)
    lista_retorno = []
    for r in resultados:
        lista_retorno.append({"intencao": classes[r[0]], "probabilidade": str(r[1])})
    return lista_retorno

def obter_resposta(ints, intents_json):
    if not ints:
        return "Desculpe, não entendi sua pergunta."
    
    tag = ints[0]['intencao']
    lista_de_intencoes = intents_json['intents']
    for i in lista_de_intencoes:
        if i['tag'] == tag:
            resultado = random.choice(i['responses'])
            break
    return resultado

def resposta_chatbot(texto):
    ints = prever_intencao(texto, modelo)
    res = obter_resposta(ints, dados)
    return res

# Interação com o chatbot
print("Inicie a conversa com o bot (digite 'sair' para encerrar)")
while True:
    mensagem = input("Você: ")
    if mensagem.lower() in ["sair"]:
        break
    print("Bot:", resposta_chatbot(mensagem.lower()))
