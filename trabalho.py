import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def sigmoid (x):
    return 1/(1+(np.exp(-x)))

def sigmoid_prime(sigmoid_):
    return sigmoid_ * (1 - sigmoid_)

def combi_linear (i, w):
    inputs = np.array(i)
    weights = np.array(w)
    return np.dot(inputs, weights)

DataSet=pd.read_csv('Data.csv')
DataSet.head()
DataSet.columns
X_train, X_test, y_train, y_test = train_test_split(DataSet.drop(['Output1', 'Output2'],axis=1), DataSet[['Output1', 'Output2']], test_size=0.30, random_state=101)

#Tamanho do DataSet de Treinamento
n_records, n_features = X_train.shape

#Arquitetura da MPL
N_input = 3
N_hidden = 4
N_output = 2
b = 0.1

weights_inputs = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_out_ocultos = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

tentativas = 5000
last_loss=None
EvolucaoError=[]
IndiceError=[]

for e in range(tentativas):

    for inputs, y in zip(X_train.values, y_train.values):
        out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
        out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))

        error = y - out_ 
        gradiente_out = error * sigmoid_prime(out_)

        error_ocultos = combi_linear(weights_out_ocultos, gradiente_out)
        gradiente_out_ocultos = error_ocultos * sigmoid_prime(out_ocultos) 

    weights_inputs = b * (b * gradiente_out_ocultos * inputs[:, None] / n_records)
    weights_out_ocultos = b * (b * gradiente_out * out_ocultos[:, None] / n_records)
    # Imprimir o erro quadrático médio no conjunto de treinamento
    if  e % (tentativas / 20) == 0:
        out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
        out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
        loss = np.mean((out_ - y) ** 2)

        if last_loss and last_loss < loss:
            print("Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando")
        else:
            print("Erro quadrático no treinamento: ", loss)
        last_loss = loss
         
        EvolucaoError.append(loss)
        IndiceError.append(e)

plt.plot(IndiceError, EvolucaoError, 'r') # 'r' is the color red
plt.xlabel('')
plt.ylabel('Erro Quadrático')
plt.title('Evolução do Erro no treinamento da MPL')
plt.show()

n_records, n_features = X_test.shape
MSE_Output1=0
MSE_Output2=0

for xi, yi in zip(X_test.values, y_test.values):
# Forward Pass
    out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
    out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
#-------------------------------------------    
#Cálculo do Erro
    error = yi - out_
    MSE_Output1 += (yi[0] - out_[0])**2
    MSE_Output2 += (yi[1] - out_[1])**2          
#Erro Quadrático Médio 
MSE_Output1/=n_records
MSE_Output2/=n_records

print('##############################################')
print('Erro Quadrático Médio da Saída Output1 é: ',MSE_Output1)
print('Erro Quadrático Médio da Saída Output2 é: ',MSE_Output2)
print('##############################################')
print('Output da rede = ',out_)