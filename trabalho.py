import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid (x):
    return 1/(1+(np.exp(-x)))

def sigmoid_prime(sigmoid_):
    return sigmoid_ * (1 - sigmoid_)

def combi_linear (i, w):
    inputs = np.array(i)
    weights = np.array(w)
    return np.dot(i, w)

DataSet=pd.read_csv('arruela_.csv')
#DataSet.drop(['Hora','Tamanho','Referencia'],axis=1,inplace=True)
DataSet.drop(['Hora','Tamanho'],axis=1,inplace=True)
DataSet.head()
DataSet.columns
DataSet.describe()

"""sns.pairplot(DataSet)
plt.show()
"""

scaler=StandardScaler()
DataScaled=scaler.fit_transform(DataSet)
DataSetScaled=pd.DataFrame(np.array(DataScaled),columns = ['Referencia','NumAmostra', 'Area', 'Delta', 'Output1','Output2'])



X = DataSetScaled.drop(['Output1', 'Output2'],axis=1)
y = DataSet[['Output1','Output2']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Tamanho do DataSet de Treinamento
n_records, n_features = X_train.shape

#Arquitetura da MPL
N_input = 4
N_hidden = 4
N_output = 2
b = 0.3

weights_inputs = np.random.normal(0, scale=b, size=(N_input, N_hidden))
weights_out_ocultos = np.random.normal(0, scale=b, size=(N_hidden, N_output))

tentativas = 5000
last_loss=None
EvolucaoError=[]
IndiceError=[]

for e in range(tentativas):

    delta_ocultos = np.zeros(weights_inputs.shape)
    delta_out = np.zeros(weights_out_ocultos.shape)

    for inputs, y in zip(X_train.values, y_train.values):
        
        out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
        out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))

        error = y - out_ 
        gradiente_out = error * sigmoid_prime(out_)

        error_ocultos = combi_linear(weights_out_ocultos, gradiente_out)
        gradiente_out_ocultos = error_ocultos * sigmoid_prime(out_ocultos) 

        delta_ocultos += gradiente_out_ocultos * inputs[:,None]
        delta_out += gradiente_out * out_ocultos[:, None]

    weights_inputs += b * delta_ocultos / n_records
    weights_out_ocultos += b * delta_out / n_records
    
    """out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
    out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
    loss = np.mean((out_ - y) ** 2)   
    EvolucaoError.append(loss)
    IndiceError.append(e)

    
    # Imprimir o erro quadrático médio no conjunto de treinamento
    if  e % (tentativas / 20) == 0:
        if last_loss and last_loss < loss:
            print("Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando")
        else:
            print("Erro quadrático no treinamento: ", loss)
        last_loss = loss
    
    
plt.plot(IndiceError, EvolucaoError, 'r') # 'r' is the color red
plt.xlabel('Numero de amostras')
plt.ylabel('Valor do erro quadratico')
plt.title('Evolução do Erro no treinamento da MPL')
plt.show()
"""


n_records, n_features = X_test.shape
MSE_Output1=0
MSE_Output2=0
predictions=0

trand_y_1 = []
trand_out_1 = []
trand_y_2 = []
trand_out_2 = []
#trand_erro = []

for xi, yi in zip(X_test.values, y_test.values):

# Forward Pass
    out_ocultos = sigmoid(combi_linear(xi, weights_inputs))
    out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
    
#-------------------------------------------    
#Cálculo do Erro
    error = yi - out_
    trand_y_1.append(yi[0])
    trand_out_1.append(out_[0])
    trand_y_2.append(yi[1])
    trand_out_2.append(out_[1])
    #trand_erro.append(error[0])
    MSE_Output1 += (yi[0] - out_[0])**2
    MSE_Output2 += (yi[1] - out_[1])**2

    #Cálculo do Erro da Predição
        ## TODO: Cálculo do Erro        
    if (out_[0]>out_[1]):
        if (yi[0]>yi[1]):
            predictions+=1
                
    if (out_[1]>=out_[0]):
        if (yi[1]>yi[0]):
            predictions+=1
          
#Erro Quadrático Médio 
MSE_Output1/=n_records
MSE_Output2/=n_records

print('##############################################')
print('Erro Quadrático Médio da Saída Output1 é: ',MSE_Output1)
print('Erro Quadrático Médio da Saída Output2 é: ',MSE_Output2)
print('##############################################')
print('Output da rede = ',out_)

print("A Acurácia da Predição é de: {:.3f}".format(predictions/n_records))

plt.plot(trand_out_1, 'r') # 'r' is the color red
plt.plot(trand_y_1, 'y') # 'r' is the color red
plt.plot(trand_out_2, 'b') # 'r' is the color red
plt.plot(trand_y_2, 'g') # 'r' is the color red
plt.xlabel('h')
plt.ylabel('hhhh')
plt.title('Evolução do Erro no treinamento da MPL')
plt.show()
