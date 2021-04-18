import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Função de ativação.
def sigmoid (x):
    return 1/(1+(np.exp(-x)))

#Derivada da função de ativação.
def sigmoid_prime(sigmoid_):
    return sigmoid_ * (1 - sigmoid_)

#Função de combinação linear das matrizes.
def combi_linear (i, w):
    inputs = np.array(i)
    weights = np.array(w)
    return np.dot(i, w)

#-------------------------------------------    
#Dados.
DataSet=pd.read_csv('arruela_.csv')#Arquivo contendo os dados para a RNA.
DataSet.drop(['Hora','Tamanho'],axis=1,inplace=True)#Retira as informações de data e hora (informações irelevantes para a RNA).
DataSet.head()
DataSet.columns
DataSet.describe()
#-------------------------------------------    
#Escalonamento dos dados.
scaler=StandardScaler()
DataScaled=scaler.fit_transform(DataSet)
DataSetScaled=pd.DataFrame(np.array(DataScaled),columns = ['Referencia','NumAmostra', 'Area', 'Delta', 'Output1','Output2'])
#-------------------------------------------    
#Separação dos valores para a base de treinamento e teste
X = DataSetScaled.drop(['Output1', 'Output2'],axis=1)
y = DataSet[['Output1','Output2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Tamanho do DataSet de Treinamento
n_records, n_features = X_train.shape

#Graficos da recorrência dos valores nas Outputs
"""
sns.set_style('whitegrid')
sns.countplot(x='Referencia', hue='Output2', data=DataSet, palette='rainbow')
plt.xlabel('Valor da Referencia')
plt.ylabel('Número de Recorrência ')
plt.title('Relação entre a Referencia e os OUTPUTs')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='NumAmostra', hue='Output2', data=DataSet, palette='rainbow')
plt.xlabel('Valor do Número de Amostras')
plt.ylabel('Número de Recorrência')
plt.title('Relação entre a Número de Amostras e os OUTPUTs')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Area', hue='Output2', data=DataSet, palette='rainbow')
plt.xlabel('Valor da Área')
plt.ylabel('Número de Recorrência')
plt.title('Relação entre a Área e os OUTPUTs')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Delta', hue='Output2', data=DataSet, palette='rainbow')
plt.xlabel('Valor do Delta')
plt.ylabel('Número de Recorrência')
plt.title('Relação entre o Delta e os OUTPUTs')
plt.show()
"""
#Arquitetura da MPL
N_input = 4
N_hidden = 4
N_output = 2
b = 0.3
tentativas = 3000 #Número de iterações na base de treino para o aprendizado da RNA.

#Pesos gerados randomicamente.
weights_inputs = np.random.normal(0, scale=b, size=(N_input, N_hidden))
weights_out_ocultos = np.random.normal(0, scale=b, size=(N_hidden, N_output))

last_loss=None
#Váriaves para o grafico de erro.
EvolucaoError=[]
IndiceError=[]

#-------------------------------------------    
#Treinamento da RNA (aplica a função de ativação e calcula o gradiente do erro, )
for e in range(tentativas):

    delta_ocultos = np.zeros(weights_inputs.shape)
    delta_out = np.zeros(weights_out_ocultos.shape)

    for inputs, y in zip(X_train.values, y_train.values):
        
        #Cálculo dos valores de saida.
        out_ocultos = sigmoid(combi_linear(inputs, weights_inputs))
        out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
        
        #Erro da camada de entrada (entrada dos neuronios)
        error = y - out_ 
        gradiente_out = error * sigmoid_prime(out_)
        
        #Erro da camada de saida (Saida dos neuronios)
        error_ocultos = combi_linear(weights_out_ocultos, gradiente_out)
        gradiente_out_ocultos = error_ocultos * sigmoid_prime(out_ocultos) 

        #Variação dos erros.
        delta_ocultos += gradiente_out_ocultos * inputs[:,None]
        delta_out += gradiente_out * out_ocultos[:, None]

    #Atualização dos pesos.
    weights_inputs += b * delta_ocultos / n_records
    weights_out_ocultos += b * delta_out / n_records
    
    #Dados para gerar gráfico da evolução do erro.
    loss = np.mean((out_ - y) ** 2)   
    EvolucaoError.append(loss)
    IndiceError.append(e)

#Gráfico da evolução do erro na base de treinamento.
plt.plot(IndiceError, EvolucaoError, 'r') # 'r' is the color red
plt.xlabel('Numero de amostras')
plt.ylabel('Valor do erro quadratico')
plt.title('Evolução do Erro no treinamento da MPL')
plt.show()

#Tamanho da base de teste.
n_records, n_features = X_test.shape

MSE_Output1=0
MSE_Output2=0

predictions=0

#Variaveis para gerar grafico de comparação das saidas da rede.
trend_y_1 = []
trend_out_1 = []
trend_y_2 = []
trend_out_2 = []

#Aplicação da RNA na base de teste.
for xi, yi in zip(X_test.values, y_test.values):
# Forward Pass
    out_ocultos = sigmoid(combi_linear(xi, weights_inputs))
    out_ = sigmoid(combi_linear(out_ocultos, weights_out_ocultos))
    
#-------------------------------------------    
#Cálculo do Erro
    error = yi - out_
    #Atribuição de valores para o grafico.
    trend_y_1.append(yi[0])
    trend_out_1.append(out_[0])
    trend_y_2.append(yi[1])
    trend_out_2.append(out_[1])
    
    #Erro quadratico acumulado do teste da RNA.
    MSE_Output1 += (yi[0] - out_[0])**2
    MSE_Output2 += (yi[1] - out_[1])**2

    #Cálculo do Erro da Predição   
    if (out_[0]>out_[1]):
        if (yi[0]>yi[1]):
            predictions+=1
                
    if (out_[1]>=out_[0]):
        if (yi[1]>yi[0]):
            predictions+=1
          
#Erro Quadrático Médio 
MSE_Output1/=n_records
MSE_Output2/=n_records
"""
print('##############################################')
print('Erro Quadrático Médio da Saída Output1 é: ',MSE_Output1)
print('Erro Quadrático Médio da Saída Output2 é: ',MSE_Output2)
print('##############################################')
print("A Acurácia da Predição é de: {:.3f}".format(predictions/n_records))
"""
#-------------------------------------------    
#Elaboração do grafico comparativo do resultado da RNA.
#Trend em vermelho mostrando os valores calculados da output 1 
plt.plot(trend_out_1, 'r')
#Trend em amarelo mostrando os valores da output 1
plt.plot(trend_y_1, 'y')
#Trend em azul mostrando os valores calculadors da output 2
plt.plot(trend_out_2, 'b')
#Trend em verde mostrando os valores da output 2
plt.plot(trend_y_2, 'g')

plt.xlabel('VERMELHO = OUTPUT-1 CALCULADA \n AMARELO = OUTPUT-1 \n AZUL = OUTPUT-2 CALCULADA \n VERDE = OUTPUT-2')
plt.ylabel('Valor da Output')
plt.title('Resultados da RNA - Acurácia da Predição = {:.3f}'.format(predictions/n_records))
plt.show()