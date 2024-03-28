# Importando as bibliotecas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import os


# Definindo a quantidade de THREADS a ser utilizada pelo KMeans
# Obs.1: suprimindo o aviso de falha de vazamento de memória do windows quando está utilizando o algoritmo KMeans 
# Obs.2: basicamente acontece quando o computador possui mais de um núcleo de processamento, dependendo da versão do Scikit-learn
os.environ["OMP_NUM_THREADS"] = "9"

# ===========================================================================================================
# Criando uma função genérica "inspect_outliers" para a Inspeção dos Outliers no DataFrame que poderá em qualquer Dataset
# Obs.: passando para a função os parâmetros dataframe, column e whisker_width = 1.5
# ===========================================================================================================
def inspect_outliers(dataframe, column, whisker_width = 1.5):
    """Função para inspecionar outliers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    column : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    whisker_width : float, opcional
        Valor considerado para detecção de outliers, por padrão 1.5

    Returns
    -------
    pd.DataFrame
        Dataframe com os outliers.
    """

    # q1, primeiro intervalo interquartil, menos 1.5 * intervalo interquartil para pegar os Outliers inferiores 
    # Obs.: o .quantile(0.25) é um método do Pandas que corresponde a 25% no describe da Coluna
    q1 = dataframe[column].quantile(0.25)
    
    # q3, segundo intervalo interquartil, mais 1.5 * intervalo interquartil para pegar os Outliers superiores
    # Obs.: o .quantile(0.75) é um método do Pandas que corresponde a 75% no describe da Coluna
    q3 = dataframe[column].quantile(0.75)
    
    # iqr é o interquartil range, que corresponde à Caixa do BoxPlot
    iqr = q3 - q1
    
    # Limite Inferior (whisker_width = bigode do BoxPlot)
    # Obs.: valor convencionado do whisker_width = 1.5
    lower_bound = q1 - whisker_width * iqr
    
    # Limite Superior (whisker_width = bigode do BoxPlot)
    # Obs.: valor convencionado do whisker_width = 1.5
    upper_bound = q3 + whisker_width * iqr

    # Retornando o resultado da função, sendo o DataFrame Filtrado
    return dataframe[
        # Pegando as colunas e filtrando os valores que estão abaixo do Limite Inferior ou os valores que estão acima do Limite Superior, ou seja, os possíveis Outliers Inferiores e Superiores
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]


# ===========================================================================================================
# Criando uma função genérica "pairplot" para a geração de pairplots kde passando valores distintos (colunas distintas) para o parâmetro hue
# Obs.: passando os parâmetros para a função
# ===========================================================================================================
def pairplot(
    dataframe, columns, hue_column = None, alpha = 0.5, corner = True, palette = "tab10"
):
    """Função para gerar pairplot.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    hue_column : str, opcional
        Coluna utilizada para hue, por padrão None
    alpha : float, opcional
        Valor de alfa para transparência, por padrão 0.5
    corner : bool, opcional
        Se o pairplot terá apenas a diagonal inferior ou será completo, por padrão True
    palette : str, opcional
        Paleta a ser utilizada, por padrão "tab10"
    """

    # Criando a variável "analysis" e atribuindo como valor a concatenação (combinação) das colunas que foram passadas (criando uma cópia delas internamente na função) com o [hue_column] na forma de uma Lista, de modo que não será necessário sobreescrever a variável para o parâmetro hue.
    analysis = columns.copy() + [hue_column]

    # Gerando o PairPlot kde
    sns.pairplot(
        # Filtrando as colunas do DataFrame
        dataframe[analysis],
        # Definindo o Tipo de Gráfico
        diag_kind = "kde",
        # Definindo um valor genérico para o parâmetro hue que deverá ser informado como parâmetro ao chamar a função "pairplot"
        hue = hue_column,
        # Colocando um pouco de transparência no pairplot
        plot_kws = dict(alpha = alpha),
        # Mostrando as dispersões somente abaixo da diagonal
        corner = corner,
        # Passando o palette de cores 
        palette = palette,
    )


# ============================================================================================================ 
# CRIANDO UM GRÁFICO GENÉRICO PARA SER UTILIZADO SEMPRE QUE FOR NECESSÁRIO DETERMINAR A QUANTIDADE DE CLUSTERS 
# IDEALMENTE DEVERÁ SER SALVA NA PASTA Scripts do Ambiente Virtual criado para o Projeto
# ============================================================================================================

# Criando a Function plot_elbow_silhouette
# Obs.1: A variável X deverá ser passada como parâmetro para definir as colunas do gráfico 
# Obs.2: O parãmetro random_state deverá ser informado se desejar que seja diferente de random_state = 42
# Obs.3: O parãmetro range_k deverá ser informado se desejar que seja diferente de range_k = (2, 11)
def plot_elbow_silhouette(X, random_state = 42, range_k = (2, 11)):
    """Gera os gráficos para os métodos Elbow e Silhouette.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe com os dados.
    random_state : int, opcional
        Valor para fixar o estado aleatório para reprodutibilidade, por padrão 42
    range_k : tuple, opcional
        Intervalo de valores de cluster, por padrão (2, 11)
    """

    # Criando uma figura com 2 sistemas de eixos sendo 1 linha e 2 colunas 
    # Obs.: usando tight_layout = True para os gráficos se ajustarem evitando a sobreposição
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6), tight_layout = True)

    # Criando uma estrutura de dados, um dicionário vazio para o gráfico do cotovelo
    elbow = {}

    # Criando uma estrutura de dados, uma lista vazia para o gráfico da silhouette
    silhouette = []

    # Criando uma variável com um range de 2 a 10 (11 exclusive)
    # Obs.: fazendo um teste de 2 a 10 para definir qual é o melhor valor de k para criação dos clusters
    k_range = range(*range_k) # fazendo um Unpacking Implícito (*), pois os valores estão dentro de uma Tupla

    # Percorrendo a variável x em k_range
    for i in k_range:
        # Aplicando o algoritmo e atribuindo à variável "kmeans"
        kmeans = KMeans(n_clusters = i, random_state = random_state, n_init = 10)
        # Fazendo o fit do Modelo
        kmeans.fit(X)
        # Passando para o gráfico elbow a chave "i" que indica o número de clusters e o kmeans.inertia_
        elbow[i] = kmeans.inertia_
        # Passando para o gráfico silhouette os rótulos(labels)
        labels = kmeans.labels_
        # Fazendo um append para o silhouette os parâmetros silhouette_score(X, labels)
        silhouette.append(silhouette_score(X, labels))

    # Plotando o gráfico de linhas do seaborn elbow(cotovelo)
    # Obs.1: passando no parâmetro x uma lista das chaves do dicionário
    # Obs.2: passando no parâmetro y uma lista dos valores do dicionário
    # Obs.3: passando no parâmetro ax a posição de índice 0 para o sistema de eixos
    sns.lineplot(x = list(elbow.keys()), y = list(elbow.values()), ax = axs[0])
    # Definindo o rótulo do Eixo X
    axs[0].set_xlabel("K")
    # Definindo o rótulo do Eixo Y
    axs[0].set_xlabel("Inertia")
    # Definindo o título para o gráfico
    axs[0].set_title("Elbow Method")

    # Plotando o gráfico de linhas do seaborn silhouette
    # Obs.1: passando no parâmetro x uma lista do k_range
    # Obs.2: passando no parâmetro y é o resultado da lista silhouette
    # Obs.3: passando no parâmetro ax a posição de índice 1 para o sistema de eixos
    sns.lineplot(x = list(k_range), y = silhouette, ax = axs[1])
    # Definindo o rótulo do Eixo X
    axs[1].set_xlabel("K")
    # Definindo o rótulo do Eixo Y
    axs[1].set_xlabel("Silhouette Score")
    # Definindo o título para o gráfico
    axs[1].set_title("Silhouette Method")

    # Exibindo o gráfico
    plt.show()
    



#===========================================================================================================   # Criando uma Figura Completa com projeção em 3 Dimensões
# Obs.: tornando a função mais genérica para poder ser utilizada em quaisquer outros Scripts
#=========================================================================================================== 
# Criando uma função para a visualização dos clusters
def visualizar_clusters(
    # Passando os parâmetros para a função
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids = True, 
    mostrar_pontos = False,
    coluna_clusters = None
):
    """Gerar gráfico 3D com os clusters.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    n_colors : int
        Número de cores para o gráfico.
    centroids : np.ndarray
        Array com os centroides.
    show_centroids : bool, opcional
        Se o gráfico irá mostrar os centroides ou não, por padrão True
    show_points : bool, opcional
        Se o gráfico irá mostrar os pontos ou não, por padrão False
    column_clusters : List[int], opcional
        Coluna com os números dos clusters para colorir os pontos
        (caso mostrar_pontos seja True), por padrão None
    """
    
    # Criando a figura
    fig = plt.figure()
    
    # Criando um sistema de eixos e adicionando um subplot à figura
    # Obs.: passando o parâmetro 111, projection="3d" para gerar o gráfico 3D (ver documentação do matplotlib)
    ax = fig.add_subplot(111, projection = "3d")
    
    # Atribuindo à variável "cores" a quantidade de cores a serem utilizadas do color map da paleta de cores tab10 do matplotlib
    # Obs.: funciona para qualquer quantidade de cores que for passada como parâmetro da função
    cores = plt.cm.tab10.colors[:quantidade_cores]
    
    # Sobreescrevendo a variável cores passando a variável cores original para o ListedColormap(cores)
    cores = ListedColormap(cores)
    
    # Definindo a lista de colunas do DataFrame de acordo com os índices, colunas Age, Annual Income (k$) e Spending Score (1-100)
    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]
    
    # Mostrar os Centróides: usar parâmetro True
    ligar_centroids = mostrar_centroids
    
    # Mostrar os Pontos: usar parâmetro True
    ligar_pontos = mostrar_pontos
    
    # Percorrendo as variáveis "i" e "centroid" em enumerate(centroids)
    for i, centroid in enumerate(centroids):
        # Se for verdadeiro
        if ligar_centroids: 
            # Mostrando as coordenadas dos pontos x, y e z no gráfico de dispersão
            # Obs.: fazendo o Unpacking Implícito, passando o tamanho dos pontos no parâmetro s = 500 e alpha = 0.5
            ax.scatter(*centroid, s = 500, alpha = 0.5)
            # Mostrando nas coordenadas os números dos clusters
            # Obs.:passando os parâmetros tamanho da fonte e alinhamento horizontal/vertical
            ax.text(*centroid, f"{i}", fontsize = 20, horizontalalignment = "center", verticalalignment = "center")
        # Se for verdadeiro
        if ligar_pontos:
            # Mostrando todos os demais pontos e atribuindo à variável "s" (s de scatter)
            # Obs.1: definindo as cores no parâmetro c = coluna_clusters
            # Obs.2: ajustando o mapa de cores no parâmetro cmap = cores
            s = ax.scatter(x, y, z, c = coluna_clusters, cmap = cores)
            # Passando os parãmetros de legenda do gráfico de dispersão
            # Obs.: ajustando a posição da legenda, fazendo a ancoragem de legenda para não sobrepor o gráfico: bbox_to_anchor = (1.3, 1)
            ax.legend(*s.legend_elements(), bbox_to_anchor = (1.35, 1))
    
    # Definindo o Rótulo dos Eixos x, y e z de acordo com os índices, sendo Age, Annual Income (k$) e Spending Score (1-100) respectivamente
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    # Definindo o Título 
    ax.set_title("Viewing the Clusters")
    # Exibindo o gráfico
    plt.show()
    

# ============================================================================================================
# Criando uma Figura Completa com projeção em 2 dimensões para a visualização dos Clusters após ter passado pelo processo de PCA
# Obs.: tornando a função mais genérica para poder ser utilizada em quaisquer outros Scripts
# ============================================================================================================
def plot_clusters_2D(
    # Passando os parâmetros para a função
    dataframe,
    columns,
    n_colors,
    centroids,
    show_centroids = True,
    show_points = False,
    column_clusters = None,
):
    """Gerar gráfico 2D com os clusters.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    n_colors : int
        Número de cores para o gráfico.
    centroids : np.ndarray
        Array com os centroides.
    show_centroids : bool, opcional
        Se o gráfico irá mostrar os centroides ou não, por padrão True
    show_points : bool, opcional
        Se o gráfico irá mostrar os pontos ou não, por padrão False
    column_clusters : List[int], opcional
        Coluna com os números dos clusters para colorir os pontos
        (caso mostrar_pontos seja True), por padrão None
    """

    # Criando a figura
    fig = plt.figure()

    # Criando um sistema de eixos e adicionando um subplot à figura
    # Obs.: passando o parâmetro 111
    ax = fig.add_subplot(111)

    # Atribuindo à variável "cores" a quantidade de cores a serem utilizadas do color map da paleta de cores tab10 do matplotlib
    # Obs.: funciona para qualquer quantidade de cores que for passada como parâmetro da função
    cores = plt.cm.tab10.colors[:n_colors]
    
    # Sobreescrevendo a variável cores passando a variável cores original para o ListedColormap(cores)
    cores = ListedColormap(cores)

    # Definindo a lista de colunas do DataFrame de acordo com os índices
    y = dataframe[columns[1]]

    # Mostrar os Centróides: usar parâmetro True
    ligar_centroids = show_centroids
    
    # Mostrar os Pontos: usar parâmetro True
    ligar_pontos = show_points

    # Percorrendo as variáveis "i" e "centroid" em enumerate(centroids)
    for i, centroid in enumerate(centroids):
        # Se for verdadeiro
        if ligar_centroids: 
            # Mostrando as coordenadas dos pontos x, y e z no gráfico de dispersão
            # Obs.: fazendo o Unpacking Implícito, passando o tamanho dos pontos no parâmetro s = 500 e alpha = 0.5
            ax.scatter(*centroid, s = 500, alpha = 0.5)
            # Mostrando nas coordenadas os números dos clusters
            # Obs.:passando os parâmetros tamanho da fonte e alinhamento horizontal/vertical
            ax.text(*centroid,
                    f"{i}", 
                    fontsize = 20, 
                    horizontalalignment = "center", 
                    verticalalignment = "center",
            )
            
        # Se for verdadeiro
        if ligar_pontos:
            # Mostrando todos os demais pontos e atribuindo à variável "s" (s de scatter)
            # Obs.1: definindo as cores no parâmetro c = coluna_clusters
            # Obs.2: ajustando o mapa de cores no parâmetro cmap = cores
            s = ax.scatter(x, y, c = column_clusters, cmap = cores)
            # Passando os parãmetros de legenda do gráfico de dispersão
            # Obs.: ajustando a posição da legenda, fazendo a ancoragem de legenda para não sobrepor o gráfico: bbox_to_anchor = (1.3, 1)
            ax.legend(*s.legend_elements(), bbox_to_anchor = (1.35, 1))

    # Definindo o Rótulo dos Eixos x e y de acordo com os índices, respectivamente
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_title("Clusters")

    # Exibindo o gráfico
    plt.show()





# ============================================================================================================
#  Criando um Histograma com a proporção de clientes em cada Cluster
# Utilizando barras uma acima da outra proporcionalmente, como se fosse uma porcentagem e dando uma ideia de proporção, usando funcionalidades do HistPlot
# ============================================================================================================
def plot_columns_percent_by_cluster(
    # Passando os parâmetros para a função:
    dataframe,
    columns,
    rows_cols = (2, 3),
    figsize = (16, 10),
    column_cluster = "cluster",
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor por cluster.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tupla com o número de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e a altura da figura, por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna com os números dos clusters, por padrão "cluster"
    """
    
    # Criando uma figura em um sistema de eixos com 2 Linhas e 3 Colunas
    # Obs.1: usando o parâmetro sharey = True para que a escala de porcentagem seja exibida somente uma vez em cada linha do lado esquerdo e os demais gráficos compartilham essa mesma escala
    # Obs.2: excluindo o parâmetro tight_layout = True utilizado para ocupar o epaçamento sem sobrepor os gráficos, para obter o controle dos espaçamentos através do plt.subplots_adjust(hspace = 0.3, wspace = 0.3) vide abaixo
    fig, axs = plt.subplots(nrows = rows_cols[0], ncols = rows_cols[1], figsize = figsize, sharey = True)
    
    # Verificando se o eixo "axs" não é um array do numpy
    if not isinstance(axs, np.ndarray):
        # Transformando o eixo "axs" em um array do numpy
        axs = np.array(axs)

    # Percorrendo cada sistema de eixo e cada feature (coluna) na combinação para a par com o sistema de eixos e as colunas de análise
    for ax, col in zip(axs.flatten(), columns):
        # Plotando o HistPlot usando os parâmetros e atribuindo à variável "h" de Histogram
        # Obs.1: passando no eixo x a coluna cluster
        # Obs.2: passando no hue o valor da coluna que está sendo olhada
        # Obs.3: passando no conjunto de dados o dataframe
        # Obs.4: o sistema de eixo é o ax
        # Obs.5: passando "fill" no parâmetro multiple para preencher a coluna de baixo até em cima (ver documentação)
        # Obs.6: passando "percent" no parâmetro stat para o agregado estatístico que será computado em cada intervalo
        # Obs.7: passando "True" no parâmetro discrete para colocar a largura das barras do histograma no padrão de 1
        # Obs.8: passando o valor 0.8 no parâmetro shrink afim de comprimir as barras gerando um espaçamento entre elas
        h = sns.histplot(
                x = column_cluster,
                hue = col,
                data = dataframe,
                ax = ax,
                multiple = "fill",
                stat = "percent",
                discrete = True,
                shrink = 0.8,
        )
        
        # Criando a variável "n_clusters" de forma genérica para que possa se adaptar a um número variável de Clusters que se esteja trabalhando futuramente
        # Obs.: atribuindo à variável a quantidade de valores únicos existentes na coluna clusters do DataFrame usando o método .nunique() do Pandas
        n_clusters = dataframe[column_cluster].nunique()
        
        # Configurando os valores do eixo x dos histogramas de acordo com a quantidade de valores únicos existentes na coluna cluster do DataFrame
        h.set_xticks(range(n_clusters))
        
        # Formatando os valores do eixo y 
        # Obs.: passando o parâmetro "PercentFormatter(1)" para valores percentuais de 0% a 100%
        h.yaxis.set_major_formatter(PercentFormatter(1))
        
        # Eliminando o rótulo Percent no eixo y por ser redundante e desnecessário
        # Obs.: passando como parâmetro uma string vazia
        h.set_ylabel("")
        
        # Modificando os parâmetros de configuração dos ticks (tracinhos)
        # Obs.1: usando both no parâmetro axis para modificar ambos os eixos
        # Obs.2: usando both no parâmetro which para modificar os major's e os minor's
        # Obs.3: usando 0 no parâmetro length para o comprimento
        h.tick_params(axis = "both", which = "both", length = 0)
        
        # Mostrando os valores percentuais em cada barra, passando por cada bar em h.containers
        # Obs.1: h é um sistema de eixos com label = "clusters"
        # Obs.2: h.containers é uma Lista com 2 containers de barras [<BarContainer object of 3 artists>, <BarContainer object of 3 artists>] que são as 3 barras laranjas e as 3 barras azuis
        for bars in h.containers:
            # Colocando um rótulo em cada barra através dos parâmetros
            # Obs.1: passando a bar
            # Obs.2: passando o label_type = "center" para centralizar o rótulo
            # Obs.3: passando os valores (em cada container tem mais de uma barra) utilizando o parâmetro labels e um List Comprehension passando por várias barras
            # Obs.4: passando "white" para o parâmetro color, cor da fonte
            # Obs.5: passando "bold" para o parâmetro weight, negrito
            # Obs.6: passando o valor 11 para o parâmetro fontsize, tamanho da fonte
            h.bar_label(bars, label_type = "center", labels = [f'{b.get_height():.1%}' for b in bars], color = "white", weight = "bold", fontsize = 11)
        
        # Suprimindo as linhas de separação dentro das barras, melhorando a visualização
        for bar in h.patches:
            # Obs.: passando o valor 0 para o parâmetro .set_linewidth()
            bar.set_linewidth(0)

    # Modificando os ajustes de espaçamento entre cada um dos Histogramas, tanto na horizontal quanto na vertical
    # Obs.1: passando o parâmetro hspace para controlar os espaçamentos de altura (vertical)
    # Obs.2: passando o parâmetro wspace para controlar os espaçamentos de largura (horizontal)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
    # Exibindo o gráfico
    plt.show()




# ============================================================================================================
#  Criando um Histograma com a proporção de clientes em cada Hue Cluster
# Utilizando barras uma acima da outra proporcionalmente, como se fosse uma porcentagem e dando uma ideia de proporção, usando funcionalidades do HistPlot
# ============================================================================================================
def plot_columns_percent_hue_cluster(
    # Passando os parâmetros para a função:
    dataframe,
    columns,
    rows_cols = (2, 3),
    figsize = (16, 10),
    column_cluster = "cluster",
    palette = "tab10"
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor com cluster
    como hue.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tupla com o número de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e a altura da figura, por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna com os números dos clusters, por padrão "cluster"
    palette : str, opcional
        Paleta a ser utilizada, por padrão "tab10"
    """
    # Criando uma figura em um sistema de eixos com 2 Linhas e 3 Colunas
    # Obs.1: usando o parâmetro sharey = True para que a escala de porcentagem seja exibida somente uma vez em cada linha do lado esquerdo e os demais gráficos compartilham essa mesma escala
    # Obs.2: excluindo o parâmetro tight_layout = True utilizado para ocupar o epaçamento sem sobrepor os gráficos, para obter o controle dos espaçamentos através do plt.subplots_adjust(hspace = 0.3, wspace = 0.3) vide abaixo
    fig, axs = plt.subplots(nrows = rows_cols[0] , ncols = rows_cols[1], figsize = figsize, sharey = True)

    # Verificando se o eixo "axs" não é um array do numpy
    if not isinstance(axs, np.ndarray):
        # Transformando o eixo "axs" em um array do numpy
        axs = np.array(axs)
        
    # Percorrendo cada sistema de eixo e cada feature (coluna) na combinação para a par com o sistema de eixos e as colunas de análise
    for ax, col in zip(axs.flatten(), columns):
        # Plotando o HistPlot usando os parâmetros e atribuindo à variável "h" de Histogram
        # Obs.1: passando no eixo x o valor da coluna que está sendo olhada
        # Obs.2: passando no hue a coluna "cluster"
        # Obs.3: passando no conjunto de dados o dataframe
        # Obs.4: o sistema de eixo é o ax
        # Obs.5: passando "fill" no parâmetro multiple para preencher a coluna de baixo até em cima (ver documentação)
        # Obs.6: passando "percent" no parâmetro stat para o agregado estatístico que será computado em cada intervalo
        # Obs.7: passando "True" no parâmetro discrete para colocar a largura das barras do histograma no padrão de 1
        # Obs.8: passando o valor 0.8 no parâmetro shrink afim de comprimir as barras gerando um espaçamento entre elas
        # Obs.9: passando a paleta de cores tab10
        h = sns.histplot(
                x = col,
                hue = column_cluster,
                data = dataframe,
                ax = ax,
                multiple = "fill",
                stat = "percent",
                discrete = True,
                shrink = 0.8,
                palette = palette,
        )
        
        # Verificando se a coluna é Numérica
        # Obs.: se for Categórica não entratá no if e serão passados diretamente os valores das colunas categóricas
        if dataframe[col].dtype != "object":
            # Se for diferente de "object" pegar o número de entradas únicas das colunas numéricas
            h.set_xticks(range(dataframe[col].nunique()))
        
        # Formatando os valores do eixo y 
        # Obs.: passando o parâmetro "PercentFormatter(1)" para valores percentuais de 0% a 100%
        h.yaxis.set_major_formatter(PercentFormatter(1))
        
        # Eliminando o rótulo Percent no eixo y por ser redundante e desnecessário
        # Obs.: passando como parâmetro uma string vazia
        h.set_ylabel("")
        
        # Modificando os parâmetros de configuração dos ticks (tracinhos)
        # Obs.1: usando both no parâmetro axis para modificar ambos os eixos
        # Obs.2: usando both no parâmetro which para modificar os major's e os minor's
        # Obs.3: usando 0 no parâmetro length para o comprimento
        h.tick_params(axis = "both", which = "both", length = 0)
        
        # Mostrando os valores percentuais em cada barra, passando por cada bar em h.containers
        # Obs.1: h é um sistema de eixos com label = "clusters"
        # Obs.2: h.containers é uma Lista com 2 containers de barras [<BarContainer object of 3 artists>, <BarContainer object of 3 artists>] que são as 3 barras laranjas e as 3 barras azuis
        for bars in h.containers:
            # Colocando um rótulo em cada barra através dos parâmetros
            # Obs.1: passando a bar
            # Obs.2: passando o label_type = "center" para centralizar o rótulo
            # Obs.3: passando os valores (em cada container tem mais de uma barra) utilizando o parâmetro labels e um List Comprehension passando por várias barras
            # Obs.4: passando "white" para o parâmetro color, cor da fonte
            # Obs.5: passando "bold" para o parâmetro weight, negrito
            # Obs.6: passando o valor 11 para o parâmetro fontsize, tamanho da fonte
            h.bar_label(bars, label_type = "center", labels = [f'{b.get_height():.1%}' for b in bars], color = "white", weight = "bold", fontsize = 11)
        
        # Suprimindo as linhas de separação dentro das barras, melhorando a visualização
        for bar in h.patches:
            # Obs.: passando o valor 0 para o parâmetro .set_linewidth()
            bar.set_linewidth(0)
            
        # Capturando as informações de Legenda e atribuindo à variável "legend"
        legend = h.get_legend()
        
        # Removendo a Legenda
        legend.remove()
        
    # Inserindo os textos que estarão na Legenda a nível da Figura
    # Obs.: iterando os valores dos clusters e atribuindo à variável "labels"
    labels = [text.get_text() for text in legend.get_texts()]

    # Colocando efetivamente a Legenda da Figura
    # Obs.1: passando os parâmetros handles (caixas), labels (valores), loc (posição), title (título) e ncols (qtde de colunas)
    # Obs.2: a qtde de colunas será obtida de forma automática de acordo com a quantidade de valores únicos da coluna "clusters" do DataFrame
    fig.legend(handles = legend.legend_handles, labels = labels, loc = "upper center", title = "Clusters", ncols = dataframe[column_cluster].nunique())

    # Modificando os ajustes de espaçamento entre cada um dos Histogramas, tanto na horizontal quanto na vertical
    # Obs.1: passando o parâmetro hspace para controlar os espaçamentos de altura (vertical)
    # Obs.2: passando o parâmetro wspace para controlar os espaçamentos de largura (horizontal)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
    # Exibindo o gráfico
    plt.show()


