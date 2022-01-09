from KNN import X_NN
import numpy as np
import sys

def ler_arquivo(file):
    info = []
    points = []
    classes = []
    classes_name = []
    with open(file) as f:
        for line in f:
            if line[0]=="@":
                info.append(line[:-1])
                if "outputs" in line:
                    classes_name = line.split(' ')[1][:-1]
                                       
            else:
                line = line[:-1]
                aux = line.split(',')
                for i in range(len(aux)):
                    if i!=len(aux)-1: #se não for atributo classe
                        aux[i] = float(aux[i])
                classe = str(aux[-1].strip())
                point = aux[:-1]
                points.append({"ponto":point, "classe":classe})

    if classes == []:
        for line in info:
            if f"{classes_name}" in line:
                classes = line.split('{')[1]
                classes = classes.split('}')[0]
                classes = classes.split(',')
                break

    classes = [i.strip() for i in classes]

    return points, classes


'''def euclidian_distance(A, B):
    assert len(A)==len(B), "Tentando calcular distância de pontos de dimensões diferentes"
    soma = sum((A[i]-B[i])**2 for i in range(len(A)))
    dist = np.sqrt(soma)
    return dist   '''


def eliminar_duplicados(P):
    P_aux = []
    pontos_adicionados = []
    for i in range(len(P)):
        if P[i]['ponto'] not in pontos_adicionados:
            P_aux.append(P[i])
            pontos_adicionados.append(P[i]['ponto'])
    return P_aux
    
    
def separar_treino_teste(points):
    P = points.copy()
    k = len(P)
    tamanho_treino = (7*k)//10
    treino = []
    for i in range(tamanho_treino):
        indice = np.random.randint(len(P))
        treino.append(P[indice])
        del P[indice]
    teste = P
    return treino, teste


if __name__ == "__main__":

    assert len(sys.argv) == 3, "Necessario 2 argumentos(caminho do arquivo de pontos e o valor de k para o kNN)"
    
    arquivo = sys.argv[1].strip()
    k = int(sys.argv[2].strip())
    
    P, classes_geral = ler_arquivo(arquivo)
    P = eliminar_duplicados(P)
    treino, teste = separar_treino_teste(P)
    
    xNN = X_NN(treino, teste, classes_geral, k)
    
    precisao = xNN.precisao()
    revocacao = xNN.revocacao()
    acuracia = xNN.acuracia()
    
    print(f"Precisao por classe: {precisao}")
    print(f"Precisao media: {str(sum(precisao)/len(precisao))}")
    print(f"Revocacao por classe: {revocacao}")
    print(f"Revocacao media: {str(sum(revocacao)/len(revocacao))}")
    print(f"Acuracia: {acuracia}")
