from KD_tree import ArvoreKD
import numpy as np

class X_NN:
    def __init__(self, treino, teste, labels, x = 2):
        self.arvore_instancia = ArvoreKD(treino)
        self.arvore = self.arvore_instancia.raiz
        self.treino = treino
        self.teste = teste
        self.labels_possiveis = labels
        self.x = x
        self.resultado = {   
            tuple(query_p['ponto']): (query_p['classe'], self.__nearest_neighbor_classifier(query_p))
            for query_p in teste
        }
    
    def acuracia(self):
        tp_tn = sum((value[1]==value[0]) for key,value in self.resultado.items())
        total = len(self.teste)
        return tp_tn/total

    def precisao(self):
        precisao = []
        for i in range(len(self.labels_possiveis)):
            tp = sum((value[0]==self.labels_possiveis[i] and value[1]==value[0]) for key,value in self.resultado.items())
            fp_tp = sum((value[1]==self.labels_possiveis[i]) for key,value in self.resultado.items())
            if fp_tp==0:
                precisao.append(0.0)
            else:
                precisao.append(tp/fp_tp)
        return precisao

    def revocacao(self):
        revocacao = []
        for i in range(len(self.labels_possiveis)):
            tp = sum((value[0]==self.labels_possiveis[i] and value[1]==value[0]) for key,value in self.resultado.items())
            fn_tp = sum((value[0]==self.labels_possiveis[i]) for key,value in self.resultado.items())
            if fn_tp==0:
                revocacao.append(0.0)
            else:
                revocacao.append(tp/fn_tp)
        return revocacao

    def __find_nearest_neighbor(self, ponto):
        '''
          * Recebe um ponto de teste
          * Retorna os x vizinhos mais próximos desse ponto de teste
        '''
        def distancia_euclidiana_quadrado(A, B):#distância euclidiana ao quadrado. Economiza tempo de execução ao invés de calcular a distancia euclidiana normal, porque tirar raiz quadrada deixa os números mais complexos de calcular.
            assert len(A)==len(B), "Tentando calcular distância de pontos de dimensões diferentes"
            soma = sum((A[i]-B[i])**2 for i in range(len(A)))
            return soma
        
        def encontrar_indice(k_best_neighbors, distancia, valor):#função auxiliar ao processo usada em percorrer_arvore
            k_best_neighbors_copy = k_best_neighbors.copy()
            indice = 0
            for i in range(len(k_best_neighbors_copy[0])): #verificar em qual indice da lista o ponto atual se encaixa
                if distancia<k_best_neighbors_copy[0][i][1]:
                    indice = i
                    k_best_neighbors_copy[0].insert(indice, (valor, distancia))
                    break
            return k_best_neighbors_copy
        
        k = len(ponto)
        k_best_neighbors = None
        k_best_neighbors = [[], 0] #fila de prioridade
        def percorrer_arvore(node, depth):
            '''
              * Percorre a árvore recursivamente para encontrar os x vizinhos mais próximos.
              * Quando o nó não é folha e a fila de prioridades está cheia, foi imposta uma restrição que se 
                o nó, que representa uma reta de corte, for mais distante do ponto de teste que o último ponto
                da fila de prioridades, não vale a pena ir para o lado do corte em que o ponto de teste não está,
                já que os pontos lá nunca entrarão na fila.
            '''

            nonlocal k_best_neighbors #referencia a fila de prioridade fora da função
            
            if node.isFolha():
                distance = distancia_euclidiana_quadrado(tuple(node.valor['ponto']), ponto['ponto'])
                if k_best_neighbors[1]<self.x: #se a lista de melhores ainda não tiver x pontos
                    if k_best_neighbors[1]==0: #se tiver 0 pontos
                        k_best_neighbors[0].append((node.valor, distance))
                        k_best_neighbors[1] = k_best_neighbors[1] + 1
                    else:
                        if k_best_neighbors[0][-1][1]<distance: #se a distância ao ponto mais distante na lista for menor que a distância ao ponto atual, colocar o novo ponto na última posição
                            k_best_neighbors[0].append((node.valor, distance))
                            k_best_neighbors[1] = k_best_neighbors[1] + 1
                        else: #nesse caso, o novo ponto deve ser colocado no meio da lista
                            k_best_neighbors = encontrar_indice(k_best_neighbors.copy(), distance, node.valor)
                else:#se a lista já tiver x pontos
                    if distance<k_best_neighbors[0][-1][1]: #se a distância ao novo ponto for menor que a distancia ao último ponto da lista dos mais próximos, deve-se substituir
                        k_best_neighbors = encontrar_indice(k_best_neighbors.copy(), distance, node.valor)
                        k_best_neighbors[0].pop()
                    else:
                        pass

                return

            else:
                if k_best_neighbors[1]==self.x:
                    axis = depth % k
                    dist_ponto_corte = ponto['ponto'][axis] - node.valor[1]
                    
                    if dist_ponto_corte <= 0:
                        lado_ponto_teste = node.esquerda
                        lado_contrario = node.direita
                    else:
                        lado_ponto_teste = node.direita
                        lado_contrario = node.esquerda
                    
                    percorrer_arvore(node=lado_ponto_teste, depth=depth+1)
                    if dist_ponto_corte**2 < k_best_neighbors[0][-1][1]: #Se a distância do ponto do teste até o corte for menor que a distancia do ponto mais distante da fila de prioridade, vale a pena ir para o outro lado da árvore. Senão, não vale.
                        percorrer_arvore(node=lado_contrario, depth=depth+1)
                    else:
                        pass
                else:
                    percorrer_arvore(node=node.esquerda, depth=depth+1)
                    percorrer_arvore(node=node.direita, depth=depth+1)
            
        percorrer_arvore(node=self.arvore, depth=0)
        return k_best_neighbors[0]

    def __nearest_neighbor_classifier(self, ponto):
        '''
            * Recebe um ponto de teste
            * Retorna a classe estimada do ponto de teste

            * Tendo os k vizinhos mais próximos, verifica qual a classe predominante entre esses vizinhos e atribui essa classe ao ponto de teste. 
        '''
        k_best_neighbors = self.__find_nearest_neighbor(ponto=ponto)
        classes = {}
        for neighbor in k_best_neighbors:
            if neighbor[0]['classe'] in classes.keys():
                classes[neighbor[0]['classe']] = classes[neighbor[0]['classe']]+1
            else:
                classes[neighbor[0]['classe']] = 1
        return max(classes, key=classes.get)