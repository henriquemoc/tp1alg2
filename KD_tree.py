import numpy as np

class NoArvore:
    def __init__(self, valor, esquerda=None, direita=None):
        self.valor = valor #no caso de ser folha, salva o ponto e a classe dele. No caso de não ser, salva o eixo em que partiu os pontos e o valor nesse eixo
        self.esquerda = esquerda
        self.direita = direita

    def __str__(self):
        if self.esquerda!=None and self.direita!=None:
            return f"{self.esquerda.valor} - {self.valor} - {self.direita.valor}"
        elif self.esquerda==None and self.direita==None:
            return f"{self.esquerda} - {self.valor} - {self.direita}"
        elif self.esquerda==None:
            return f"{self.esquerda} - {self.valor} - {self.direita.valor}"
        elif self.direita==None:
            return f"{self.esquerda.valor} - {self.valor} - {self.direita}"
        else:
            return f"None"

    def isFolha(self):
        if self.direita==None and self.esquerda==None:
            return True
        else:
            return False

class ArvoreKD:
    def __init__(self, P):
        self.raiz = self.montarArvore(P, 0)

    def montarArvore(self, P, depth):
        
        if len(P)==0:
            return None
        
        k = len(P[0]['ponto'])
        axis = depth % k
        axis_str = ""
        if len(P)==1:
            return NoArvore(P[0])
        else:
            axis_str = str(axis)
            sorted_P = sorted(P, key=lambda x: x['ponto'][axis])

            size = len(sorted_P)
            res = 0
            ind = 0
            if size%2==0:
                ind = int((size/2)-1)
                res = (sorted_P[ind]['ponto'][axis] + sorted_P[ind+1]['ponto'][axis])/2
            else:
                ind = size//2
                res = sorted_P[ind]['ponto'][axis]
            l = res
            P1,P2 = [],[]
            P1 = sorted_P[:ind+1]
            P2 = sorted_P[ind+1:]

        v_esquerda = self.montarArvore(np.array(P1),depth+1)
        v_direita = self.montarArvore(np.array(P2),depth+1)
        v = NoArvore((axis_str, l), v_esquerda, v_direita)
        return v
