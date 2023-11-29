import numpy as np 
from scipy.special import softmax
import math


sentence = "Today is sunday"

vacabulary = ['Today', 'is', 'sunday', 'saturday']

# Initial Embedding (one-hot encoding):

x_1 = [1,0,0,0] # Today 
x_2 = [0,1,0,0] # is 
x_3 = [0,0,1,0] # Sunday
x_4 = [0,0,0,1] # Saturday

X_example = np.stack([x_1, x_2, x_3], axis=0)

W_Q = np.random.uniform(-1, 1, size=(4, 4))
W_K = np.random.uniform(-1, 1, size=(4, 4))
W_V = np.random.uniform(-1, 1, size=(4, 4))

Q = np.matmul(X_example, W_Q)
K = np.matmul(X_example, W_K)
V = np.matmul(X_example, W_V)

Attention_scores = np.matmul(Q, np.transpose(K))

# Renormalization: division by square root of model-dimension:

Attention_scores = Attention_scores / 2 

Softmax_Attention_Matrix = np.apply_along_axis(softmax, 1, Attention_scores)

One_Head_Attention = np.matmul(Softmax_Attention_Matrix,V)


# exit()

# ----------------------------------------------------------- 


class W_matrices:
    def __init__(self, n_lines, n_cols):
        self.W_Q = np.random.uniform(low=-1, high=1, size=(n_lines, n_cols))
        self.W_K = np.random.uniform(low=-1, high=1, size=(n_lines, n_cols))
        self.W_V = np.random.uniform(low=-1, high=1, size=(n_lines, n_cols))

    def print_W_matrices(self):
        print('W_Q : \n', self.W_Q)
        print('W_K : \n', self.W_K)
        print('W_V : \n', self.W_V)

class One_Head_Attention:
    def __init__(self, d_model, X):
        self.d_model = d_model
        self.W_mat = W_matrices(d_model, d_model)

        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)


    def compute_1_head_attention(self):
        Attention_scores = np.matmul(self.Q, np.transpose(self.K)) 
        print('Attention_scores before normalization : \n', Attention_scores)
        Attention_scores = Attention_scores / np.sqrt(self.d_model) 
        print('Attention scores after Renormalization: \n ', Attention_scores)
        Softmax_Attention_Matrix = np.apply_along_axis(softmax, 1, Attention_scores)
        print('result after softmax: \n', Softmax_Attention_Matrix)
        # print('Softmax shape: ', Softmax_Attention_Matrix.shape)

        result = np.matmul(Softmax_Attention_Matrix, self.V)
        print('softmax result multiplied by V: \n', result)


        return result

    def _backprop(self):
        # do smth to update W_mat
        pass

class Multi_Head_Attention:
    def __init__(self, n_heads, d_model, X):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_concat = self.d_model*self.n_heads # 4*8
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_model))
        # print('W_0 shape : ', self.W_0.shape)
        self.heads = []
        self.heads_results = []
        i = 0
        while i < self.n_heads:
            self.heads.append(One_Head_Attention(self.d_model, X))
            i += 1

    def print_W_0(self):
        print('W_0 : \n', self.W_0)

    def print_QKV_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].print_QKV()
            i += 1

    def print_W_matrices_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].W_mat.print_W_matrices()
            i += 1

    def compute(self):
        for head in self.heads:
            self.heads_results.append(head.compute_1_head_attention())
            # print('dimensao do resultado da head: ', self.heads_results[-1].shape)

        multi_head_results = np.concatenate(self.heads_results, axis=1)
        # print('multi_head_results shape = ', multi_head_results.shape)

        V_updated = np.matmul(multi_head_results, self.W_0)
        return V_updated

    def back_propagate(self):
        # backpropagate W_0
        # call _backprop for each head
        pass

class FFN:
    def __init__(self, V_updated, layer1_sz, layer2_sz):
        self.layer1_sz = layer1_sz
        self.layer2_sz = layer2_sz
        self.layer1_weights = np.random.uniform(low=-1, high=1, size=(V_updated.shape[1], layer1_sz))
        self.layer2_weights = np.random.uniform(low=-1, high=1, size=(layer2_sz, V_updated.shape[1]))

    def compute(self, V_updated):
        result = np.matmul(V_updated, self.layer1_weights)
        result = np.matmul(result, self.layer2_weights)

        return result

    def backpropagate_ffn(self):
        pass

class Positional_Encoding:
    def __init__(self, X):
        self.PE_shape = X.shape
        self.PE = np.empty(self.PE_shape)
        self.d_model = self.PE_shape[1]

    def compute(self, X):
        for i in range(self.PE_shape[0]): 
            for j in range(self.PE_shape[1]):
                self.PE[i,2*j] = math.sin(i/(10000**(2*j/self.d_model)))
                self.PE[i,2*j+1] = math.cos(i/(10000**(2*j/self.d_model)))
                #by this way we are assuming that the vectors are ordered stacked in X

        return X + self.PE


# -------------------------------------------
# -------------------------------------------

X = np.stack([x_1, x_2, x_3], axis=0)

multi_head_attention = Multi_Head_Attention(2, 4, X)

multi_head_attention.print_W_matrices_each_head()

multi_head_attention.print_QKV_each_head()

multi_head_attention.print_W_0()

V_updated_by_context = multi_head_attention.compute()

print(V_updated_by_context)
