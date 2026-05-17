#Implementing the core algorithm of flash attention
#Why can't I implement this in pytorch?  how can i fuse them all into one efficient I/O kernel?

#key techniques
'''
Tiling
Stable softmax
Online softmax
Kernel fusion
'''

#Understanding the dimensionalities
'''
    Br -> Q ; Bc -> K,V
    O, m, l are  linked with Q && operated upon inside the inner loop
'''

#pseudocode
'''
Decide block sizes
Initialize matrices Q,K,V and O
Initialize row wise statistics m and l
Split matrices K,V acc to b_c x d
Split matrix Q into T_r blocks (b_r x d each), and split m,l into T_r blocks

Load K,V block
    Load Q blocks
    Perform Q*K_t
    track row-wise m 
    perform softmax
    track row-wise l
    merge block stats
'''

import torch 
import time 
import math
import matplotlib.pyplot as plt

class FlashAttention(torch.nn.Module):
    #How do we add data members in Python?

    def __init__(self,M,d):  
        super().__init__()
        #deciding the block dimensions                       
        self.b_c = math.floor(M/(4*d))
        self.b_r = min(self.b_c,d)
        self.d = d
        #define layers  -- Linear projection
        self.weight_Q = torch.nn.Linear(d,d,bias=False)   #why do we not learn an additive bias
        self.weight_K = torch.nn.Linear(d,d,bias=False)
        self.weight_V = torch.nn.Linear(d,d,bias=False) 

    def forward(self,input_emb):
        Q = self.weight_Q(input_emb)    #BxNxd
        K = self.weight_K(input_emb)
        V = self.weight_V(input_emb)

        #Initialize intermediates and the output matrix
        O = torch.zeros(Q.shape[0],Q.shape[1],self.d,device=Q.device)    #BxNxd            #can use zeros_like
        l = torch.zeros(Q.shape[0],Q.shape[1],device=Q.device)          #BxN
        m = torch.full((Q.shape[0],Q.shape[1]),float('-inf'),device=Q.device)  

        #Split the original matrices without copying
        #Just split logically instead of introducing copying overhead
        for j in range(0,Q.shape[1],self.b_c):
            #loading tiled K,V blocks for the outer loop
            k_j,v_j = K[:,j:j+self.b_c,:],V[:,j:j+self.b_c,:]  #B x b_c x d                     #slicing along the key dimension, not the embedding dimensions            
            for i in range(0,Q.shape[1],self.b_r):
                #loading tiled Q,l,m,O blocks
                q_i = Q[:,i:i+self.b_r,:]
                o_i = O[:,i:i+self.b_r,:]                      #B x b_r x d
                m_i,l_i = m[:,i:i+self.b_r],l[:,i:i+self.b_r]

                #the first operation
                k_j_t = torch.transpose(k_j,1,2)                #B x d x b_c
                tiled_intermediate = (q_i @ k_j_t)/(self.d**0.5)               #B x b_r x b_c

                #row wise intermediate and softmax
                #Got to iterate over columns to get the row wise maximum
                m_ij,_ = torch.max(tiled_intermediate,dim=2)   #returns a tuple of values and indices
                
                online_softmax = torch.exp(tiled_intermediate-(m_ij.unsqueeze(-1)))     #How to match dimensions? Using unsqueeze to add a dummy dimension
                l_ij = online_softmax.sum(dim=2)                #B x b_r

                #Merging statistics
                m_new = torch.maximum(m_i,m_ij)
                
                prev_inf,current_inf = torch.exp(m_i-m_new),torch.exp(m_ij-m_new)
                l_new = (prev_inf)*l_i + (current_inf)*l_ij                        #Like a weighted average

                #Updating the rows in Oi by scaling (eventually converges to original attention's output)
                #line12 of the algorithm
                scale_prev = prev_inf*l_i                       # [B, Br]
                scale_prev= scale_prev.unsqueeze(-1)            # [B, Br, 1]
                term_prev  = scale_prev*o_i                      # [B, Br, d]

                output_ij = torch.matmul(online_softmax, v_j)  # [B, Br, d]
                scale_current = current_inf.unsqueeze(-1) # [B, Br, 1]
                term_current  = scale_current*output_ij            # [B, Br, d]

                #final output
                output_updated = (term_prev+term_current)/l_new.unsqueeze(-1)

                #Writing back calculated values to the HBM
                m[:,i:i+self.b_r]=m_new
                l[:,i:i+self.b_r]=l_new
                O[:,i:i+self.b_r,:]=output_updated

        #The final output        
        return O
    
def benchmark(input,eval):
    #warming up :? 
    for _ in range(3):
        eval(input)
    
    #benchmarking
    torch.cuda.reset_peak_memory_stats()                   #Reset GPU memory usage stats
    start=time.time()                                      #Returns the time in seconds since the epoch!
    with torch.no_grad():
        attention = eval(input)
    torch.cuda.synchronize()                               #Synchronise to prevent CPU form intervening, until all GPU work is done 
    stop=time.time()
    mem=torch.cuda.max_memory_allocated()/(1024*1024)      #GPU memory allocated since the last reset
    return stop-start,mem

def performance():
    batch,seq_len,emb_dim=1,0,512
    M,d=(65536,512)
    attention_calculation=FlashAttention(M,d).to('cuda')

    x_seq,y_time=[],[]
    mem_usage=[]

    for seq_len in range(512,10001,512):
        input = torch.randn(batch,seq_len,emb_dim,device='cuda')
        bench=benchmark(input,attention_calculation)
        x_seq.append(seq_len)
        y_time.append(bench[0])
        mem_usage.append(bench[1])
        print(f"Time taken for sequence length {seq_len} is -- {bench[0]}  memory used --- {bench[1]} MB")

    plt.plot(x_seq,y_time,color='blue',label='compute time')
    plt.title('Compute time as a function of sequence length')
    plt.show()
    plt.plot(x_seq,mem_usage,color='red',label='memory usage')
    plt.title('Memory usage as a function of sequence length')
    plt.show()

performance()

'''
Further directives
    Work on the backward pass using autograd
    How to visualize the computation graph
    Add control for irregular last block case
'''

'''
Observations
    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_mm) 
    
    Right now, Flash Attention is just a much clunkier and inefficient version of single headed attention
    
    To observe the real benefits of Flash Attention I must implment this in CUDA and fuse operations to a single kernel
'''