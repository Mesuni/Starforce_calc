import math
import random
import numpy as np


def cal_coff(num):

    coff = (1 - num*1.05)/(1-num)
    return coff


def create_transition(isbefore,anti_destroy = [] ,starcatch = [] ,is1516 = False ,isdestroy = False):
    # possible anti_destroy : [15,16] if isbefore else [15,16,17]
    # we map the state (i,j) -> 3i+j where i \in {0,1,...29} , j \in {0,1,2}
    # The matrix size can be reduced for example (3,1) state cannot be reached since in the position 4 there is no failing stars but...
    # before 0 ~ 12 : 0,0 1,0, ... 15,0 15,2 16,0 : 17 , 16,1 : 18 .. (i,j) : 3i+j - 31 ... 
    # (18,2) (19,1) (19,2) (20,1) : not exist then (25,0) : 75-36 = 39 (12,2) : -1
    # after 0 ~ 30 : 0,0 1,0 ... (12,2) : 31 
    if isbefore:
        anti_destroy = anti_destroy[:2]

    
    p_before = [[0 for i in range(39)]for j in range(39)]
    p_after = [[0 for i in range(32)]for j in range(32)]


    if isbefore:
        
        for i in range(2):
            p_before[i][i+1] = 0.95-0.05*i
            p_before[i][i] = 1-p_before[i][i+1]
        for i in range(2,15):
            p_before[i][i+1] = 1-0.05*i
            p_before[i][i] = 1-p_before[i][i+1]
        
        # 15성
        p_before[15][17] = 0.3
        p_before[15][-1] = 0.021
        p_before[15][15] = 1-0.321

        p_before[16][17] = 1
        
        # 16,17성
        for i in range(17,22,3):
            p_before[i][i+3] = 0.3
            p_before[i][-1] = 0.021
            p_before[i][i-2] = 1- 0.321
            
            p_before[i+1][i+3] = 0.3
            p_before[i+1][-1] = 0.021
            p_before[i+1][i-2] = 1- 0.321
            
            p_before[i+2][i+3] = 1

        #18성
        p_before[23][25] = 0.3
        p_before[23][-1] = 0.028
        p_before[23][21] = 1-0.328

        p_before[24][25] = 0.3
        p_before[24][-1] = 0.028
        p_before[24][22] = 1-0.328

        #19성
        p_before[25][26] = 0.3
        p_before[25][-1] = 0.028
        p_before[25][24] = 1-0.328

        #20성
        p_before[26][28] = 0.3
        p_before[26][-1] = 0.07
        p_before[26][26] = 1-0.37

        p_before[27][28] = 1

        #21성
        p_before[28][31] = 0.3
        p_before[28][-1] = 0.07
        p_before[28][26] = 1-0.37

        p_before[29][31] = 0.3
        p_before[29][-1] = 0.07
        p_before[29][27] = 1-0.37

        p_before[30][31] = 1

        #22성
        p_before[31][34] = 0.03
        p_before[31][-1] = 0.194
        p_before[31][29] = 1-0.224

        p_before[32][34] = 0.03
        p_before[32][-1] = 0.194
        p_before[32][30] = 1-0.224

        p_before[33][34] = 1

        #23성
        p_before[34][36] = 0.02
        p_before[34][-1] = 0.294
        p_before[34][32] = 1-0.314

        p_before[35][36] = 0.02
        p_before[35][-1] = 0.294
        p_before[35][33] = 1-0.314

        #24성
        p_before[36][37] = 0.01
        p_before[36][-1] = 0.396
        p_before[36][35] = 1-0.406

        #25성
        #p_before[37][37] = 1
        
        # 터짐
        p_before[-1][12] = 1


        
        
        for i in starcatch:
            ind,nextind,dropind = state_index(state=(i,0),isbefore=True,transition=p_before)
            y = cal_coff(p_before[ind][nextind])

            p_before[ind][nextind] = p_before[ind][nextind] * 1.05
            p_before[ind][dropind] = p_before[ind][dropind] * y
            p_before[ind][-1] = p_before[ind][-1] * y 

        for i in anti_destroy:
            ind,nextind,dropind = state_index(state=(i,0),isbefore=True,transition=p_before)
            p_before[ind][dropind] += p_before[ind][-1]
            p_before[ind][-1] =0

        if is1516:
            p_before[5][6] = 1
            p_before[5][5] = 0
        
            p_before[10][11] = 1
            p_before[10][10] = 0

            p_before[15][17] = 1
            p_before[15][15] = 0
            p_before[15][-1] = 0


            

        return p_before
    else:

        for i in range(2):
            p_after[i][i+1] = 0.95-0.05*i
            p_after[i][i] = 1-p_after[i][i+1]
        for i in range(2,15):
            p_after[i][i+1] = 1-0.05*i
            p_after[i][i] = 1-p_after[i][i+1]
        
        
        for i in range(15,17):
            p_after[i][i+1] = 0.3
            p_after[i][i] = 0.679
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]

        for i in range(17,19):
            p_after[i][i+1] = 0.15
            p_after[i][i] = 0.782
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]


        for i in range(19,20):
            p_after[i][i+1] = 0.15
            p_after[i][i] = 0.765
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]

        for i in range(20,21):
            p_after[i][i+1] = 0.3
            p_after[i][i] = 0.595
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]

        for i in range(21,23):
            p_after[i][i+1] = 0.15
            p_after[i][i] = 0.7225 - 0.0425*(i-21)
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]


        for i in range(23,26):
            p_after[i][i+1] = 0.1
            p_after[i][i] = 0.72 
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]

        for i in range(26,30):
            p_after[i][i+1] = 0.07-0.02*(i-26)
            p_after[i][i] = 0.7440 - 0.016*(i-26)
            p_after[i][-1] = 1-p_after[i][i+1]-p_after[i][i]

        
        #p_after[30][30] = 1
        
        p_after[-1][12] = 1

        for i in anti_destroy:
            p_after[i][i] += p_after[i][-1]
            p_after[i][-1] =0
        
        if isdestroy:
            for i in range(15,30):
                p_after[i][-1] = p_after[i][-1] * 0.7
                p_after[i][i] = 1-p_after[i][i+1]-p_after[i][-1]

        for i in starcatch:
            y = cal_coff(p_after[i][i+1])
            p_after[i][i+1] = p_after[i][i+1] * 1.05
            p_after[i][i] = p_after[i][i-1] * y
            p_after[i][-1] = p_after[i][-1] * y 

        return p_after



def state_transform(state,isbefore):
    # transform (,) to index
    if isbefore:
        notgoodcase = dict([
                            ((15,0),15),
                            ((15,2),16),
                            ((19,0),25),
                            ((20,0),26),
                            ((20,2),27),
                            ((24,0),36),
                            ((25,0),37),
                            ((-1,-1),-1),
                            
                            ])

        if state in notgoodcase.keys():
            return notgoodcase[state]
        if state[0]<15:
            return state[0]
        if 15<=state[0]<=18:
            return 3*state[0] + state[1] - 31
        if 20<=state[0]<=23:
            return 3*state[0] + state[1] - 35
        
    else:
        if state[1] == -1:

            return -1
        else:
            return state[0]


def state_index(state,isbefore,transition):
    ind = state_transform(state,isbefore = isbefore)
    P = transition[ind]

    nextposition = 0
    dropind = ind
    for e, prob in enumerate(P[:-1]):
        if prob and e<ind:
            dropind = e
        if prob and e>ind:
            nextposition = e

    return ind,nextposition,dropind

def starforce(state,transition):

    P_ij = transition

    if len(P_ij) == 39:
        isbefore = True
    else:
        isbefore = False

    if state[1] == 2:
        return (state[0]+1,0)
    if state[1] == -1:
        return (12,0)
    
    ind = state_transform(state=state,isbefore=isbefore)

    drop = False

    nextposition = 0
    for e, prob in enumerate(P_ij[ind][:-1]):
        if prob and e<ind:
            drop = True
        if prob and e>ind:
            nextposition = e
    
    success, destroy = P_ij[ind][nextposition] , P_ij[ind][-1]
    fail = 1-(success+destroy)
    s = random.random()
    if s<success:
        return (state[0]+1,0)
    elif s<success + fail:
        if drop:
            return (state[0]-1,state[1] + 1)
        else:
            return (state[0],state[1])
    else:
        return (-1,-1)



def create_cost(level,isbefore,initial_price,mvprank = 0,ispc = False,is30per = False, is1516 = False,anti_destroy = []):
    # mvprank 0 : bronze, 1 : silver, 2 : gold, 3 : diamond, 4 : red
    # possible anti_destroy : [15,16] if isbefore else [15,16,17]

    if isbefore:
        anti_destroy = anti_destroy[:2]

    mvptable = [0, 0.03, 0.05, 0.1, 0.1]
    mvprate = mvptable[mvprank]
    pc = 0.05
    
    costlst = [0]*27 if isbefore else [0] * 32

    if isbefore:
        for i in range(0,10):
            costlst[i] = 1000 + ((level**3)*(i+1))/36
        for i in range(10,11):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/571
        for i in range(11,12):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/314
        for i in range(12,13):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/214
        for i in range(13,14):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/157
        for i in range(14,15):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/107
        for i in range(15,25):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/200
        costlst[25] = 0
        costlst[26] = initial_price
    
    else:
        for i in range(0,10):
            costlst[i] = 1000 + ((level**3)*(i+1))/36
        for i in range(10,11):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/571
        for i in range(11,12):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/314
        for i in range(12,13):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/214
        for i in range(13,14):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/157
        for i in range(14,15):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/107
        for i in range(15,17):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/200
        for i in range(17,18):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/150
        for i in range(18,19):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/70
        for i in range(19,20):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/45
        for i in range(20,21):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/200
        for i in range(21,22):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/125
        for i in range(23,30):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/200
        costlst[30] = 0
        costlst[31] = initial_price
        
    
    discount = 1-(mvprate+pc) if ispc else 1-mvprate
    anti_destroy_cost = [ (i,round(costlst[i],-2)) for i in anti_destroy]

    costlst = [round(i * discount * 0.7,-2) for i in costlst] if is30per else [round(i * discount,-2) for i in costlst]

    for i,cost in anti_destroy_cost:
        if isbefore:
            if is1516 and i == 15:
                continue
            costlst[i] += cost
        else: 
            costlst[i] += 2*cost

    return costlst

def simulation(numofsim,start_state,end_state,initial_price,level,isbefore,
               mvprank = 0,ispc = False,is30per = False,is1516 = False ,isdestroy = False,
               anti_destroy = [],starcatch = [],isshining = False):

    # mvprank 0 : bronze, 1 : silver, 2 : gold, 3 : diamond, 4 : red
    # possible anti_destroy : [15,16] if isbefore else [15,16,17]
    if isshining:
        is30per = True
        is1516 = True
        isdestroy = True
    
    
    end_state = (end_state,0)

    cost_table = create_cost(level,isbefore=isbefore,initial_price=initial_price,mvprank=mvprank,ispc=ispc,is30per=is30per,is1516=is1516,anti_destroy=anti_destroy)
    meso_cost = [0] * numofsim
    destroy_num = [0] * numofsim
    P_ij = create_transition(isbefore=isbefore,anti_destroy=anti_destroy,starcatch=starcatch,is1516=is1516,isdestroy=isdestroy)


    for i in range(numofsim):
        cost_once = 0
        destroy_once = 0

        
        state = (start_state,0)

        while True:

            nex_state = starforce(state,transition=P_ij)
            cost_once += cost_table[state[0]]
            
        
            if nex_state == (-1,-1):
                destroy_once += 1

            state = nex_state

            if state == end_state:

                break
        meso_cost[i] = cost_once
        destroy_num[i] = destroy_once
    return meso_cost, destroy_num
    

def calc_sample_index(mesos,destroys):
    
    assert len(mesos) != len(destroys)

    n = len(mesos)
    sample_meso_mean = sum(mesos)/n
    sample_destroy_mean = sum(destroys)/n

    sample_meso_var = sum((mesos-sample_meso_mean)^2)/(n-1)
    sample_destroy_var = sum((destroys-sample_destroy_mean)^2)/(n-1)

    sample_meso_sigma = math.sqrt(sample_meso_var)
    sample_dest_sigma = math.sqrt(sample_destroy_var)

    return sample_meso_mean, sample_destroy_mean, sample_meso_sigma, sample_dest_sigma


def calc_c_f_i(initial_state,final_state,isbefore,initial_price):
    C = create_cost(level = 200, initial_price=initial_price,isbefore= isbefore, is30per= True, mvprank = 0, ispc = True)
    P = create_transition(isbefore= isbefore,anti_destroy= [], starcatch= [], is1516 = False, isdestroy= True)
    ini = state_transform((initial_state,0),isbefore=isbefore) 
    fin = state_transform((final_state,0),isbefore=isbefore) 

    
    Pminus = P-np.identity(len(P))
    Pminus[fin] = [0 if i !=fin else 1 for i in range(len(P))]
    C_prime = [C[state_transform((i,0),isbefore=isbefore)] for i in range(len(P)-1)] + [C[-1]]
    C_prime[fin] = 0

    C_prime = np.array(C_prime)
    ans = np.linalg.inv(Pminus)@(-1 * C_prime.T) 

    return ans,Pminus,C_prime

def constant_calc(cf,P,C):
    P = P + np.identity(len(P))
    T = P@cf
    const_matrix = [c**2 + 2*c*t  for c,t in zip(C,T)]
    return const_matrix

def calc_c_2_f_i(initial_state,final_state,isbefore,initial_price):
    c_f_i,Pminus,C_prime = calc_c_f_i(initial_state,final_state,isbefore = isbefore,initial_price=initial_price)
    cc = constant_calc(c_f_i,Pminus,C_prime)

    ini = state_transform((initial_state,0),isbefore=isbefore) 
    fin = state_transform((final_state,0),isbefore=isbefore)
    

    cc = np.array(cc)
    ans = np.linalg.inv(Pminus)@(-1 * cc)

    return ans

def calc_with_markov():
    calc_c_f_i()
    calc_c_2_f_i()

    expectation = calc_c_f_i()
    variance = calc_c_2_f_i() - expectation **2

    return expectation, variance

####