import math
import random

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
    p_after = [[0 for i in range(32)]for j in range(39)]


    if isbefore:
        
        for i in range(2):
            p_before[i][i+1] = 0.95-0.05*i
            p_before[i][i] = 1-p_before[i][i+1]
        for i in range(2,15):
            p_before[i][i+1] = 1-0.05*i
            p_before[i][i] = 1-p_before[i][i+1]
        
        # 15성
        p_before[15][17] = 0.03
        p_before[15][-1] = 0.021
        p_before[15][15] = 1-0.321

        p_before[16][17] = 1
        
        # 16,17성
        for i in range(17,22,3):
            p_before[i][i+3] = 0.3
            p_before[i][-1] = 0.021
            p_before[i][i-2] = 1- 0.321
            
            p_before[i+1][i+3] = 0.3
            p_before[i][-1] = 0.021
            p_before[i][i-2] = 1- 0.321
            
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
        p_before[37][37] = 1


        
        
        for i in starcatch:
            y = cal_coff(p_before)
            p_before[i][i+1] = p_before[i][i+1] * 1.05
            p_before[i][i-1] = p_before[i][i-1] * y
            p_before[i][-1] = p_before[i][-1] * y 

        if is1516:
            p_before[5][6] = 1
            p_before[5][5] = 0
        
            p_before[10][11] = 1
            p_before[10][10] = 0

            p_before[15][16] = 1
            p_before[15][15] = 0
            p_before[15][-1] = 0


            

        return p_before
    else:
        for i in range(15):
            p_after[i][i] = 0.3
        for i in range(15,30):
            p_after[i][i] = 0.3
            p_after[i][-1] = 0.7
        
        p_after[30][30] = 1
        
        if isdestroy:
            p_after[16][13] = p_after[15][13] * 0.7
            p_after[16][16] = p_after[16][16]
            p_after[16][17] = p_after[16][17]

        return p_after




def starforce(state,isbefore, anti_destroy = [], is1516 = False, isdestroy = False):

    create_transition(isbefore=isbefore,anti_destroy=anti_destroy,is1516=is1516,isdestroy=isdestroy)


    if state[1] == 2:
        return (state[0]+1,0)

    s = random.random(0,10)
    
    if s>1: # 성공
        return (state[0]+1,0)
    if s<1: # 실패
        if state[0] <=15:
            return state
        else:
            return (state[0]-1,(state[1]+1)%3)
        
    if s==0: # 파괴
        return (12,2)

def create_cost(level,isbefore,mvprank = 0,ispc = False,is30per = False, is1516 = False,anti_destroy = []):
    # mvprank 0 : bronze, 1 : silver, 2 : gold, 3 : diamond, 4 : red
    # possible anti_destroy : [15,16] if isbefore else [15,16,17]

    if isbefore:
        anti_destroy = anti_destroy[:2]

    mvptable = [0, 0.03, 0.05, 0.1, 0.1]
    mvprate = mvptable[mvprank]
    pc = 0.05
    costlst = [0]*29

    if isbefore:
        for i in range(0,10):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/36
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
    
    else:
        for i in range(0,10):
            costlst[i] = 1000 + ((level**3)*(i+1)**2.7)/36
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
        
    
    discount = 1-(mvprate+pc) if ispc else 1-mvprate
    anti_destroy_cost = [ (i,costlst[i]) for i in anti_destroy]

    costlst = costlst * discount * 0.7 if is30per else costlst * discount

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
               anti_destroy = [],isshining = False):

    if isshining:
        is30per = True
        is1516 = True
        isdestroy = True

    cost_table = create_cost(level,isbefore=isbefore,mvprank=mvprank,ispc=ispc,is30per=is30per,is1516=is1516,anti_destroy=anti_destroy)
    meso_cost = [0] * numofsim
    destroy_num = [0] * numofsim
    

    for i in numofsim:
        cost_once = 0
        destroy_once = 0

        state = start_state,0

        while state != end_state:

            nex_state = starforce(state,is1516=is1516,isdestroy=isdestroy)
            cost_once += cost_table(state[0])
        
            if nex_state == (12,2):
                cost_once += initial_price
                destroy_once += 1

            state = nex_state
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


def calc_c_f_i(initial_state,final_state):
    pass

def calc_c_2_f_i(initial_state,final_state):
    c_f_i = calc_c_f_i(initial_state,final_state)

    pass

def calc_with_markov():
    calc_c_f_i()
    calc_c_2_f_i()

    expectation = calc_c_f_i()
    variance = calc_c_2_f_i() - expectation **2

    return expectation, variance

####