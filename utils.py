import math
import random


def create_transition(beforeafter):
    p_before = []
    p_after = []
    if beforeafter:
        
        return p_before
    else:
        return p_after




def starforce(state):

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
        return (12,-1)

def create_cost(level):
    lst = []
    return lst

def simulation(nofsim,start_state,end_state,initial_price,level):

    cost_table = create_cost(level)
    meso_cost = [0] * nofsim
    destroy_num = [0] * nofsim
    

    for i in nofsim:
        cost_once = 0
        destroy_once = 0

        state = start_state
        while state != end_state:

            nex_state = starforce(state)
            cost_once += cost_table(state[0])
        
            if nex_state == (12,-1):
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


def calc_c_f_i():
    pass

def calc_c_2_f_i():
    c_f_i = calc_c_f_i()

    pass

def calc_with_markov():
    calc_c_f_i()
    calc_c_2_f_i()

    expectation = calc_c_f_i()
    variance = calc_c_2_f_i() - expectation **2

    return expectation, variance

####