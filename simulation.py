import math
import utils as utils
import argparse


'''
이번에는 시뮬레이션으로
평균이랑 표준편차를 도출해보자
표준편차는 표본 표준편차가 되겠지
'''


if __name__ == '__main__':
    
    parser = argparse.PARSER() # 시뮬레이션 argment 받기

    meso, destroy = utils.simulation(parser.num,parser.beforeafter)
    

    utils.calc_sample_index(meso,destroy)