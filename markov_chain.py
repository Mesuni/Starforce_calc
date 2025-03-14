import math
import utils as utils
import argparse


'''
transition matrix를 정의하고 
cost vector를 불러온다. 
그리고 markov chain에서 계산한 결과를 바탕으로 
실제 평균, 편차를 도출해보자
'''


if __name__ == '__main__':
    parser = argparse.PARSER()
    # startstate 어쩌구다 받아오기
    utils.create_cost(parser.level)

    utils.create_transition(parser.beforeafter)

    # utils.calculate_markov(cost,transition)
