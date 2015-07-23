
import numpy as np
import copy
class PolicyEvaluation(object):

    def __init__(self):

        self.grid_size = 16
        self.actions = ['up','down','left','right']
        self.action_prob = 0.25
        self.values = [0.]*self.grid_size

        '''
        T denote Terminal state
        S = {1,2,3,...,T}
        A = {up,down,left,right} (Equiprobable)
        R(s,s',a) = -1 for each transition

        |   T   |   1  |    2   |   3   |
        ---------------------------------
        |   4   |   5  |    6   |   7   |
        ---------------------------------
        |   8   |   9  |   10   |  11   |
        ---------------------------------
        |  12   |  13  |   14   |   T   |

        '''

    def get_pi_s_a(self,s):
        if s==0 or s==15:
            return 0.0
        else:
            return 0.25

    def get_transition_prob(self,s,s_dash,a):
        #up direction
        if s_dash-s==4 and a == self.actions[0]:
            return 1.0
        #down direction
        elif s-s_dash==4 and a == self.actions[1]:
            return 1.0
        #left direction
        elif s-s_dash==1 and a == self.actions[2]:
            if not (s==4 or s==8 or s==12):
                return 1.0
            else:
                return 0.0
        #right direction
        elif s_dash-s==1 and a == self.actions[3]:
            if not (s==3 or s==7 or s==11):
                return 1.0
            else:
                return 0.0
        #corner cases
        elif s-s_dash==0:
            if (s==0 or s==1 or s==2 or s==3) and a == self.actions[0]:
                return 1.0
            elif (s==12 or s==13 or s==14 or s==15) and a == self.actions[1]:
                return 1.0
            elif (s==0 or s==4 or s==8 or s==12) and a == self.actions[2]:
                return 1.0
            elif (s==3 or s==7 or s==11 or s==15) and a == self.actions[3]:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def get_reward(self,s,s_dash):
        if (s==0 and s_dash==0) or (s==15 and s_dash==15):
            return 0.0
        else:
            return -1.0

    def eval_policy(self):


        gamma = 1.0
        k = 0
        while True:
            delta = 0.0
            curr_values = [0.]*self.grid_size
            curr_values = copy.deepcopy(self.values)
            for s in xrange(len(self.values)):
                val = 0.0
                for a in self.actions:
                    for s_dash in xrange(len(self.values)):
                        pi_s_a = self.action_prob
                        val += self.get_pi_s_a(s)*self.get_transition_prob(s,s_dash,a)*(self.get_reward(s,s_dash)+ gamma*curr_values[s_dash])

                self.values[s]=val
                #print abs(curr_values[s]-val)
                delta = max(delta,abs(curr_values[s]-val))

            print "K: ", k
            print "Delta: ", delta
            print "Values: ", self.values

            k += 1
            if delta >= 10 or k > 10:
                break

    def improv_policy(self):
        policy_stable = True
        for s in xrange(len(self.values)):
            b = [0.25,0.25,0.25,0.25]

if __name__ == '__main__':
    pe = PolicyEvaluation()
    pe.eval_policy()
