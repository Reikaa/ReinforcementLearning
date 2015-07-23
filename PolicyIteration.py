__author__ = 'Thushan Ganegedara'


import numpy as np
import copy
class PolicyIteration(object):

    def __init__(self):

        self.grid_size = 16
        self.actions = ['up','down','left','right']
        self.action_prob = 0.25
        self.values = [0.]*self.grid_size
        self.policy = ['up,down,left,right']*self.grid_size
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
        #down direction
        if s_dash-s==4 and a == self.actions[1]:
            return 1.0
        #up direction
        elif s-s_dash==4 and a == self.actions[0]:
            return 1.0
        #left direction
        elif s-s_dash==1 and a == self.actions[2]:
            if not (s==0 or s==4 or s==8 or s==12):
                return 1.0
            else:
                return 0.0
        #right direction
        elif s_dash-s==1 and a == self.actions[3]:
            if not (s==3 or s==7 or s==11 or s==15):
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
        elif s_dash==0 or s_dash==15:
            return 0.0
        else:
            return -1.0

    def eval_policy(self):

        print '\nEvaluating Policy...'
        gamma = 1.0
        k = 0
        while True:
            delta = 0.0
            curr_values = copy.deepcopy(self.values)
            for s in xrange(len(self.values)):
                val = 0.0
                action_str = ''
                if ',' not in self.policy[s]:
                    action_str = self.policy[s]
                    for s_dash in xrange(len(self.values)):
                        val += self.get_transition_prob(s,s_dash,action_str)*(self.get_reward(s,s_dash)+ gamma*curr_values[s_dash])
                else:
                    actions = self.policy[s].split(',')
                    for a in actions:
                        for s_dash in xrange(len(self.values)):
                            val += self.get_transition_prob(s,s_dash,a)*(self.get_reward(s,s_dash)+ gamma*curr_values[s_dash])
                    val = val/len(actions)

                self.values[s]=val
                #print abs(curr_values[s]-val)
                delta = max(delta,abs(curr_values[s]-val))
            print '\n'
            print "K: ", k
            print "Delta: ", delta
            print "Values: ", self.values
            print "Policy: ", self.policy
            k += 1
            if delta <= 1e-5 or k > 10:
                break

        self.improv_policy()


    def improv_policy(self):

        print '\nImproving Policy...'
        gamma = 1.0
        policy_stable = True
        for s in xrange(len(self.values)):
            b = self.policy[s]
            pi_s_arr = []
            for a in self.actions:
                val = 0.
                for s_dash in xrange(len(self.values)):
                    val += self.get_transition_prob(s,s_dash,a)*(self.get_reward(s,s_dash)+gamma * self.values[s_dash])
                pi_s_arr.append(val)

            print pi_s_arr
            max = np.max(pi_s_arr)
            max_idx = [i for i, j in enumerate(pi_s_arr) if j == max]
            str_max_actions = ''
            for i in max_idx:
                str_max_actions += self.actions[i]+","

            #str_max_actions = self.actions[np.argmax(pi_s_arr)]
            self.policy[s] = str_max_actions

            if not b==self.policy[s]:
                policy_stable=False

        if policy_stable:
            return
        else:
            self.eval_policy()

if __name__ == '__main__':
    pe = PolicyIteration()
    pe.eval_policy()

