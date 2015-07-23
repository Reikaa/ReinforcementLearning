__author__ = 'Thushan Ganegedara'

__author__ = 'Thushan Ganegedara'


import numpy as np
import copy
class JacksCarRental(object):

    ##Assume # of returns and requests per day are lambda values (coz mean of poisson is lambda)
    def __init__(self):

        self.numStates = 2
        self.actions = [0] # number of cars moved from 1 to 2 (positive) and 2 to 1 (negative)
        self.action_prob = 0.5
        self.values = [0.]*self.numStates
        self.policy = [0]*self.numStates

    def get_pi_s_a(self,s):
        if s==0 or s==15:
            return 0.0
        else:
            return 0.25

    def get_transition_prob(self,s,s_dash,a):
        #moving from 1 to 2
        if a>0:
            if s[0]-s_dash[0]==a and s_dash[1]-s[1]==a:
                return 1.0
            else:
                return 0.0
        #moving frmo 2 to 1
        elif a<0:
            if s[0]-s_dash[0]==a and s_dash[1]-s[1]==a:
                return 1.0
            else:
                return 0.0
        #no moving
        else:
            if -5 <= a <= 5:
                return 1.0
            else:
                return 0.0

    def get_reward(self,s,s_dash,a):
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
    pe = JacksCarRental()
    pe.eval_policy()

