__author__ = 'Thushan Ganegedara'

__author__ = 'Thushan Ganegedara'


import numpy as np
import copy
class JacksCarRental(object):

    ##Assume # of returns and requests per day are lambda values (coz mean of poisson is lambda)
    def __init__(self):

        self.numDays = 10
        self.numStates = 2
        self.actions = [-5,-4,-3,-2,-1,0,1,2,3,4,5] # number of cars moved from 1 to 2 (positive) and 2 to 1 (negative)
        self.action_prob = 0.5
        self.values = [[0.]*self.numStates]*self.numDays
        self.policy = [0]*self.numDays

        self.states = [[0]*self.numStates]*self.numDays
        self.states[0]=[5,5]

        self.requests_1 = np.random.poisson(3,self.numDays)
        self.requests_2 = np.random.poisson(4,self.numDays)
        self.returns_1 = np.random.poisson(3,self.numDays)
        self.returns_2 = np.random.poisson(2,self.numDays)

    def get_pi_s_a(self,s):
        if s==0 or s==15:
            return 0.0
        else:
            return 0.25

    #transition happens at night (think after closing the shop)
    #so no returns or requests at the time
    def get_transition_prob(self,s,s_dash,a):
        #moving from 1 to 2
        if 5>=a>0:
            if s[0]-s_dash[0]==a and s_dash[1]-s[1]==a:
                return 1.0
            elif s_dash[0] > 20:
                return 0.0
            else:
                return 0.0

        #moving frmo 2 to 1
        elif -5<=a<0:
            if s[0]-s_dash[0]==a and s_dash[1]-s[1]==a:
                return 1.0
            elif s[0] < -20:
                return 0.0
            else:
                return 0.0

        #no moving
        else:
            return 1.0

    def get_reward(self,s,s_dash,a,rent_1,rent_2):
        return -2.*a + (rent_1 + rent_2) * 10.

    def eval_policy(self):

        print '\nEvaluating Policy...'
        gamma = 0.9


        k = 0
        while True:
            delta = [0.0]*self.numStates
            curr_values = copy.deepcopy(self.values)
            for s in xrange(self.numDays):
                rent_1 = np.min([self.requests_1[s],self.states[s][0]])
                rent_2 = np.min([self.requests_2[s],self.states[s][1]])
                val = [0.0]*self.numStates

                a = self.policy[s]
                for s_dash in xrange(self.numDays):
                    val[0] += self.get_transition_prob(s,s_dash,a)*(self.get_reward(s,s_dash,a,rent_1,rent_2)+ gamma*curr_values[s_dash][0])
                    val[1] += self.get_transition_prob(s,s_dash,a)*(self.get_reward(s,s_dash,a,rent_1,rent_2)+ gamma*curr_values[s_dash][1])

                self.values[s]=val
                #print abs(curr_values[s]-val)
                for i in xrange(self.numStates):
                    delta[i] = max(delta[i],abs(curr_values[s][i]-val[i]))

            print '\n'
            print "K: ", k
            print "Delta: ", delta
            print "Values: ", self.values
            print "Policy: ", self.policy
            k += 1
            if delta <= 1e-5 or k > 10:
                break

        self.improv_policy(d,rent_1,rent_2)

        self.states[d+1][0]=self.states[d][0]+self.returns_1[d]-rent_1-self.actions[d]
        self.states[d+1][0]=self.states[d][1]+self.returns_2[d]-rent_2+self.actions[d]


    def improv_policy(self,day,rent_1,rent_2):

        print '\nImproving Policy...'
        gamma = 0.9

        policy_stable = True
        for s in xrange(self.numStates):
            b = self.policy[s]
            pi_s_arr = []
            for a in self.actions:
                val = [0.,0.]
                for s_dash in xrange(self.numStates):
                    val[0] += self.get_transition_prob(s,s_dash,a,day)*(self.get_reward(s,s_dash,a,rent_1,rent_2)+gamma * self.values[day][s_dash])
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

