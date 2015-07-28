__author__ = 'Thushan Ganegedara'

__author__ = 'Thushan Ganegedara'


import numpy as np
import copy
from scipy.stats import poisson
class JacksCarRental(object):

    ##Assume # of returns and requests per day are lambda values (coz mean of poisson is lambda)
    def __init__(self):

        self.max_cars = 20
        self.max_moves = 5
        self.states_1 = np.zeros((self.max_cars,self.max_cars),dtype=np.float32)
        self.states_2 = np.zeros((self.max_cars,self.max_cars),dtype=np.float32)
        self.actions = [-5,-4,-3,-2,-1,0,1,2,3,4,5] # number of cars moved from 1 to 2 (positive) and 2 to 1 (negative)
        self.values = np.zeros((self.max_cars,self.max_cars),dtype=np.float32)
        self.policy = np.zeros((self.max_cars,self.max_cars),dtype=np.float32)

        self.req_1_lam = 3
        self.req_2_lam = 4
        self.ret_1_lam = 3
        self.ret_2_lam = 2

    def get_pois_prob(self,lam,n):
        denom = 1.
        for i in xrange(1,n+1):
            denom  = denom * i

        return (lam^n/denom)*np.exp(-n)

    #transition happens at night (think after closing the shop)
    #so no returns or requests at the time
    def get_transition_prob(self,s_1,s_dash_1,s_2,s_dash_2,a):
        #moving from 1 to 2
        if 5>=a>0:
            cars_1_before_move = s_dash_1 + a
            cars_2_before_move = s_dash_2 - a

            #more returns less requests
            if s_dash_1>s_1:
                min_ret_1 = s_dash_1-s_1
                prob = 0.
                for ret_1 in xrange(min_ret_1,self.max_cars):
                    req_1 = ret_1 - s_dash_1
                    p_val_req = self.get_pois_prob(self.req_1_lam,req_1)
                    p_val_ret = self.get_pois_prob(self.ret_1_lam,ret_1)
                    prob += p_val_req*p_val_ret

                #return prob
            #less returns more requests
            elif s_dash_1<s_1:
                min_req_1 = s_1-s_dash_1
                prob = 0.
                for req_1 in xrange(min_req_1,self.max_cars):
                    ret_1 = req_1 - s_dash_1
                    p_val_req = self.get_pois_prob(self.req_1_lam,req_1)
                    p_val_ret = self.get_pois_prob(self.ret_1_lam,ret_1)
                    prob += p_val_req*p_val_ret

                #return prob
            else:


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

        self.improv_policy(rent_1,rent_2)

    def improv_policy(self,rent_1,rent_2):

        print '\nImproving Policy...'
        gamma = 0.9

        policy_stable = True
        for s in xrange(self.numDays):
            b = self.policy[s]
            pi_s_arr = []
            for a in self.actions:
                val = [0.,0.]
                for s_dash in xrange(self.numDays):
                    val[0] += self.get_transition_prob(s,s_dash,a,day)*(self.get_reward(s,s_dash,a,rent_1,rent_2)+gamma * self.values[day][s_dash])
                pi_s_arr.append(val)

            print pi_s_arr
            amax = np.argmax(pi_s_arr)

            self.policy[s] = self.actions[amax]

            self.states[s+1][0]=self.states[s][0]+self.returns_1[s]-rent_1[s]-self.actions[s]
            self.states[s+1][0]=self.states[s][1]+self.returns_2[s]-rent_2[s]+self.actions[s]
            if not b==self.policy[s]:
                policy_stable=False

        if policy_stable:
            return
        else:
            self.eval_policy()

if __name__ == '__main__':
    pe = JacksCarRental()
    pe.eval_policy()

