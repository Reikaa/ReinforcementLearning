__author__ = 'Thushan Ganegedara'

__author__ = 'Thushan Ganegedara'


import numpy as np
import copy
from scipy.stats import poisson
class JacksCarRental(object):

    ##Assume # of returns and requests per day are lambda values (coz mean of poisson is lambda)
    def __init__(self):

        self.max_cars = 10
        self.max_moves = 5
        self.numStates = 2
        self.actions = [-5,-4,-3,-2,-1,0,1,2,3,4,5] # number of cars moved from 1 to 2 (positive) and 2 to 1 (negative)

        self.policy = {}
        for i in xrange(self.max_cars):
            for j in xrange(self.max_cars):
                key = str(i)+","+str(j)
                value = self.actions
                self.policy[key] = value

        self.values = {}
        for i in xrange(self.max_cars):
            for j in xrange(self.max_cars):
                key = str(i)+","+str(j)
                value = 0
                self.values[key] =  value

        self.req_1_lam = 3
        self.req_2_lam = 4
        self.ret_1_lam = 3
        self.ret_2_lam = 2

    def get_pois_prob(self,lam,n):
        denom = 1.
        for i in xrange(1,n+1):
            denom  = denom * i

        return (lam**n/denom)*np.exp(-n)

    #transition happens at night (think after closing the shop)
    #so no returns or requests at the time
    def get_transition_prob(self,s_1,s_dash_1,req_1,ret_1,s_2,s_dash_2,req_2,ret_2,a):
        req_prob_1 = self.get_pois_prob(self.req_1_lam,req_1)
        req_prob_2 = self.get_pois_prob(self.req_2_lam,req_2)
        ret_prob_1 = self.get_pois_prob(self.ret_1_lam,ret_1)
        ret_prob_2 = self.get_pois_prob(self.ret_2_lam,ret_2)

        return req_prob_1*req_prob_2*ret_prob_1*ret_prob_2


    def get_reward(self,rent_1,rent_2,a):
        return -2.*a + (rent_1 + rent_2) * 10.

    def get_req_and_ret(self, s, s_dash_b4_a):

        all_combs = []
        tmp1 = s_dash_b4_a - s
        return [[x,y] for x in xrange(s+1) for y in xrange(self.max_cars) if y-x == tmp1]

    def get_str_key(self,s1,s2):
        return str(s1)+","+str(s2)

    def eval_policy(self):

        print '\nEvaluating Policy...'
        gamma = 0.9


        k = 0
        while True:
            delta = 0.0
            curr_values = copy.deepcopy(self.values)

            #s_1 and s_2 are the available number of cars
            for s_1 in xrange(self.max_cars):
                for s_2 in xrange(self.max_cars):

                    for s_dash_1 in xrange(self.max_cars):
                        for s_dash_2 in xrange(self.max_cars):

                            actions = self.policy.get(self.get_str_key(s_1,s_2))

                            for a in actions:
                                s_dash_1_b4_a = s_dash_1 + int(a)
                                s_dash_2_b4_a = s_dash_2 - int(a)

                                req_and_ret_1 = self.get_req_and_ret(s_1,s_dash_1_b4_a)
                                req_and_ret_2 = self.get_req_and_ret(s_2,s_dash_2_b4_a)
                                val = 0.

                                for rr1 in req_and_ret_1:
                                    for rr2 in req_and_ret_2:
                                        val += self.get_transition_prob(s_1,s_dash_1,rr1[0],rr1[1],s_2,s_dash_2,rr2[0],rr2[1],a)*\
                                               (self.get_reward(rr1[0],rr2[0],a)+ gamma*curr_values[self.get_str_key(s_1,s_2)])

                    self.values[self.get_str_key(s_1,s_2)]=val

                #print abs(curr_values[s]-val)
                    for i in xrange(self.numStates):
                        delta = max(delta,abs(curr_values[self.get_str_key(s_1,s_2)]-val))

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
        gamma = 0.9

        policy_stable = True

        #s_1 and s_2 are the available number of cars
        for s_1 in xrange(self.max_cars):
            for s_2 in xrange(self.max_cars):

                b = self.policy.get(self.get_str_key(s_1,s_2))
                pi_s_arr = []

                for a in self.actions:

                    for s_dash_1 in xrange(self.max_cars):
                        for s_dash_2 in xrange(self.max_cars):

                            s_dash_1_b4_a = s_dash_1 + a
                            s_dash_2_b4_a = s_dash_2 - a

                            req_and_ret_1 = self.get_req_and_ret(s_1,s_dash_1_b4_a)
                            req_and_ret_2 = self.get_req_and_ret(s_2,s_dash_2_b4_a)
                            val = 0.

                            for rr1 in req_and_ret_1:
                                for rr2 in req_and_ret_2:
                                    val += self.get_transition_prob(s_1,s_dash_1,rr1[0],rr1[1],s_2,s_dash_2,rr2[0],rr2[1],a)*\
                                           (self.get_reward(rr1[0],rr2[0],a)+ gamma*self.values[self.get_str_key(s_dash_1,s_dash_2)])


                    pi_s_arr.append(val)

                max = np.max(pi_s_arr)
                max_idx = [i for i, j in enumerate(pi_s_arr) if j == max]
                actions = []
                for idx in max_idx:
                    actions.append(self.actions[idx])

                self.policy[self.get_str_key(s_1,s_2)] = actions

                if not b==self.policy[self.get_str_key(s_1,s_2)]:
                    policy_stable=False

        if policy_stable:
            return
        else:
            self.eval_policy()


if __name__ == '__main__':
    pe = JacksCarRental()
    pe.eval_policy()

