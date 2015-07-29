__author__ = 'Thush'

import numpy as np
import random

class BridgeCrossing(object):

    ##Assume # of returns and requests per day are lambda values (coz mean of poisson is lambda)
    def __init__(self):

        self.width = 3
        self.length = 7
        self.actions = ['up','down','left','right']
        self.policy = [['right' for x in range(self.length)] for y in range(self.width)]
        self.values = [[0. for x in range(self.length)] for y in range(self.width)]
        self.Q_values = [[[0. for x in range(len(self.actions))] for y in range(self.length)] for z in range(self.width)]


    def get_transition_prob(self,s,s_dash,a):

        #edge cases
        if ((s[0]==0 and a==self.actions[0]) or
            (s[0]==self.width-1 and a==self.actions[1]) or
            (s[1]==0 and a==self.actions[2]) or
            (s[1]==self.length-1 and a==self.actions[3])):
            return 0.0

        else:
            if a==self.actions[0] and s_dash[0]==s[0]-1 and s_dash[1]==s[1]:
                return 1.0
            elif a==self.actions[1] and s_dash[0]==s[0]+1 and s_dash[1]==s[1]:
                return 1.0
            elif a==self.actions[2] and s_dash[0]==s[0] and s_dash[1]==s[1]-1:
                return 1.0
            elif a==self.actions[3] and s_dash[0]==s[0] and s_dash[1]==s[1]+1:
                return 1.0
            else:
                return 0.0

    def reward(self,s,s_dash):
        if s_dash[0]==0 and 1<=s_dash[1]<=5:
            return -50.0
        elif s_dash[0]==self.width-1 and 1<=s_dash[1]<=5:
            return -50.0
        elif s_dash[0]==1 and s_dash[1]==0:
            return 100.0
        elif s_dash[0]==1 and s_dash[1]==self.length-1:
            return 200.0
        else:
            return -1.0

    def get_next_state(self,s,a):
        if ((s[0]==0 and a==self.actions[0]) or
            (s[0]==self.width-1 and a==self.actions[1]) or
            (s[1]==0 and a==self.actions[2]) or
            (s[1]==self.length-1 and a==self.actions[3])):
            return s

        else:
            if a==self.actions[0]:
                return [s[0]-1,s[1]]
            elif a==self.actions[1]:
                return [s[0]+1,s[1]]
            elif a==self.actions[2]:
                return [s[0],s[1]-1]
            elif a==self.actions[3]:
                return [s[0],s[1]+1]
            else:
                return None

    def q_learning(self,alpha=0.5):

        gamma = 1.0
        eps_prob = 0.8
        for episode in xrange(500000):

            print "Episode", episode, "..."
            #select a random starting state
            i = random.randint(0,self.width-1)
            j = random.randint(0,self.length-1)
            s = [i,j]

            #if the random number is a goal state generate a new rand
            while ((s[0] == 0 and 1<=s[1]<=self.length-2) or
                           (s[0] == self.width-1 and 1<=s[1]<=self.length-2) or
                           (s[0]==1 and s[1]==0) or
                           (s[0]==1 and s[1]==self.length-1)):
                i = random.randint(0,self.width-1)
                j = random.randint(0,self.length-1)
                s = [i,j]

            s_dash = [-1,-1]

            #while not reached the end goal
            while not ((s_dash[0] == 0 and 1<=s_dash[1]<=self.length-2) or
                           (s_dash[0] == self.width-1 and 1<=s_dash[1]<=self.length-2) or
                           (s_dash[0]==1 and s_dash[1]==0) or
                           (s_dash[0]==1 and s_dash[1]==self.length-1)):

                #select an action
                randOrGreedy = np.random.binomial(0,eps_prob)
                if randOrGreedy == 0:
                    a = self.actions[random.randint(0,3)]
                else:
                    a = self.policy[i][j]

                #get next state
                s_dash = self.get_next_state(s,a)
                i2 = s_dash[0]
                j2 = s_dash[1]

                sample = self.reward(s,s_dash) + gamma * np.max(self.Q_values[i2][j2])

                idx = np.argmax(self.Q_values[i2][j2])
                a_idx = self.actions.index(a)
                self.Q_values[i][j][a_idx] = self.Q_values[i][j][a_idx] + alpha*(sample - self.Q_values[i][j][a_idx])

                s = s_dash
                max_idx = np.argmax(self.Q_values[i][j])
                self.policy[i][j] = self.actions[max_idx]

            alpha *= 0.999

            if eps_prob>0.01:
                eps_prob *= 0.99

            print "Policy: ", self.policy

if __name__ == '__main__':
    bc = BridgeCrossing()
    bc.q_learning(0.75)
