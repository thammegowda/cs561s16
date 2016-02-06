# Author : ThammeGowda Narayanaswamy
# USC ID : 2074-6694-39
# Session: Spring 2016
# Course : USC CSCI 561 Foundations of Artificial Intelligence
# Topic  : Home work 1 : Squirrel Fight Game Strategy

import argparse
from pprint import pprint


NEXT_STATE_FILE = "next_state.txt"
LOG_FILE = "traverse_log.txt"

class SquirrelProblem(object):
    '''
    Squirrel Problem stated by Home work 1 of USC CSCI 561 - Spring 2016
    '''

    def __init__(self, prob_file):
        self.boardSize = 5                               # Fixed, as per description
        self.emptyCell = '*'                             # Fixed, as per description
        with open(prob_file) as f:
            lines = f.readlines()
            self.nextMoveAlgorithm = int(lines[0].strip())
            self.myPiece = lines[1].strip()
            self.oppPiece = 'O' if self.myPiece == 'X' else 'X'

            # line 3 to 7 are board scores
            # n x n board. each cell has value
            self.costs = [[int(j) for j in lines[3 + i].strip().split()]
                          for i in range(self.boardSize)]

            # lines 8 to 12 are positions
            # # n x n board. each cell has playerSign
            self.state = [[j for j in lines[8 + i].strip()]
                          for i in range(self.boardSize)]


    def printState(self, state, debug=True, fileName=None):
        '''
        Prints current state of the board to console
        :return:
        '''
        out_format = lambda i, j: '  %2d|%s' % (self.costs[i][j], state[i][j])\
            if debug else self.state[i][j]
        res = ""
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                res += out_format(i, j)
            res += "\n"
        if fileName:
            with open(fileName, 'w') as w:
                w.write(res)
        else:
            print(res)


    def getSimpleHeuristic(self):
        '''
        Computes a simple heuristic for the squirrel fight game.
        :return: an nxn array having expected gain value for each cell
        '''
        # nxn array with all zeros initially
        heuristic = [[0 for col in range(self.boardSize)]
                      for row in range(self.boardSize)]

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.state[i][j] == '*':                     # empty cell else: it's owned by somebody
                    heuristic[i][j] = self.costs[i][j]         # Sneak, The value of the cell
                    # checking if this can be a raid
                    # left raid possible ?
                    if i > 0 and self.state[i-1][j] == self.oppPiece:
                        heuristic[i][j] += self.costs[i-1][j]     # Strike left
                    # right raid possible?
                    if i < self.boardSize - 1 and self.state[i+1][j] == self.oppPiece:
                        heuristic[i][j] += self.costs[i+1][j]     # Strike right

                    # up raid possible?
                    if j > 0 and self.state[i][j-1] == self.oppPiece:
                        heuristic[i][j] += self.costs[i][j-1]     # Strike Up
                    # Down raid possible?
                    if j < self.boardSize - 1 and self.state[i][j+1] == self.oppPiece:
                        heuristic[i][j] += self.costs[i][j+1]     # Strike Down
        return heuristic

    def takeOver(self, row, col):
        '''
        Takes over a specified cell
        :param row: the row number
        :param col: column number
        :return: (myGain, opponentLoss)
        '''
        i, j = row, col
        myGain, oppLoss = 0, 0
        if self.state[i][j] == self.emptyCell:

            # checking if this can be a raid
            # left raid possible ?
            raidScore = 0
            if i > 0 and self.state[i-1][j] == self.oppPiece:
                raidScore  += self.costs[i-1][j]
            # right raid possible?
            if i < self.boardSize - 1 and self.state[i+1][j] == self.oppPiece:
                raidScore  += self.costs[i+1][j]
            # up raid possible?
            if j > 0 and self.state[i][j-1] == self.oppPiece:
                raidScore  += self.costs[i][j-1]
            # Down raid possible?
            if j < self.boardSize - 1 and self.state[i][j+1] == self.oppPiece:
                raidScore  += self.costs[i][j+1]

            #raid score is a gain for me and loss for opponent
            myGain = raidScore
            oppLoss = raidScore

            # I own this cell now. No longer empty
            self.state[i][j] = self.myPiece
            myGain += self.costs[i][j]

        else:
            raise Exception("I don't break Game Rules! The cell is not empty")
        return (myGain, oppLoss)

    def greedyBestFirstSearch(self):
        '''
        Performs next move by using Greedy best first search strategy
        :return:
        '''
        heuristic = self.getSimpleHeuristic()
        maxVal = 0
        pos = None
        # Find greedy best first position
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if heuristic[i][j] > maxVal:
                    maxVal = heuristic[i][j]
                    pos = (i, j)
        if maxVal > 0:
            self.takeOver(pos[0], pos[1])
        #else: no available slot, die


    def miniMax(self):
        print("Mini Max - Not implemented")

    def alphaBetaPruning(self):
        print("Alpha - Beta Not implemented")

    def nextMove(self, algorithm):
        '''
        Makes the next move as per the algorithm
        :param algorithm: the strategy for next move
        :return:
        '''
        if algorithm == 1:
            self.greedyBestFirstSearch()
        elif algorithm == 2:
            self.miniMax()
        elif algorithm == 3:
            self.alphaBetaPruning()
        else:
            raise Exception("Algorithm %d is unknown!" % algorithm)

    def readStateFile(self, fileName):
        with open(fileName) as f:
            return [[j for j in next(f).strip()] for _ in range(self.boardSize)]

    def areStatesSame(self, state1, state2):
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if state1[i][j] != state2[i][j]:
                    return False
        return True


    def printResult(self, debug=False):
        '''
        prints the result to files as per problem description
        :param debug:
        :return:
        '''
        self.printState(debug=debug, state=self.state, fileName=NEXT_STATE_FILE)
        logAvailable = self.nextMoveAlgorithm != 1

        if logAvailable:
            #FIXME: print log
            print("Log printing not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI-561 - HW 1 Solutions - by Thamme Gowda N.')
    parser.add_argument('-i','--input', help='Input File', required=True)
    parser.add_argument('-t','--test', help='Test/Terminal File', required=False)
    args = vars(parser.parse_args())
    problem = SquirrelProblem(args['input'])
    problem.nextMove(problem.nextMoveAlgorithm)
    if 'test' in args:
        terminalState = problem.readStateFile(args['test'])
        res = problem.areStatesSame(problem.state, terminalState)
        print("Test Passed?: %s" % res)
        if not res:
            print("Error:\n Expected :\n")
            problem.printState(terminalState)
            print("But actual :\n")
            problem.printState(problem.state)

    problem.printResult()