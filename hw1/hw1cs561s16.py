# Author : ThammeGowda Narayanaswamy
# USC ID : 2074-6694-39
# Session: Spring 2016
# Course : USC CSCI 561 Foundations of Artificial Intelligence
# Topic  : Home work 1 : Squirrel Fight Game Strategy

import argparse
from pprint import pprint

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

            self.board = []                              # n x n board. each cell has [value, playerSign]

            # line 3 to 7 are board scores
            for i in range(self.boardSize):
                row = [[int(i), None] for i in lines[3 + i].strip().split()]
                self.board.append(row)

            # lines 8 to 12 are positions
            for i in range(self.boardSize):
                line = lines[8 + i].strip()
                for j in range(self.boardSize):
                    self.board[i][j][1] = line[j]


    def printCurrentState(self, debug=True):
        '''
        Prints current state of the board to console
        :return:
        '''
        out_format = lambda cell: '  %2d|%s' % (cell[0], cell[1])  if debug else cell[1]
        print('\n'.join([''.join([out_format(cell) for cell in row]) for row in self.board]))


    def getSimpleHeuristic(self):
        '''
        Computes a simple heuristic for the squirrel fight game.
        :return: an nxn array having expected gain value for each cell
        '''
        # nxn array with all zeros initially
        heuristic = [[0 for col in range(self.boardSize)]
                      for row in range(self.boardSize)]

        # possible moves
        possibleMoves = 0
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.board[i][j][1] == '*':                     # empty cell else: it's owned by somebody
                    heuristic[i][j] = self.board[i][j][0]         # Sneak, The value of the cell
                    # checking if this can be a raid
                    # left raid possible ?
                    if i > 0 and self.board[i-1][j][1] == self.oppPiece:
                        heuristic[i][j] += self.board[i-1][j][0]     # Strike left
                    # right raid possible?
                    if i < self.boardSize - 1 and self.board[i+1][j][1] == self.oppPiece:
                        heuristic[i][j] += self.board[i+1][j][0]     # Strike right

                    # up raid possible?
                    if j > 0 and self.board[i][j-1][1] == self.oppPiece:
                        heuristic[i][j] += self.board[i][j-1][0]     # Strike Up
                    # Down raid possible?
                    if j < self.boardSize - 1 and self.board[i][j+1][1] == self.oppPiece:
                        heuristic[i][j] += self.board[i][j+1][0]     # Strike Down
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
        if self.board[i][j][1] == self.emptyCell:

            # checking if this can be a raid
            # left raid possible ?
            raidScore = 0
            if i > 0 and self.board[i-1][j][1] == self.oppPiece:
                raidScore  += self.board[i-1][j][0]
            # right raid possible?
            if i < self.boardSize - 1 and self.board[i+1][j][1] == self.oppPiece:
                raidScore  += self.board[i+1][j][0]
            # up raid possible?
            if j > 0 and self.board[i][j-1][1] == self.oppPiece:
                raidScore  += self.board[i][j-1][0]
            # Down raid possible?
            if j < self.boardSize - 1 and self.board[i][j+1][1] == self.oppPiece:
                raidScore  += self.board[i][j+1][0]

            #raid score is a gain for me and loss for opponent
            myGain = raidScore
            oppLoss = raidScore

            # I own this cell now. No longer empty
            self.board[i][j][1] = self.myPiece
            myGain += self.board[i][j][0]

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
        if algorithm == 1:
            self.greedyBestFirstSearch()
        elif algorithm == 2:
            self.miniMax()
        elif algorithm == 3:
            self.alphaBetaPruning()
        else:
            raise Exception("Algorithm %d is unknown!" % algorithm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI-561 - HW 1 Solutions - by Thamme Gowda N.')
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i','--input', help='Input File', required=True)
    args = vars(parser.parse_args())
    problem = SquirrelProblem(args['input'])
    problem.nextMove(problem.nextMoveAlgorithm)
    print("Done")