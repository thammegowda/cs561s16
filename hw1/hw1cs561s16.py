# -*- coding: utf-8 -*-
# Author : ThammeGowda Narayanaswamy
# USC ID : 2074-6694-39
# Session: Spring 2016
# Course : USC CSCI 561 Foundations of Artificial Intelligence
# Topic  : Home work 1 : Squirrel Fight Game Strategy

import argparse
from copy import deepcopy
from pprint import pprint
import sys
from os import path
from decimal import Decimal
import string

NEXT_STATE_FILE = "next_state.txt"
LOG_FILE = "traverse_log.txt"
MAX_INT = Decimal("inf")
MIN_INT = Decimal("-inf")

COLNAMES = [i for i in string.uppercase[0:5]]
ROWNAMES = [str(i) for i in range(1, 6)]


def tracklogformat(row, col, depth, val):
    node = "root" if row == col == None else "%s%s" % (COLNAMES[col], ROWNAMES[row])
    depth = str(depth)
    val = str(val)
    return "%s,%s,%s" % (node, depth, val)

def tracklogNode(node):
    name = "root" if node.depth == 0 else "%s%s" % (COLNAMES[node.pos[1]], ROWNAMES[node.pos[0]])
    return "%s,%s,%s" % (name, node.depth, node.score)


class SquirrelProblem(object):
    '''
    Squirrel Problem stated by Home work 1 of USC CSCI 561 - Spring 2016
    '''
    def __init__(self, prob_file):

        self.boardSize = 5                               # Fixed, as per description
        self.emptyCell = '*'                             # Fixed, as per description
        self.opponent = lambda piece: 'O' if piece == 'X' else 'X'

        with open(prob_file) as f:
            lines = f.readlines()
            self.nextMoveAlgorithm = int(lines[0].strip())
            self.myPiece = lines[1].strip()
            self.oppPiece = self.opponent(self.myPiece)
            self.cutoff = int(lines[2].strip())

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

    def evaluateState(self, state):
        '''
        Evaluates the score of game at any given state.
         The game score = my score - opponent score
        :param state: state of game
        :return: game score which is an integer
        '''
        score = 0
        for (i,j) in self.yieldAllCells():
            if state[i][j] == self.myPiece:
                    score += self.costs[i][j]
            elif state[i][j] == self.oppPiece:
                score -= self.costs[i][j]
            #else it is a free node
        return score

    def determineNextStateHeuristic(self, state, playerPiece):
        """
        Determines heuristics for the next possible state of given player from any given state
        :param state: the current state
        :param playerPiece: the player who has a move
        :return: heuristic
        """

        oppPiece = self.opponent(playerPiece)
        heuristic = [[None for j in range(self.boardSize)] for i in range(self.boardSize)]
        # the heuristic is computed without actually making the move
        # this is done using delta with current score
        currentScore = self.evaluateState(state)
        for i, j in self.yieldAllCells():
            if state[i][j] == self.emptyCell:  # empty cell else: it's owned by somebody
                  # Sneak, The value of the cell; game score goes up
                heuristic[i][j] = currentScore + self.costs[i][j]
                # checking if this can also be a raid

                # left raid possible ?
                oppLoss = 0
                if i > 0 and self.state[i-1][j] == oppPiece:
                    oppLoss =  self.costs[i-1][j]
                # right raid possible?
                if i < self.boardSize - 1 and state[i+1][j] == oppPiece:
                    oppLoss += self.costs[i+1][j]

                # up raid possible?
                if j > 0 and state[i][j-1] == oppPiece:
                    oppLoss += self.costs[i][j-1]     # Strike Up
                # Down raid possible?
                if j < self.boardSize - 1 and state[i][j+1] == oppPiece:
                    oppLoss += self.costs[i][j+1]     # Strike Down

                # for all the raids, the new score goes up by 2 times the raided cell
                # its 2 times because the foe loses it and we gain it. Thus difference is large by 2 times
                heuristic[i][j] += 2 * oppLoss
        return heuristic

    def yieldAllCells(self):
        """
        Yields indices for all possibles cells in the board
        :return: (row, column) indices from top left to bottom right
        """
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                yield (i, j)

    def findEmptyCells(self, state):
        res = []
        for i, j in self.yieldAllCells():
            if state[i][j] == self.emptyCell:
                res.append((i, j))
        return res


    def isTerminalState(self, state):
        """
        checks if the game state is terminal
        :param state: game state
        :return: boolean True if the game is complete, False if further moves are possible
        """
        #state is complete if all cells are occupied
        for i, j in self.yieldAllCells():
            if state[i][j] == self.emptyCell:
                return False
        return True

    def moveToCell(self, state, row, col, playerPiece):
        '''
        Takes over a specified cell
        :param row: the row number
        :param col: column number
        :return: list of triples (row, col, piece);
         this can be used for reverting the state by undoing the moves
        '''
        oppPiece = self.opponent(playerPiece)
        undoLog = []
        if state[row][col] == self.emptyCell:

            # player owns this cell now. so it is no longer empty
            self.state[row][col] = playerPiece
            undoLog.append((row, col, self.emptyCell))

            # checking if this can be a raid
            adjacentCells = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            raid = False
            oppCells = []
            for i, j in adjacentCells:
                if 0 < i and i < (self.boardSize - 1) and  0 < j and j < (self.boardSize - 1):
                    if state[i][j] == oppPiece:
                        oppCells.append((i, j))
                    elif state[i][j] == playerPiece:
                        raid = True
            if raid:
                for x, y in oppCells:
                    state[x][y] = playerPiece
                    undoLog.append((x, y, oppPiece))
        else:
            raise Exception("I don't break Game Rules! The cell is not empty")
        return undoLog


    def applyMoves(self, state, moveLog):
        """
        Applies a sequence of moves on a state
        :param state: the initial state
        :param moveLog: sequence of moves (row, col, player)
        :return:
        """
        for action in moveLog:
            state[action[0]][action[1]] = action[2]


    def greedyBestFirstSearch(self):
        '''
        Performs next move by using Greedy best first search strategy
        :return:
        '''
        heuristic = self.determineNextStateHeuristic(self.state, self.myPiece)
        maxVal = MIN_INT
        pos = None
        # Find greedy best first position
        for i, j in self.yieldAllCells():
            if heuristic[i][j] != None and heuristic[i][j] > maxVal:
                maxVal = heuristic[i][j]
                pos = (i, j)
        if maxVal > 0:
            return self.moveToCell(self.state, pos[0], pos[1], self.myPiece)
        #else: no available slot, die


    def miniMax(self):
        solver = MiniMaxSolver(self)
        root = solver.solve(self.state)
        #root.prettyPrint()
        #paths = root.leafPaths()
        #for path in paths:
        #    print(path)

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

    def readStateFile(self, fileName, n):
        """
        Reads game state from a file
        :param fileName: path to file
        :param n: the matrix/board size
        :return: nxn matrix having game state
        """
        with open(fileName) as f:
            return [[j for j in next(f).strip()] for _ in range(n)]

    def areStatesSame(self, state1, state2):
        """
        Returns True if give two states are same
        :param state1: first state
        :param state2: second state
        :return: True if states are same; false otherwise
        """
        for i, j in self.yieldAllCells():
            if state1[i][j] != state2[i][j]:
                return False
        return True

    def saveResult(self, debug=False):
        '''
        Saves the result to files as per problem description
        :param debug:
        :return:
        '''
        self.printState(debug=debug, state=self.state, fileName=NEXT_STATE_FILE)
        logAvailable = self.nextMoveAlgorithm != 1

        if logAvailable:
            #FIXME: print log
            print("Log printing not implemented")


class Node(object):

    def __init__(self, score, pos, piece, depth=0, parent=None):
        self.parent = parent
        self.children = None
        self.score = score
        self.pos = pos
        self.piece = piece
        self.depth = depth

    def add_child(self, node):
        node.parent = self
        node.depth = self.depth + 1
        if self.children == None:
            self.children = []
        self.children.append(node)


    def prettyPrint(self, prefix="", isTail=True):
        name = "%3s %s" % (self.score, self.pos)
        print(prefix + ("└── " if isTail else "├── ") + name)
        if (self.children):
            formatstring = "    " if isTail else  "│   "
            for i in range(0, len(self.children) - 1):
                self.children[i].__prettyPrint(prefix + formatstring, False)
            self.children[-1].__prettyPrint(prefix + formatstring, True)



    def leafPaths(self, prefix=None):
        res = tracklogformat(self.pos[0], self.pos[1], self.depth, self.score)
        if prefix:
            res = prefix + "\n" + res
        if self.children: # has children, recurse
            for c in self.children:
                for i in c.leafPaths(res):
                    yield i
        else:
            yield res

class MiniMaxSolver(object):

    def __init__(self, problem):
        self.problem = problem
        self.evaluateState = problem.evaluateState
        self.maxPlayer = problem.myPiece
        self.minPlayer = problem.oppPiece
        self.tracklog = []
        self.maxdepth = problem.cutoff

    def solve(self, state):
        # first turn is my players'. My player is maxPlayer
        print("Node,Depth,Value")
        root = Node(MIN_INT, (None, None), None)
        self.maximum(state, root)
        return root

    def maximum(self, state, parent):
        cells = self.problem.findEmptyCells(state)
        if parent.depth == self.maxdepth or not cells:
            parent.score = self.problem.evaluateState(state)
            print(tracklogNode(parent))
        else:
            print(tracklogNode(parent))
            for x, (i, j) in enumerate(cells):
                if i == 2 and j == 4:
                    # error case
                    pass
                undoMoves = self.problem.moveToCell(state, i, j, self.maxPlayer)    # max's move
                child = Node(MAX_INT, (i, j), self.maxPlayer)
                parent.add_child(child)
                child.score = self.minimum(state, child)                     # turn goes to min player
                if child.score > parent.score:
                    parent.score = child.score
                    parent.nextMove = child
                if x < len(cells) -1: # for all except the last one
                    print(tracklogNode(parent))
                self.problem.applyMoves(state, undoMoves)
            print(tracklogNode(parent))
        return parent.score

    def minimum(self, state, parent):
        cells = self.problem.findEmptyCells(state)
        if parent.depth == self.maxdepth or not cells:
            parent.score = self.problem.evaluateState(state)
            print(tracklogNode(parent))
        else:
            print(tracklogNode(parent))
            for x, (i, j) in enumerate(cells):
                undoMoves = self.problem.moveToCell(state, i, j, self.minPlayer)     # min's move
                child = Node(MIN_INT, (i, j), self.minPlayer)
                parent.add_child(child)
                self.maximum(state, child)                             # turn goes to max, depth reduced by 1
                if child.score < parent.score:
                    parent.score = child.score
                    parent.nextMove = child
                if x < len(cells) -1: # for all except the last one
                    print(tracklogNode(parent))
                self.problem.applyMoves(state, undoMoves)
            print(tracklogNode(parent))
        return parent.score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI-561 - HW 1 Solutions - by Thamme Gowda N.')
    parser.add_argument('-i','--input', help='Input File', required=True)
    parser.add_argument('-t','--test', action="store_true", help='Auto detect tests in directory')
    parser.add_argument('-tf','--testfile', required = False, help='Use this test file')
    args = vars(parser.parse_args())

    problem = SquirrelProblem(args['input'])
    problem.nextMove(problem.nextMoveAlgorithm)

    testfile = None
    if args['test']: # test was requested
        tmp = path.join(path.dirname(path.abspath(args['input'])), NEXT_STATE_FILE)
        if path.exists(tmp): # see if there is a test file
            testfile = tmp
    if 'testfile' in args and args['testfile']:
        testfile = args['testfile']
    if testfile:
        terminalState = problem.readStateFile(testfile, problem.boardSize)
        res = problem.areStatesSame(problem.state, terminalState)
        print("Test Passed?: %s" % res)
        if not res:
            print("Error:\n Expected state:\n")
            problem.printState(terminalState)
            print("But actual state:\n")
            problem.printState(problem.state)
