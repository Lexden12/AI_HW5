import math

from AIPlayerUtils import *
from GameState import *
from Move import Move
from Ant import UNIT_STATS
from Construction import CONSTR_STATS
from Constants import *
from Player import *
import random
import sys
sys.path.append("..")  # so other modules can be found in parent dir

import numpy as np

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "NeuralNet")
        self.nn = NeuralNetwork(10, 1, [20, 7], 0.1)
        self.nn.load('../thoma20_schendel21_nn.npy')
        self.useNN = False
        self.eval = {}
        self.moveCount = 0

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 7)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        self.moveCount += 1
        if self.moveCount > 200:
          return None
        buildCache(currentState)
        frontierNodes = []
        expandedNodes = []
        if not self.useNN:
          steps = self.heuristicStepsToGoal(currentState)
        else:
          steps = self.nn.evaluate(utilityComponents(currentState, currentState.whoseTurn))[0]
        self.eval[currentState] = steps
        root = Node(None, currentState, 0, 0, None)
        frontierNodes.append(root)

        # For loops goes until set depth
        # in this case a depth of 3 
        for x in range(2):
          frontierNodes.sort(key=self.sortAttr)
          leastNode = root
          leastNode = self.bestMove(frontierNodes)
          frontierNodes.remove(leastNode)
          if len(frontierNodes) >= 50:
            frontierNodes = frontierNodes[:50]
          frontierNodes.extend(self.expandNode(leastNode))
          root = frontierNodes[0]

        # after finding best path return to
        # first node of path to send that move
        while not leastNode.depth == 1:
          leastNode = leastNode.parent
        
        return leastNode.move

    ##
    #bestMove
    #Description: Gets the best move from the list of possible moves
    #
    #Parameters:
    #   nodes - List of nodes which contain the possible moves from this location and their rank
    #           Used to find the best move
    #
    #Return: Best node from the evalutaion in each node
    ##
    def bestMove(self, nodes):
      return nodes[0]

    ##
    # expandNode
    #
    # takes in a node and expands it by
    # taking all valid moves from that state
    # and creating new nodes for each new move
    #
    # returns a list of nodes
    ##
    def expandNode(self, node):
      moves = listAllLegalMoves(node.state)
      nodes = []
      for move in moves:
        nextState = getNextState(node.state, move)
        if not self.useNN:
          steps = self.heuristicStepsToGoal(nextState)
        else:
          steps = self.nn.evaluate(utilityComponents(nextState, nextState.whoseTurn))[0]
        newDepth = node.depth + 1
        newNode = Node(move, nextState, newDepth, steps, node)
        nodes.append(newNode)
      return nodes

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##

    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        print("Move Count: {}".format(self.moveCount))
        self.moveCount = 0
        if self.useNN:
          return
        training, validation = self.getTrainingData()
        for state, target in training:
          input = utilityComponents(state, state.whoseTurn)
          self.nn.train(input, [target])
        self.nn.save('./thoma20_schendel21_nn.npy')

        err = []
        for state, target in validation:
          input = utilityComponents(state, state.whoseTurn)
          err.append(abs(target - self.nn.evaluate(input)[0]))
        print("Min Error: {}".format(min(err)))
        print("Avg Error: {}".format(sum(err)/len(err)))
        print("Max Error: {}\n".format(max(err)))

    def getTrainingData(self):
        all_data = random.sample(self.eval.items(), k=len(self.eval.items()))
        training = all_data[:int(len(self.eval.items()) * 0.8)]
        validation = all_data[int(len(self.eval.items()) * 0.8):]
        return (training, validation)

    ##
    # sortAttr
    #
    # This a helper function for sorting the frontierNodes
    ##
    def sortAttr(self, node):
      return node.steps

    ##
    #heuristicStepsToGoal
    #Description: Gets the number of moves to get to a winning state from the current state
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #                 This will assumed to be a fast clone of the state
    #                 i.e. the board will not be needed/used
    ##
    def heuristicStepsToGoal(self, currentState):
      myState = currentState.fastclone()
      me = myState.whoseTurn
      enemy = abs(me - 1)
      myInv = getCurrPlayerInventory(myState)
      myFood = myInv.foodCount
      enemyInv = getEnemyInv(self, myState)
      tunnels = getConstrList(myState, types=(TUNNEL,))
      myTunnel = tunnels[1] if (tunnels[0].coords[1] > 5) else tunnels[0]
      enemyTunnel = tunnels[0] if (myTunnel is tunnels[1]) else tunnels[1]
      hills = getConstrList(myState, types=(ANTHILL,))
      myHill = hills[1] if (hills[0].coords[1] > 5) else hills[0]
      enemyHill = hills[1] if (myHill is hills[0]) else hills[0]
      enemyQueen = enemyInv.getQueen()

      foods = getConstrList(myState, None, (FOOD,))

      myWorkers = getAntList(myState, me, (WORKER,))
      myOffense = getAntList(myState, me, (SOLDIER,))
      enemyWorkers = getAntList(myState, enemy, (WORKER,))

      # "steps" val that will be returned
      occupyWin = 0

      # keeps one offensive ant spawned
      # at all times
      if len(myOffense) < 1:
        occupyWin += 20
      elif len(myOffense) <= 2:
        occupyWin += 30

      # encourage more food gathering
      if myFood < 1:
        occupyWin += 20

      # never want 0 workers
      if len(myWorkers) < 1:
        occupyWin += 100
      if len(myWorkers) > 1:
        occupyWin += 100

      # want to kill enemy queen
      if enemyQueen == None:
        occupyWin -= 1000

      # calculation for soldier going
      # to kill enemyworker and after
      # going to sit on enemy anthill
      dist = 100
      for ant in myOffense:
        if len(enemyWorkers) == 0:
          if not enemyQueen == None:
            dist = approxDist(ant.coords, enemyHill.coords)
        else:
          dist = approxDist(ant.coords, enemyWorkers[0].coords) + 10
          if len(enemyWorkers) > 1:
            dist += 10

      occupyWin += (dist) + (enemyHill.captureHealth)

      # Gather food
      foodWin = occupyWin
      if not len(myOffense) > 0:
        foodNeeded = 11 - myFood
        for w in myWorkers:
          distanceToTunnel = approxDist(w.coords, myTunnel.coords)
          distanceToHill = approxDist(w.coords, myHill.coords)
          distanceToFood = 9999
          for food in foods:
            if approxDist(w.coords, food.coords) < distanceToFood:
              distanceToFood = approxDist(w.coords, food.coords)
          if w.carrying: # if carrying go to hill/tunnel
            foodWin += min(distanceToHill, distanceToTunnel) - 9.5
            if w.coords == myHill.coords or w.coords == myTunnel.coords:
              foodWin += 1.5
            if not len(myOffense) == 0:
              foodWin -= 1
          else: # if not carrying go to food
            if w.coords == foods[0].coords or w.coords == foods[1].coords:
              foodWin += 1.2
              break
            foodWin += distanceToFood/3 - 1.5
            if not len(myOffense) == 0:
              foodWin -= 1
        occupyWin += foodWin * (foodNeeded)

      # encourage queen killing or sitting on tunnel
      if not enemyQueen == None:
        if enemyQueen.coords == enemyHill.coords:
          occupyWin += 20

      return 1-math.exp(-0.001*(occupyWin))

class Cache:
    def __init__(self, state):
        self.foodCoords = [0]*2
        self.depositCoords = [0]*2
        self.rtt = [0]*2

        foods = getConstrList(state, None, (FOOD,))
        for player in [0,1]:
            deposits = getConstrList(state, player, (ANTHILL, TUNNEL))

            #find the best combo, based on steps to reach one to the other
            bestCombo = min([(d, f) for d in deposits for f in foods], key=lambda pair: stepsToReach(state, pair[0].coords, pair[1].coords))

            self.depositCoords[player] = bestCombo[0].coords
            self.foodCoords[player] = bestCombo[1].coords

            self.rtt[player] = approxDist(self.depositCoords[player], self.foodCoords[player])+1

globalCache = None

def buildCache(state):
    global globalCache

    if globalCache is None or not cacheValid(state):
        globalCache = Cache(state)

#check whether the cache still refers to the current game
def cacheValid(state):
    allFood = [food.coords for food in getConstrList(state, None, (FOOD,))]
    allDeposits = [deposit.coords for deposit in getConstrList(state, None, (ANTHILL, TUNNEL))]
    return all(foodCoord in allFood for foodCoord in globalCache.foodCoords) and \
           all(depositCoord in allDeposits for depositCoord in globalCache.depositCoords)

# evaluate the utility of a state from a given player's perspective
# return a tuple of relevant unweighted components
def utilityComponents(state, perspective):
    enemy = 1-perspective

    # get lists for ants
    myWorkers = getAntList(state, perspective, types=(WORKER,))
    enemyWorkers = getAntList(state, enemy, types=(WORKER,))

    myWarriors = getAntList(state, perspective, types=(DRONE,SOLDIER,R_SOLDIER))
    enemyWarriors = getAntList(state, enemy, types=(DRONE,SOLDIER,R_SOLDIER))

    myQueen = state.inventories[perspective].getQueen()
    enemyQueen = state.inventories[enemy].getQueen()

    foodCoords = globalCache.foodCoords[perspective]
    depositCoords = globalCache.depositCoords[perspective]
    anthillCoords = state.inventories[perspective].getAnthill().coords

    # it's bad if the queen is on the food
    queenInTheWayScore = 0

    queenCoords = myQueen.coords
    if queenCoords in [foodCoords, depositCoords, anthillCoords]:
        queenInTheWayScore -= 1

    queenHealthScore = myQueen.health

    workerDistScore = 0
    workerDangerScore = 0
    for worker in myWorkers:

        # If the worker is carrying food, add the distance to the tunnel to the score
        if worker.carrying == True:
            distanceFromTunnel = approxDist(worker.coords, depositCoords)
            workerDistScore -= distanceFromTunnel

        # if the worker is not carrying food, add the distance from the food and tunnel to the score
        else:
            distTunnelFood = approxDist(foodCoords, depositCoords)
            workerDistScore -= distTunnelFood
            distanceFromFood = approxDist(worker.coords, foodCoords)
            workerDistScore -= distanceFromFood

        #its bad to be close to enemy warriors
        for warrior in enemyWarriors:
            #warriorRange = UNIT_STATS[warrior.type][RANGE] + UNIT_STATS[warrior.type][MOVEMENT]
            if approxDist(worker.coords, warrior.coords) < 2:
                workerDangerScore -= 1

    # Aim to attack workers, if there are no workers, aim to attack queen
    if len(enemyWorkers) != 0:
        targetCoords = enemyWorkers[0].coords
    else:
        targetCoords = enemyQueen.coords

    warriorDistScore = 0
    # Add distance from fighter ants to their targets to score, with a preference to move vertically
    for warrior in myWarriors:
        warriorDistScore -= (warrior.coords[0] - targetCoords[0])**2
        warriorDistScore -= (warrior.coords[1] - targetCoords[1])**2

    #do we have an attacker?
    attackScore = UNIT_STATS[myWarriors[0].type][ATTACK] if len(myWarriors) == 1 else 0

    # punishment for if the enemy has workers
    enemyWorkerScore = - (len(enemyWorkers) * len(myWarriors))

    # Heavy punishment for not having workers, since workers are needed to win
    noWorkerScore = -1 if len(myWorkers) == 0 else 0

    foodScore = state.inventories[perspective].foodCount

    antCountScore = -len(getAntList(state, perspective))

    return [sigmoid(component, 0.01) for component in (queenInTheWayScore, workerDistScore, workerDangerScore, warriorDistScore, enemyWorkerScore,
            noWorkerScore, foodScore, attackScore, antCountScore, queenHealthScore)]

##
# Node Class
#
# Defines how our Node is set up to use for searching
#
##
class Node:
  def __init__(self, move, state, depth, steps, parent):
    self.move = move
    self.state = state
    self.depth = depth
    self.steps = steps + self.depth
    self.parent = parent


class NeuralNetwork:

    # Create a neueral network with random weights given the shape of the network
    # numInputs and numOutputs are ints and
    # hidden layers is a list of ints, representing the number of nodes in each hidden layer
    def __init__(self, numInputs, numOutputs, hiddenLayers, learningRate):
        self.numInputs = numInputs
        self.numOutputs = numOutputs

        layerSizes = [numInputs] + hiddenLayers + [numOutputs]

        # create a list of matrices each representing a layer in the neural network
        # number of rows based on output, number of columns based on input + bias
        self.layers = [self.randomMatrix(layerSizes[i+1] , layerSizes[i] + 1) for i in range(len(layerSizes)-1)]

        self.learningRate = learningRate

    def save(self, filepath):
      np.save(filepath, self.layers)

    def load(self, filepath):
      #try:
      self.layers = np.load(filepath, allow_pickle=True)
      #except:
      #  print("Could not load NN")

    def randomMatrix(self, numRows, numCols):
        # create a matrix of random values in range [-1, 1)
        return 2 * np.random.random_sample((numRows, numCols)) - 1

    # given an input, return the result of the nueral network
    # input should be a list of length self.numInputs
    # returns a list of lists representing the output from each layer
    def evaluateComplete(self, input):

        layerOutputs = []

        for layer in self.layers:
            # append the bias
            input = np.append(input, [1])

            input = np.matmul(layer, input)

            input = self.activationFunction(input)

            layerOutputs.append(input)

        return layerOutputs

    # given an input, return the result of the nueral network
    # input should be a list of length self.numInputs
    # returns a list of length self.numOutputs
    def evaluate(self, input):
        #return the output from the last layer
        return self.evaluateComplete(input)[-1]

    # given a list of values, apply the activation function to each and
    # return a list of the same length
    def activationFunction(self, input):
        return [sigmoid(x) for x in input]



    # given an input list and its target outputs,
    # adjust the weights of this neural network
    def train(self, input, target):
        actualLayerOutputs = self.evaluateComplete(input)

        # compute error terms of output nodes
        actual = actualLayerOutputs[-1]

        # a list of lists, where errorTerms[i] is a list of error terms of the ith layer
        errorTerms = [ [ (target[i]-actual[i])*actual[i]*(1-actual[i]) for i in range(len(actual))] ]

        #compute errors of hidden nodes
        for layerIndex in range(len(self.layers)-2, -1, -1):
            currLayer = self.layers[layerIndex]
            # the layer in front of the layer we are currently calculating weights for
            frontLayer = self.layers[layerIndex+1]

            # the error of each node in the current layer
            layerErr = []

            #number of weights used to calculate error (doesn't include bias)
            numWeights = frontLayer.shape[0]

            # term of the fist node in the first hidden layer
            for h in range(currLayer.shape[0]):
                layerErr.append( sum( [frontLayer[i][h] * errorTerms[0][i] for i in range(numWeights)] ) )

            layerErrTerms = [layerErr[i]*sigmoidPrime(actualLayerOutputs[layerIndex][i]) for i in range(len(layerErr))]

            errorTerms = [layerErrTerms] + errorTerms

        # apply error terms to all weights
        for layerIndex in range(len(self.layers)):
            layer = self.layers[layerIndex]

            #add error term to each value in the row
            for n in range(layer.shape[0]):
                for w in range(len(layer[n])):
                    if w == len(layer[n])-1:
                      inputValue = 1
                    elif layerIndex == 0:
                      inputValue = input[w] 
                    else:
                      inputValue = actualLayerOutputs[layerIndex-1][w]
                    layer[n][w] += self.learningRate*errorTerms[layerIndex][n]*inputValue

def sigmoid(x, p=1):
    return 1/(1+ math.exp(-p*x))

# given the value of sigmoid, return its slope
def sigmoidPrime(sig):
    return sig*(1-sig)

def testNN():
    np.random.seed(10)
    nn = NeuralNetwork(2, 2, [2], .1)

    print(nn.evaluate([1, 2]))

def testTrain():
    np.random.seed(10)
    nn = NeuralNetwork(2, 2, [2], .1)

    nn.train([1,2], [.5,.5])

if __name__ == '__main__':
    testTrain()