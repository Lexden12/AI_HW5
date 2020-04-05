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

        hiddenLayers = [20]
        learningRate = 1

        self.nn = NeuralNetwork(8, 1, hiddenLayers, learningRate)
        self.useNN = True # This is how you toggle between training(False) vs using the model(True)
        if self.useNN:
          #self.nn.load('../thoma20_schendel21_nn.npy') # This is used to load a trained model if you do not want to used the pre-trained model
          self.nn.useSavedWeights() # Loads the pre-trained model for use. Comment this out and uncomment load to use a model that you have trained
          
        self.eval = {}
        self.moveCount = 0
        self.gameCount = 0

        fileName = 'NeuralNetData/{}_{}.csv'.format(hiddenLayers, learningRate)
        self.resultFile = open(fileName, 'w')
        self.resultFile.write("Game, Training Size, Validation Size, Min Error, Mean Error, Max Error\n")

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
            self.gameCount += 1

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
        print("Game:", self.gameCount)
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

        # print statistics to console and file
        minError = min(err)
        maxError = max(err)
        meanError = sum(err)/len(err)

        print("Min Error: {}".format(minError))
        print("Avg Error: {}".format(meanError))
        print("Max Error: {}\n".format(maxError))

        self.resultFile.write('{}, {}, {}, {}, {}, {}\n'.format(
            self.gameCount, len(training), len(validation), minError, meanError, maxError))
        self.resultFile.flush()

        #clear the data set
        self.eval = {}

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

    myWarriors = getAntList(state, perspective, types=(SOLDIER,))
    enemyWarriors = getAntList(state, enemy, types=(DRONE,SOLDIER,R_SOLDIER))

    myQueen = state.inventories[perspective].getQueen()
    enemyQueen = state.inventories[enemy].getQueen()

    foodCoords = globalCache.foodCoords[perspective]
    depositCoords = globalCache.depositCoords[perspective]
    anthillCoords = state.inventories[perspective].getAnthill().coords
    myInv = getCurrPlayerInventory(state)
    myFood = myInv.foodCount
    tunnels = getConstrList(state, types=(TUNNEL,))
    myTunnel = tunnels[1] if (tunnels[0].coords[1] > 5) else tunnels[0]
    enemyTunnel = tunnels[0] if (myTunnel is tunnels[1]) else tunnels[1]
    hills = getConstrList(state, types=(ANTHILL,))
    myHill = hills[1] if (hills[0].coords[1] > 5) else hills[0]
    enemyHill = hills[1] if (myHill is hills[0]) else hills[0]

    offense = len(myWarriors)
    food = myFood
    workers = len(myWorkers)
    queen = 1 if enemyQueen is not None else 0
    tempScore = 0
    totalDist = 0
    if len(myWorkers) > 0:
        for worker in myWorkers:
          totalDist += approxDist(worker.coords, myQueen.coords)
          if worker.carrying: # if carrying go to hill/tunnel
            tempScore += 2
            distanceToTunnel = approxDist(worker.coords, myTunnel.coords)
            distanceToHill = approxDist(worker.coords, myHill.coords)
            dist = min(distanceToHill, distanceToTunnel)
            if dist <= 3:
              tempScore += 1
          else: # if not carrying go to food
            dist = approxDist(worker.coords, foodCoords)
            if dist <= 3:
              tempScore += 1
        workerScore = tempScore
        workerFood = (totalDist/len(myWorkers))
    else:
      workerScore = workerFood = 0
    workerDist = 0
    anthillDist = 0
    if len(myWarriors) > 0:
        for ant in myWarriors:
          if len(enemyWorkers) > 0:
            workerDist += approxDist(ant.coords, enemyWorkers[0].coords)
          anthillDist += approxDist(ant.coords, enemyHill.coords)
        warriorWorkers = (workerDist/len(myWarriors))
        warriorAnthill = (anthillDist/len(myWarriors))
    else:
      warriorAnthill = warriorWorkers = 1000000
    return [sigmoid(component) for component in (offense, food, workers, queen, workerScore,
                                    workerFood, warriorWorkers, warriorAnthill)]

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

    #Save weights to a file (when training)
    def save(self, filepath):
      np.save(filepath, self.layers)

    #Load weights from a file
    def load(self, filepath):
      #try:
      self.layers = np.load(filepath, allow_pickle=True)
      #except:
      #  print("Could not load NN")

    #Saved weights that function well (1 hidden layer with 20 nodes)
    def useSavedWeights(self):
      saved_weights = [[[-1.89249502e+00,  7.73376175e-01, -7.85464545e-01,
         8.02923356e-01,  5.15405736e-01, -1.26283784e+00,
         8.36500794e-01, -8.68772606e-01,  2.38949472e-01],
       [ 5.28244794e-01,  8.23054530e-01,  1.71275461e-01,
         6.42923988e-01,  3.33550132e-03, -6.04578299e-01,
        -1.83014613e-01,  5.43142115e-01, -1.15625848e+00],
       [-4.91705469e+00, -1.52180689e-01,  2.32509649e-01,
        -9.67160811e-02, -5.23124718e-02,  1.04306560e-02,
         1.31853518e+00,  3.59214700e-01,  5.38714478e-01],
       [ 2.98108165e+00,  4.27807565e-01,  7.55411421e-01,
        -1.11534203e+00, -1.06283153e-01, -6.14401566e-01,
        -2.67978387e-04, -1.07142045e+00,  5.81633352e-01],
       [ 1.40219239e+00,  9.00406966e-01, -2.40981727e-01,
        -8.55858064e-01, -7.32374515e-01,  4.91248813e-01,
         2.92498234e-02, -7.19945657e-04, -3.33115929e-01],
       [ 3.37269204e+00,  1.01067028e+00, -1.14461223e+00,
        -7.63039746e-01,  7.89895121e-01, -2.54110007e-01,
         6.77293874e-01, -6.31598229e-01, -1.06898098e+00],
       [ 1.80374063e+00, -4.54697451e-01,  8.41714227e-01,
         2.92953922e-01, -5.00214608e-01, -1.45303399e-01,
        -7.02467493e-01, -3.44785886e-01,  3.09677292e-01],
       [ 2.07814902e+00,  5.43375517e-02, -7.62888387e-01,
        -7.32533642e-01,  4.57485996e-02,  4.78968409e-01,
         2.55541936e-01,  3.56120108e-01, -9.64236068e-01],
       [-2.75505949e+00,  2.73984549e-01,  7.60476823e-02,
         9.57733214e-01,  1.90059776e-01,  6.55895860e-01,
        -1.01524655e+00,  5.40559900e-01,  3.70169746e-01],
       [ 3.19269091e-01, -1.46177655e-01,  4.59350236e-01,
         6.14793297e-01, -2.38196142e-01, -9.89212951e-01,
        -3.13981959e-01,  1.92628463e-01, -1.11163572e+00],
       [ 1.22733474e-01, -1.00542222e+00,  9.84094842e-02,
         5.07897031e-01, -7.08469422e-01,  4.65970280e-01,
        -1.80755113e-01,  7.39148286e-01,  6.90400993e-01],
       [ 3.08185132e+00,  1.70725590e-01, -8.41539379e-01,
        -1.28302892e-01, -5.97587534e-01,  1.81268897e-02,
        -1.09796005e+00,  5.75450909e-01,  4.55901979e-01],
       [ 1.23404206e+00,  3.67789282e-01,  7.44102816e-02,
        -8.15571004e-02, -2.12947644e-01, -5.25521579e-01,
        -7.84786469e-01,  3.05280995e-01,  4.85909802e-01],
       [ 1.97901643e-01,  4.49931946e-01, -6.81248945e-01,
        -4.49558765e-01,  1.36443782e-03, -3.43396864e-01,
        -3.93081176e-01, -7.81026130e-01, -1.02835677e+00],
       [-2.47245667e+00, -1.08118925e+00, -2.67488963e-02,
        -3.06343583e-02,  4.92922436e-01,  9.19089662e-01,
         3.06969350e-01, -1.05214415e-01, -2.33947982e-01],
       [-1.38896849e+00, -1.11004156e-02,  7.08624886e-01,
         5.01014400e-01, -8.65476373e-02, -2.80635378e-01,
         3.46110099e-01, -9.66232333e-01, -4.99543248e-03],
       [-1.63497298e+00,  8.50328716e-01,  8.58858108e-01,
         8.89970555e-01, -7.25292276e-01, -7.23434342e-01,
         2.18110080e-01, -4.68515005e-01, -3.26403886e-01],
       [-1.02471011e+00, -1.04628437e+00,  7.11188624e-01,
         1.01809753e+00, -4.31412845e-01, -1.42178749e-01,
        -9.28163997e-02, -2.60272651e-01,  1.17228943e-01],
       [-9.27499634e-01,  2.25302886e-01, -3.63618315e-03,
         2.89357663e-01, -8.29214343e-01, -6.01068344e-01,
        -3.58079150e-01,  3.34687899e-01,  2.78399677e-01],
       [-1.41421166e+00,  7.53845336e-02, -7.78247190e-01,
        -2.76292129e-02,  9.25934769e-02, -6.88054340e-01,
         2.61486361e-01,  7.57701828e-01, -4.15031750e-02]],
      [[ 1.73234223, -0.19525996,  4.10077343, -2.00213079, -0.99918482,
        -3.16528213, -0.67794932, -1.73533533,  2.91964606,  0.42467776,
         0.59455297, -1.95905849, -0.13903912, -0.59835941,  2.30917294,
         1.37311743,  1.75054948,  1.50435366,  0.94058212,  1.24027372,
         0.1132439 ]]]
      self.layers = saved_weights

    # create a matrix of random values in range [-1, 1)
    def randomMatrix(self, numRows, numCols):
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

#Sanity test on the evaluate function of the NN
def testNN():
    np.random.seed(10)
    nn = NeuralNetwork(2, 2, [2], .1)

    print(nn.evaluate([1, 2]))

#Sanity test on the train function of the NN
def testTrain():
    np.random.seed(10)
    nn = NeuralNetwork(2, 2, [2], .1)

    nn.train([1,2], [.5,.5])

if __name__ == '__main__':
    testTrain()