# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		currentPos = currentGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newCapsules = successorGameState.getCapsules()
		newGhostStates = successorGameState.getGhostStates()
		oldGhostStates = currentGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		oldScaredTimes = [ghostState.scaredTimer for ghostState in oldGhostStates]

		"*** YOUR CODE HERE ***"
		addOns = 0
		if (successorGameState.getNumFood() < currentGameState.getNumFood()):
		  addOns = addOns + 10
		  # print "1";
		if (newCapsules < currentGameState.getCapsules()):
		  addOns = addOns + 20
		if (set(newGhostStates) != set(oldGhostStates)):
		  addOns = addOns + 5
		  # print "2"
		if (sum(newScaredTimes) > sum(oldScaredTimes)):
		  addOns = addOns + 1
		  # print "3"
		if newPos == currentGameState.getPacmanPosition():
		  return successorGameState.getScore() - 100
		if (successorGameState.isLose()):
		  # print "4"
		  return -sys.maxint -1
		foodList = []
		w = newFood.width
		h = newFood.height
		for x in range(0, w):
		  for y in range(0, h):
			if newFood[x][y] == True:
			  foodList.append((x, y))
		closestFood = None
		distance = sys.maxint
		distanceToCapsule = sys.maxint
		closestCapsule = None
		for capsule in successorGameState.getCapsules():
		  if distanceToCapsule > manhattanDistance(currentPos, capsule):
		  	distance = manhattanDistance(currentPos, capsule)
		  	closestCapsule = capsule
		for pellet in foodList:
		  if distance > manhattanDistance(currentPos, pellet):
			distance = manhattanDistance(currentPos, pellet)
			closestFood = pellet
		if (closestCapsule != None):
			if manhattanDistance(newPos, closestCapsule) < distanceToCapsule:
				addOns += 10
		if (closestFood != None):
			if manhattanDistance(newPos, pellet) < distance:
				addOns += 5
		return successorGameState.getScore() + addOns + (1/distance)

def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game

		  gameState.isWin():
			Returns whether or not the game state is a winning state

		  gameState.isLose():
			Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"
		def helper(initial, state, agentIndex, currDepth):
			if state.isWin() or state.isLose() or currDepth == -1:
				return self.evaluationFunction(state)
			elif initial == True:
				moves = dict()
				for action in state.getLegalActions(agentIndex):
					moves[action] = helper(False, state.generateSuccessor(0, action), 1, 0)
				return max(moves.keys(), key = lambda x: moves[x])
			numAgents = state.getNumAgents()
			iterations = 0
			depth = self.depth
			actions = state.getLegalActions(agentIndex)
			states = [state.generateSuccessor(agentIndex, action) for action in actions]
			if (agentIndex == numAgents - 1) and (depth - 1 == currDepth):
				return min(helper(False, s, 0, -1) for s in states)
			if agentIndex == 0:
				return max(helper(False, s, agentIndex + 1, currDepth) for s in states)
			elif agentIndex == numAgents - 1:
				return min(helper(False, s, 0, currDepth + 1) for s in states)
			else:
				return min(helper(False, s, agentIndex + 1, currDepth) for s in states)
		return helper(True, gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		###BOOK's ALGORITHM
		###############
		def maxValue(state, agentIndex, currDepth, alpha, beta):
			if (state == gameState and agentIndex == 0 and currDepth == 0):
				moves = dict()
				actions = state.getLegalActions(agentIndex)
				v = -sys.maxint -1
				for action in actions:
					newState = state.generateSuccessor(0, action)
					moves[action] = minValue(newState, agentIndex + 1, currDepth, alpha, beta)
					v = max(v, moves[action])
					if v > beta:
						return v
					alpha = max(alpha, v)
				return max(moves.keys(), key = lambda x: moves[x])
			if state.isWin() or state.isLose():
				return self.evaluationFunction(state)
			elif currDepth == -1:
				return self.evaluationFunction(state)
			v = -sys.maxint - 1
			actions = state.getLegalActions(agentIndex)
			for action in actions:
				s = state.generateSuccessor(agentIndex, action)
				v = max(v, minValue(s, agentIndex + 1, currDepth, alpha, beta))
				if v > beta:
					return v
				alpha = max(alpha, v)
			return v

		def minValue(state, agentIndex, currDepth, alpha, beta):
			lastAgent = bool(agentIndex == state.getNumAgents() - 1)
			lastDepth = bool(currDepth == self.depth - 1)
			if state.isWin() or state.isLose():
				return self.evaluationFunction(state)
			elif currDepth == -1:
				return self.evaluationFunction(state)
			v = sys.maxint
			actions = state.getLegalActions(agentIndex)
			for action in actions:
				s = state.generateSuccessor(agentIndex, action)
				if lastAgent and lastDepth:
					v = min(v, maxValue(s, 0, -1, alpha, beta))
				elif lastAgent:
					v = min(v, maxValue(s, 0, currDepth + 1, alpha, beta))
				else:
					v = min(v, minValue(s, agentIndex + 1, currDepth, alpha, beta))
				if v < alpha:
					return v
				beta = min(beta, v)
			return v

		return maxValue(gameState, 0, 0, -sys.maxint - 1, sys.maxint)

		############
		#CUSTOM ALGO
		############
		# def helper(move, state, agentIndex, currDepth, alphaMax, betaMin):
		# 	if state.isWin() or state.isLose() or currDepth == -1:
		# 		return self.evaluationFunction(state)
		# 	elif move == 0:
		# 		moves = dict()
		# 		v = -sys.maxint - 1
		# 		for action in state.getLegalActions(agentIndex):
		# 			newState = state.generateSuccessor(0, action)
		# 			trueValue = helper(action, newState, agentIndex + 1, currDepth, alphaMax, betaMin)
		# 			v = max(v, trueValue)
		# 			alphaMax = max(alphaMax, v)
		# 			if betaMin <= alphaMax:
		# 				break
		# 			moves[action] = trueValue
		# 		return max(moves.keys(), key = lambda x: moves[x])
		# 	numAgents = state.getNumAgents()
		# 	iterations = 0
		# 	depth = self.depth
		# 	actions = state.getLegalActions(agentIndex)
		# 	# states = [state.generateSuccessor(agentIndex, action) for action in actions]
		# 	maxPlayerCond = bool(agentIndex == 0)
		# 	minPlayerCond = bool(agentIndex != 0)
		# 	lastAgent = bool(agentIndex == numAgents - 1)
		# 	lastDepth = bool(depth -1 == currDepth)
			
		# 	if maxPlayerCond:
		# 		v = -sys.maxint - 1
		# 		for action in actions:
		# 			s = state.generateSuccessor(agentIndex, action)
		# 			v = max(v, helper(move, s, agentIndex + 1, currDepth, alphaMax, betaMin))
		# 			alphaMax = max(alphaMax, v)
		# 			if betaMin < alphaMax:
		# 				break
		# 		return v
		# 	elif minPlayerCond:
		# 		v = sys.maxint
		# 		for action in actions:
		# 			s = state.generateSuccessor(agentIndex, action)
		# 			if lastAgent:
		# 				if lastDepth:
		# 					v = min(v, helper(move, s, 0, -1, alphaMax, betaMin))
		# 				else:
		# 					v = min(v, helper(move, s, 0, currDepth + 1, alphaMax, betaMin))
		# 			else:
		# 				v = min(v, helper(move, s, agentIndex + 1, currDepth, alphaMax, betaMin))
		# 			betaMin = min(v, betaMin)
		# 			if betaMin < alphaMax:
		# 				break
		# 		return v
		# return helper(0, gameState, 0, 0, -sys.maxint + 1, sys.maxint)
		##################
		#MY ALGORITHM
		##################

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		def maxValue(state, agentIndex, currDepth):
			if (state == gameState and agentIndex == 0 and currDepth == 0):
				moves = dict()
				actions = state.getLegalActions(agentIndex)
				v = -sys.maxint -1
				for action in actions:
					newState = state.generateSuccessor(0, action)
					moves[action] = ghostValue(newState, agentIndex + 1, currDepth)
					v = max(v, moves[action])
				return max(moves.keys(), key = lambda x: moves[x])
			if state.isWin() or state.isLose():
				return self.evaluationFunction(state)
			elif currDepth == -1:
				return self.evaluationFunction(state)
			v = -sys.maxint -1
			actions = state.getLegalActions(agentIndex)
			for action in actions:
				s = state.generateSuccessor(agentIndex, action)
				v = max(v, ghostValue(s, agentIndex + 1, currDepth))
			return v

		def ghostValue(state, agentIndex, currDepth):
			lastAgent = bool(agentIndex == state.getNumAgents() - 1)
			lastDepth = bool(currDepth == self.depth - 1)
			if state.isWin() or state.isLose():
				return self.evaluationFunction(state)
			elif currDepth == -1:
				return self.evaluationFunction(state)
			values = []
			actions = state.getLegalActions(agentIndex)
			for action in actions:
				s = state.generateSuccessor(agentIndex, action)
				if lastAgent and lastDepth:
					values.append(maxValue(s, 0, -1))
				elif lastAgent:
					values.append(maxValue(s, 0, currDepth + 1))
				else:
					values.append(ghostValue(s, agentIndex + 1, currDepth))
			return sum(values)/float((len(values)))
		return maxValue(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <take every food pellet, capsule
	   and subtract it from the total score. This makes 
	   new states with less food and less pellets have a better score.
	   If the game is a losing state, then give it the worst score. 
	   If the game is a winning state, give it the best score.>
	"""
	"*** YOUR CODE HERE ***"
	currentPos = currentGameState.getPacmanPosition()
	if currentGameState.isLose():
		return -sys.maxint - 1
	elif currentGameState.isWin():
		return sys.maxint
	totalScore = 0
	for pellet in currentGameState.getCapsules():
		totalScore -= 20
		#-20 for every pellet left
	food = currentGameState.getFood()
	foodExists = False
	foodList = []
	w = food.width
	h = food.height
	for x in range(0, w):
		for y in range(0, h):
			if food[x][y] == True:
				foodExists = True
				foodList.append((x, y))
				totalScore -= 20
				#-4 for every food left
	if (not foodExists):
		return sys.maxint
	closestFood = min(foodList, key = lambda food: manhattanDistance(currentPos, food))
	for ghost in currentGameState.getGhostStates():
		totalScore -= 10
	if len(currentGameState.getCapsules()) > 0:
		closestCapsule = min(currentGameState.getCapsules(), key = lambda c: manhattanDistance(currentPos, c))
		closestDistance = manhattanDistance(closestCapsule, currentPos)
		totalScore -= closestDistance
		return currentGameState.getScore() + totalScore
	if len(foodList) > 0:
		closestFood = min(foodList, key = lambda food: manhattanDistance(food, currentPos))
		closestDistance = manhattanDistance(closestFood, currentPos)
		totalScore -= closestDistance
		return currentGameState.getScore() + totalScore

	return currentGameState.getScore() + totalScore

# Abbreviation
better = betterEvaluationFunction

