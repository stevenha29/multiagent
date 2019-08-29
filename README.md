# multiagent
AI agent based project

DEPENDENCIES:

python 2.7
Tkinter for Python 2

==========================
CODE WRITTEN BY REPO OWNER:
	multiAgents.py (parts of this file, described below)
==========================

ReflexAgent (Line 55 of multiAgents.py)
	I wrote evaluationFunction under ReflexAgent.  It is a function that gets features of future game states and returns a number depending on the features. Using a couple of features in the game, each feature is given a weight and determines what the function will return. 
	
	The score of each gameState under this evaluation function determine how the ReflexAgent will move.

	To see this agent in action:
		python autograder.py -q q1

Minimax Agent (Line 156)
	An agent using Minimax algorithm with 1 level of maximizer node and 2 levels for minimizer nodes (2 ghosts and 1 pacman).

	To see this agent in action:
		python autograder.py -q q2 (same graphic)

AlphaBetaAgent (Line 208)
	An agent using minimax algorithm along with alpha beta pruning. 1 maximizer level (Pacman) and 2 minimizer levels (2 ghosts)

	to see this agent in action:
		python autograder.py -q q3 (same graphic)

ExpectimaxAgent (Line 328) 
	An agent using expectimax algorithm (instead of minimizer nodes these nodes representing the ghosts are chance nodes with equal probability)

	to see this agent in action:
		python autograder.py -q q4 (same graphic)

betterEvaluationFunction (line 382)
	Uses the same features as the reflex againt but instead of evaluating the best move in the current moment, it evaluates the game state passed into the function. The value returned by this function is the score of the game state passed into the function, determined the sum of the weighted features of the game state passed in.

	to see this agent in action:
		python autograder.py -q q5 (same graphic)
