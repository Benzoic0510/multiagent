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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        """
        返回当前 gameState 下的极小极大（minimax）动作，使用 self.depth 和 self.evaluationFunction。

        以下是实现 minimax 时可能有用的一些方法调用：

        gameState.getLegalActions(agentIndex):
        返回某个 agent 的合法动作列表
        agentIndex=0 表示吃豆人，鬼魂的索引 >= 1

        gameState.generateSuccessor(agentIndex, action):
        返回某个 agent 执行动作后的后继游戏状态

        gameState.getNumAgents():
        返回游戏中的 agent 总数

        gameState.isWin():
        返回当前游戏状态是否为胜利状态

        gameState.isLose():
        返回当前游戏状态是否为失败状态
        """

        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            if agentIndex == 0:
                maxEval = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(1, depth, successor)
                    maxEval = max(maxEval, eval)
                return maxEval
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth + 1 if nextAgent == numAgents else depth
                nextAgent = 0 if nextAgent == numAgents else nextAgent
                minEval = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    succ = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(nextAgent, nextDepth, succ)
                    minEval = min(minEval, eval)
                return minEval

        legalActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, succ)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            if agentIndex == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextValue = alphaBeta(1, depth, successor, alpha, beta)
                    value = max(value, nextValue)
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth + 1 if nextAgent == numAgents else depth
                nextAgent = 0 if nextAgent == numAgents else nextAgent
                value = float('inf')

                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextValue = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                    value = min(value, nextValue)
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legalActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, succ, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextValue = expectimax(1, depth, successor)
                    value = max(value, nextValue)
                return value
            else:
                nextAgent = agentIndex + 1
                nextDepth = depth + 1 if nextAgent == numAgents else depth
                nextAgent = 0 if nextAgent == numAgents else nextAgent
                value = 0
                legalActions = gameState.getLegalActions(agentIndex)
                probabilty = 1 / len(legalActions) if legalActions else 0
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextValue = expectimax(nextAgent, nextDepth, successor)
                    value += nextValue * probabilty
                return value

        legalActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, succ)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    legalActions = currentGameState.getLegalActions(0)

    score = 1.2 * currentGameState.getScore()

    foodList = foodGrid.asList()
    if foodList:
        closestFood = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 5.0 / closestFood
        score -= 0.5 * closestFood
    else:
        score += 2.0

    score -= 8.0 * len(foodList)
    score -= 20.0 / (len(foodList) + 1)

    if capsules:
        closestCapsule = min([manhattanDistance(pacmanPos, capsule) for capsule in capsules])
        score += 6.0 / closestCapsule
    else:
        score += 10.0

    score -= 15.0 * len(capsules)

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:
            score += 80.0 / distance
            score += 25 - 1.0 * distance
            score += 1.5 * len(legalActions)
        else:
            if distance < 2:
                score -= 5.0
            elif distance < 6:
                score -= 1.0 / distance

    if len(currentGameState.getLegalActions(0)) == 1:
        score -= 5

    score += 0.5 * len(legalActions)

    return score

# Abbreviation
better = betterEvaluationFunction
