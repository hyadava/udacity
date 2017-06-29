"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math
import numpy as np


moves = list()
recur_depth = 0

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score_1(game, player):
    active = game.active_player
    inactive = game.inactive_player
    active_moves = game.get_legal_moves(active)
    inactive_moves = game.get_legal_moves(inactive)

    #score = float(100 * ((len(active_moves) + 1) /(len(inactive_moves) + 1)))
    score = math.exp(len(active_moves) - len(inactive_moves))
    return score

def custom_score_2(game, player):


    active = game.active_player
    inactive = game.inactive_player
    active_moves = game.get_legal_moves(active)
    inactive_moves = game.get_legal_moves(inactive)

    a1 = np.array(active_moves)
    a2 = np.array(inactive_moves)

    overlap_sz = 100

    if(len(active_moves) > 1):
        a1x = np.array(a1[0])
        a1y = np.array(a1[1])
        a1_bb = ((np.min(a1x), np.min(a1y)), (np.max(a1x), np.max(a1y))) 

    if(len(active_moves) > 1 and len(inactive_moves) > 1):
        a2x = np.array(a2[0])
        a2y = np.array(a2[1])
        a2_bb = ((np.min(a2x), np.min(a2y)), (np.max(a2x), np.max(a2y))) 

        overlap = [max(a1_bb[0][0], a2_bb[0][0]), min(a1_bb[1][0], a2_bb[1][0]), max(a1_bb[0][1], a2_bb[0][1]), min(a1_bb[1][1], a2_bb[1][1])]
        overlap_sz = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])

    if(overlap_sz <= 0):
        if(len(active_moves) - len(inactive_moves) > 0):
            score = 20
        else:
            score = -20
    else:
        score = len(active_moves) - len(inactive_moves)

    return float(score)

def custom_score_3(game, player):
    active = game.active_player
    inactive = game.inactive_player
    active_moves = game.get_legal_moves(active)
    inactive_moves = game.get_legal_moves(inactive)

    a1 = np.array(active_moves)
    a2 = np.array(inactive_moves)

    overlap_sz = 100

    #create a bounding box from the coordinates of the legal moves
    #of both players
    if(len(active_moves) > 1):
        a1x = np.array(a1[0])
        a1y = np.array(a1[1])
        a1_bb = ((np.min(a1x), np.min(a1y)), (np.max(a1x), np.max(a1y))) 

    if(len(active_moves) > 1 and len(inactive_moves) > 1):
        a2x = np.array(a2[0])
        a2y = np.array(a2[1])
        a2_bb = ((np.min(a2x), np.min(a2y)), (np.max(a2x), np.max(a2y))) 

        #find the overlap in the bounding boxes
        overlap = [max(a1_bb[0][0], a2_bb[0][0]), min(a1_bb[1][0], a2_bb[1][0]), max(a1_bb[0][1], a2_bb[0][1]), min(a1_bb[1][1], a2_bb[1][1])]
        overlap_sz = (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])

    if(overlap_sz <= 0):
        #if the players are isolated and the active player has more moves return a large +ve score
        if(len(active_moves) - len(inactive_moves) > 0):
            score = 900 
            #print('overlap ', overlap_sz, ' ', overlap)
        else:
            #if the players are isolated and the inactive player has more moves return a large -ve score
            score = -500
    else:
        score = float(10 * ((len(active_moves) + 1) /(len(inactive_moves) + 1)))

    return float(score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return custom_score_3(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        intial_time = self.time_left()
        #print('intial_time ', intial_time)

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        max_move = (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            max_score = float("-inf")
            depth = 1
            while(True):
                score = None
                move = None
                if(self.method == 'minimax'):
                    score, move = self.minimax(game, depth)
                elif(self.method == 'alphabeta'):
                    score, move = self.alphabeta(game, depth)
                else:
                    print('unknown method ', self.method)
                    break
                max_move = move
                if(not self.iterative):
                    break
                depth += 1
            return max_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            #print('get_move timed out')
            return max_move

        # Return the best move from the last completed search iteration
        #raise NotImplementedError


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        max_move = (-1, -1)
        max_val = None
        if(depth == 0):
            player = game.inactive_player
            if(maximizing_player):
                player = game.active_player
            return self.score(game, player), max_move
        if(maximizing_player):
            max_val = float("-inf")
            for m in game.get_legal_moves():
                v, _ = self.minimax(game.forecast_move(m), depth-1, False)
                if(v > max_val):
                    max_val = v
                    max_move = m
        else:
            max_val = float("inf")
            max_move = (-1, -1)
            for m in game.get_legal_moves():
                v, _ = self.minimax(game.forecast_move(m), depth-1, True)
                if(v < max_val):
                    max_val = v
                    max_move = m
        return max_val, max_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        #raise NotImplementedError
        max_move = (-1, -1)
        max_val = None
        if(depth == 0):
            return self.score(game, game.inactive_player), max_move
        if(maximizing_player):
            max_val = float("-inf")
            for m in game.get_legal_moves():
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, False)
                if(v > max_val):
                    max_val = v
                    max_move = m
                if(v > alpha):
                    alpha = v
                if(beta <= alpha):
                    break
            return max_val, max_move
        else:
            max_val = float("inf")
            max_move = (-1, -1)
            for m in game.get_legal_moves():
                v, _ = self.alphabeta(game.forecast_move(m), depth-1, alpha, beta, True)
                if(v < max_val):
                    max_val = v
                    max_move = m
                if(v < beta):
                    beta = v
                if(beta <= alpha):
                    break
            return max_val, max_move




