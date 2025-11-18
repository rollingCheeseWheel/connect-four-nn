import math
import copy

# Constants for the board dimensions
ROWS = 6
COLS = 7

# Scoring weights
BOT = 2
PLAYER = 1
EMPTY = 0

# Function to check for valid columns
def is_valid_move(board, col):
    return board[0][col] == 0

# Function to make a move
def make_move(board, col, player):
    new_board = copy.deepcopy(board)
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == 0:
            new_board[row][col] = player
            return new_board

# Function to check for a win
def check_win(board, player):
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True

    # Check vertical
    for col in range(COLS):
        for row in range(ROWS - 3):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    # Check diagonals (positive slope)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    # Check diagonals (negative slope)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    return False

# Function to evaluate the board score
def evaluate_board(board):
    score = 0

    def count_sequence(sequence, player):
        return sequence.count(player), sequence.count(EMPTY)

    # Horizontal scoring
    for row in range(ROWS):
        for col in range(COLS - 3):
            segment = board[row][col:col + 4]
            bot_count, empty_count = count_sequence(segment, BOT)
            if bot_count == 4:
                return math.inf
            elif bot_count == 3 and empty_count == 1:
                score += 10

    # Vertical scoring
    for col in range(COLS):
        for row in range(ROWS - 3):
            segment = [board[row + i][col] for i in range(4)]
            bot_count, empty_count = count_sequence(segment, BOT)
            if bot_count == 4:
                return math.inf
            elif bot_count == 3 and empty_count == 1:
                score += 10

    # Diagonal scoring
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            pos_diag = [board[row + i][col + i] for i in range(4)]
            bot_count, empty_count = count_sequence(pos_diag, BOT)
            if bot_count == 4:
                return math.inf
            elif bot_count == 3 and empty_count == 1:
                score += 10

    for row in range(3, ROWS):
        for col in range(COLS - 3):
            neg_diag = [board[row - i][col + i] for i in range(4)]
            bot_count, empty_count = count_sequence(neg_diag, BOT)
            if bot_count == 4:
                return math.inf
            elif bot_count == 3 and empty_count == 1:
                score += 10

    return score

# Minimax function with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing_player):
    if check_win(board, BOT):
        return math.inf, None
    if check_win(board, PLAYER):
        return -math.inf, None
    if depth == 0 or all(board[0][col] != 0 for col in range(COLS)):
        return evaluate_board(board), None

    if maximizing_player:
        max_eval = -math.inf
        best_col = None
        for col in range(COLS):
            if is_valid_move(board, col):
                child_board = make_move(board, col, BOT)
                eval_score, _ = minimax(child_board, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        return max_eval, best_col
    else:
        min_eval = math.inf
        best_col = None
        for col in range(COLS):
            if is_valid_move(board, col):
                child_board = make_move(board, col, PLAYER)
                eval_score, _ = minimax(child_board, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        return min_eval, best_col

# Function to get the best column for the bot
def getBestColumn(board, depth=6):
    _, best_col = minimax(board, depth, -math.inf, math.inf, True)
    return best_col

# Example usage
if __name__ == "__main__":
    board = [
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 0, 0, 0, 0]
    ]
    print("Best column for bot:", getBestColumn(board))
