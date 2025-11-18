#   import pandas as pd
#   
#   csvFilePath = "filtered_connect-4.csv"
#   df = pd.read_csv(csvFilePath)
#   # Check if every column except the last contains only 0 for a row
#   def is_valid_row(row):
#       return not (row[:-1] != 0).all()
#   
#   # Filter the rows where the condition is met
#   filtered_df = df[df.apply(is_valid_row, axis=1)]
#   
#   #   filtered_df = df[df.iloc[:, :-1].eq(0.0).any(axis=1)]
#   
#   filtered_csv_path = "filtered_connect-4.2.csv"
#   filtered_df.to_csv(filtered_csv_path, index=False)
#   print("finished")

#	csvFilePath = "G:\My Drive\Informatik\lang\python\connect four nn\c4_game_database.csv"
#	
#	# Load the CSV file # Replace with your file path
#	df = pd.read_csv(csvFilePath)
#	
#	# Ensure the winner column exists
#	winner_column = df.columns[-1]  # Assuming the winner column is the last column
#	
#	# Remove rows where the winner column contains any value but 0
#	df_filtered = df[df[winner_column] == 0]
#	
#	# Display the filtered DataFrame
#	print(f"Original dataset size: {df.shape[0]} rows")
#	print(f"Filtered dataset size: {df_filtered.shape[0]} rows")
#	
#	# Save the filtered dataset to a new CSV file (optional)
#	filtered_csv_path = "filtered_connect-4.csv"
#	df_filtered.to_csv(filtered_csv_path, index=False)
#	print(f"Filtered data saved to {filtered_csv_path}")

from connect_four_bot_keith_galli import pick_best_move, create_board, get_next_open_row, drop_piece, print_board, is_terminal_node
from chatGPT_connect_four_bot import check_win
import numpy, pandas, time, random, csv
#   from chatGPT_connect_four_bot import getBestColumn, evaluate_board

class Board:
    def __init__(self, boardHeight: int = 6, boardWidth: int = 7):
        self.board = [[0 for _ in range(boardWidth)] for __ in range(boardHeight)]
        self.iterCycle = [0 for _ in range(boardHeight*boardWidth)]
        self.BOARD_HEIGHT = boardHeight
        self.BOARD_WIDTH = boardWidth
        self.PIECE_COUNT = boardHeight*boardWidth

        self.iter2Cycle = [0 for _ in range(boardWidth)]
        self.iter2Buff = [[0 for _ in range(boardHeight)] for __ in range(boardWidth)]
        self.iter2LList = [1, 2, 1]

    def clear(self) -> list[int]:
        return [[0 for _ in range(self.BOARD_WIDTH)] for __ in range(self.BOARD_HEIGHT)]
    
    def move(self, column: int = 0, piece: int = 1 | 2):
        for i in range(self.board.__len__() - 1, -1, -1): 
            if self.board[i][column] == 0:
                self.board[i][column] = piece
                return
    
    def iter(self):
        for i in range(self.iterCycle.__len__()):
            self.iterCycle[i] = [1, 2, 0][self.iterCycle[i]]
            if self.iterCycle[i] != 0:
                break
        
        """ # danke chatGPT weil es zu spät ist <3
        emptyBoard = [[0 for _ in range(self.BOARD_WIDTH)] for _ in range(self.BOARD_HEIGHT)]
    
        # Populate the board column by column to simulate gravity
        for col in range(self.BOARD_WIDTH):
            column_values = [self.iterCycle[row * self.BOARD_WIDTH + col] for row in range(self.BOARD_HEIGHT)]
            filtered_values = [value for value in column_values if value != 0]
            
            # Place the filtered values at the bottom of the column
            for row in range(self.BOARD_HEIGHT):
                if row < len(filtered_values):
                    emptyBoard[5 - row][col] = filtered_values[-(row + 1)]
        
        self.board = emptyBoard """


        self.board = self.clear()
        for i in range(self.iterCycle.__len__()):
            if self.iterCycle[i] != 0:
                self.move(i%self.BOARD_WIDTH, self.iterCycle[i])
        return
    
    def print(self):
        print(*self.board, sep="\n")
    
    #   def toBotBoard(self):
    #       botBoard = create_board()
    #       for i in range(self.board.__len__()):
    #           for j in range(self.board[i].__len__()):
    #               drop_piece(botBoard, self.board.__len__() - i - 1, j, self.board[i][j])
    #       print_board(botBoard)
    #       return botBoard

    def getReversed(self):
        board = self.board
        board.reverse()
        return board

    def getBotBoard(self):
        botBoard = numpy.array(self.getReversed())
        #   print_board(botBoard)
        return botBoard
    
    def iter2(self):
        for i in range(self.iter2Cycle.__len__()):
            tempItercycle = self.iter2Cycle[i]
            self.iter2Cycle[i] += 1
            if self.iterCycle[i] >= 2**self.BOARD_HEIGHT:
                self.iter2Buff = [0 for _ in range(self.BOARD_HEIGHT)]
                continue
            xoredStr = bin(tempItercycle^self.iter2Cycle[i])
            for bitIndex in range(xoredStr.__len__()):
                if xoredStr[bitIndex] == "0":
                    continue
                self.iter2Buff[i][bitIndex] = self.iter2LList[self.iter2Buff[i][bitIndex]]
            break
        print(self.iter2Buff)

class Board2:
    def __init__(self, boardHeight: int = 6, boardWidth: int = 7):
        self.BOARD_HEIGHT = boardHeight
        self.BOARD_WIDTH = boardWidth
        self.PIECE_COUNT = boardHeight*boardWidth
        self.board = [[0 for _ in range(boardWidth)] for __ in range(boardHeight)]
        self.moveHistory = [0 for _ in range(self.PIECE_COUNT)]
        self.hashMap = {}
        self.keyID = 0

    def clearBoard(self):
        self.board = [[0 for _ in range(self.BOARD_WIDTH)] for __ in range(self.BOARD_HEIGHT)]
    
    def clearHistory(self):
        self.moveHistory = [0 for _ in range(self.PIECE_COUNT)]

    def genBoardState(self, aiPiece: int = 2):
        self.clearBoard()
        self.clearHistory()
        COLUMN_TRACKER = [self.BOARD_HEIGHT for _ in range(self.BOARD_WIDTH)]
        for i in range(self.PIECE_COUNT):
            nextMove = random.sample([col for col in range(self.BOARD_WIDTH) if COLUMN_TRACKER[col] > 0], 1)[0]
            COLUMN_TRACKER[nextMove] -= 1
            self.moveHistory[i] = nextMove
            self.move(nextMove, i%2 + 1)
            if self.check_win() or i == self.PIECE_COUNT - 1:
                self.printBoard()
                for j in range(i - 1, -1, -1):
                    self.unMove(self.moveHistory[j])
                    currPiece = j%2 + 1
                    tempBoardTuple = tuple([cell for row in self.board for cell in row] + [pick_best_move(self.getBotBoard(), currPiece), currPiece])
                    if tempBoardTuple not in self.hashMap:
                        self.hashMap[self.keyID] = tempBoardTuple
                        self.keyID += 1
                break

    def check_win(self) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))  # Vertical, Horizontal, Diagonal /
        
        def in_bounds(r, c):
            return 0 <= r < self.BOARD_HEIGHT and 0 <= c < self.BOARD_WIDTH
        
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r][c] == 0:
                    continue
                    count = 1
                    for i in range(1, 4):
                        nr, nc = r + dr * i, c + dc * i
                        if in_bounds(nr, nc) and self.board[nr][nc] == self.board[r][c]:
                            count += 1
                        else:
                            break
                    if count == 4:
                        return True  # Win found
        return False  # No win
    
    def move(self, column: int = 0, piece: int = 1 | 2):
        for i in range(self.BOARD_HEIGHT - 1, -1, -1): 
            if self.board[i][column] == 0:
                self.board[i][column] = piece
                return
    
    def unMove(self, column: int = 0):
        for i in range(self.BOARD_HEIGHT):
            if self.board[i][column] != 0:
                self.board[i][column] = 0
                return
    
    def getReversed(self):
        board = self.board
        board.reverse()
        return board

    def getBotBoard(self):
        botBoard = numpy.array(self.getReversed())
        #   print_board(botBoard)
        return botBoard
    
    def printHashMap(self):
        for key in self.hashMap:
            print(f"{key}: {self.hashMap[key]}")
    
    def printBoard(self):
        print(*self.board, sep="\n")

class DataFrameHelper(pandas.DataFrame):
    def __init__(self, columns: list[str]):
        super().__init__(columns=columns)
        self.columns = columns
        self = self.astype(int)

    def addEntry(self, data: tuple[list[list], int]):
        entries = [cell for row in data[0] for cell in row] + [data[1]]
        if entries.__len__() != self.columns.__len__():
            raise ValueError("Entry does not match the number of columns.")
        
        self.loc[self.__len__()] = entries
    
    def addTupleEntry(self, data: tuple[int]):
        entries = list(data)
        if entries.__len__() != self.columns.__len__():
            raise ValueError("Entry does not match the number of columns.")
        
        self.loc[self.__len__()] = entries

class CSVHelper:
    STD_PATH = "boardStates.csv"
    @staticmethod
    def getWriter(filePath: str = STD_PATH):
        with open(filePath, "w") as file:
            return csv.writer(file, delimiter=" ", quotechar="|")
    
    @staticmethod
    def getWriter(filePath: str = STD_PATH):
        with open(filePath) as file:
            return csv.reader(file, delimiter=" ", quotechar="|")

if __name__ == "__main__":
    board = Board2()
    board.genBoardState()
    board.printHashMap()

exit()

if __name__ == "__main__":
    CSV_FILE_PATH = "GC4BStatesNew.csv"

    board = Board2()
    df = DataFrameHelper([str(i) for i in range(board.PIECE_COUNT)] + ["bestCol"])
    AI_PIECE = 2
    DEPTH = 10_000
    START_TIME = time.time()

    print("Generating board states...")
    for i in range(DEPTH):
        board.genBoardState(df)
        if i%1_000 == 0 and i != 0:
            print(f"Progress: {i/DEPTH*100}% - took {time.time() - START_TIME} s")
            ETA = (time.time() - START_TIME)/i*DEPTH - (time.time() - START_TIME)
            print(f"ETA: {ETA} s / {ETA/60} min / {ETA/60/60} h")
            df.to_csv(CSV_FILE_PATH)
    HASH_MAP_LEN = board.hashMap.__len__()
    print(f"Finished generating {HASH_MAP_LEN} board states")
    SAVE_START_TIME = time.time()
    print("Saving board states...")
    SNAPSHOT_AMOUNT = 20
    for i in range(HASH_MAP_LEN):
        df.addTupleEntry(board.hashMap[i])
        if i%(HASH_MAP_LEN//SNAPSHOT_AMOUNT) == 0 and i != 0:
            print(f"Progress: {i/HASH_MAP_LEN*100}% - took {time.time() - SAVE_START_TIME}")
            ETA = (time.time() - SAVE_START_TIME)/i*HASH_MAP_LEN - (time.time() - SAVE_START_TIME)
            print(f"ETA: {ETA} s / {ETA/60} min / {ETA/60/60} h")
            df.to_csv(CSV_FILE_PATH)
    df.to_csv(CSV_FILE_PATH)
    print(f"Finished, states saved to \"{CSV_FILE_PATH}\", took {time.time() - START_TIME} s")

exit()

if __name__ == "__main__":
    CSV_FILE_PATH = "GC4BStatesTest.csv"

    board = Board()
    df = DataFrameHelper([str(i) for i in range(board.PIECE_COUNT)] + ["bestCol"])
    AI_PIECE = 2
    DEPTH = 100_000
    START_TIME = time.time()
    
    #   board.iter()
    #   botBoard = board.getBotBoard()
    #   if is_terminal_node(botBoard):
    #       pass
    #   bestCol = pick_best_move(botBoard, AI_PIECE)
    #   print(f"ETA: {(time.time() - START_TIME)*DEPTH} s")

    board = Board()
    for i in range(DEPTH):
        board.iter()
        botBoard = board.getBotBoard()
        if is_terminal_node(botBoard):
            continue
        bestCol = pick_best_move(botBoard, AI_PIECE)
        df.addEntry((board.board, bestCol))
        if i%10_000 == 0 and i != 0:
            print(f"Progress: {i/DEPTH*100}%")
            print(f"ETA: {(time.time() - START_TIME)/i*DEPTH - (time.time() - START_TIME)} s")
            df.to_csv(CSV_FILE_PATH)
            print(f"Saved snapshot of dataset at \"{CSV_FILE_PATH}\"; time passed: {time.time() - START_TIME} s")
    print("Finished generating board states, begining to remove duplicates")
    noDuplicateDataFrame = df.drop_duplicates()
    noDuplicateDataFrame.to_csv(CSV_FILE_PATH)
    print(f"Finished, states saved to \"{CSV_FILE_PATH}\", took {time.time() - START_TIME} s")

#   if __name__ == "__main__":
#       board = Board()
#       board.move(0, 2)
#       board.move(1, 1)
#       board.move(2, 2)
#       board.move(0, 1)
#       board.move(1, 1)
#       board.move(2, 1)
#       board.move(3, 1)
#       board.move(3, 1)
#       board.move(2, 2)
#       board.move(2, 2)
#       board.move(2, 2)
#       print(pick_best_move(board.getBotBoard(), 2))

#   board = create_board()
#   drop_piece(board, get_next_open_row(board, 4), 4, 1)
#   for _ in range(1):
#       bestCol = pick_best_move(board, 2)
#       drop_piece(board, get_next_open_row(board, bestCol), bestCol, 2)
#       print_board(board)
#   board = Board()
#   for _ in range(100000):
#       board.iter()
#   
#   print(*board.board, sep="\n")
#   print("-"*21)