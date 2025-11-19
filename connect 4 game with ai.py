import pygame, numpy, random, time
from model import *

pygame.init()

# Wichtige Werte
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = 40
PADDING = 50
SCREEN_WIDTH = COLUMN_COUNT * SQUARESIZE + PADDING * 1.5
SCREEN_HEIGHT = ROW_COUNT * SQUARESIZE + PADDING  * 1.5
MODEL_NAME = "csharptrained.pickle" # "connectFour.pickle"
PLAY_AGAINST_AI = True

board = numpy.zeros((ROW_COUNT,COLUMN_COUNT)) #start werte
print (board)
player = 1

#bildschirm setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill((127, 127, 127))
pygame.draw.rect(screen, (10, 10, 255), pygame.Rect(PADDING/2, PADDING/2, SCREEN_WIDTH-PADDING, SCREEN_HEIGHT-PADDING)) #outline

def isFreeColumn(col: int) -> bool:
    return board[0][col] == 0

def change_value_row(col, player):
    for rowrn in range(ROW_COUNT - 1, -1, -1):  #for loop von unterster reihe
        if board[rowrn][col] == 0:
            board[rowrn][col] = player  #verändere wert zu playernummer
            return 2 if player == 1 else 1
    return player  #reihe voll

def win_detection(player): #danke chatgpt <3 (war zu faul)
    #temp player ändern für real-time win detection
    if player == 1:
        player = 2
    else:
        player = 1
    # horizontal
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if board[row][col] == player and \
               board[row][col + 1] == player and \
               board[row][col + 2] == player and \
               board[row][col + 3] == player:
                return True

    #vertical
    for col in range(COLUMN_COUNT):
        for row in range(ROW_COUNT - 3):
            if board[row][col] == player and \
               board[row + 1][col] == player and \
               board[row + 2][col] == player and \
               board[row + 3][col] == player:
                return True

    # diagonal positiv
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            if board[row][col] == player and \
               board[row + 1][col + 1] == player and \
               board[row + 2][col + 2] == player and \
               board[row + 3][col + 3] == player:
                return True

    #diagonal negativ
    for row in range(3, ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            if board[row][col] == player and \
               board[row - 1][col + 1] == player and \
               board[row - 2][col + 2] == player and \
               board[row - 3][col + 3] == player:
                return True
    return False

def update_board(): # kreise zeichnen
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            if board[row][col] == 0:
                pygame.draw.circle(screen, (170, 170, 170), (PADDING + 35 + col * SQUARESIZE, PADDING + 35 + row * SQUARESIZE),RADIUS) #leer
            elif board[row][col] == 1:
                pygame.draw.circle(screen, (255, 255, 80), (PADDING + 35 + col * SQUARESIZE, PADDING + 35 + row * SQUARESIZE),RADIUS) #player 1
            elif board[row][col] == 2:
                pygame.draw.circle(screen, (255, 80, 80), (PADDING + 35 + col * SQUARESIZE, PADDING + 35 + row * SQUARESIZE),RADIUS) #player 2
            if win_detection(1): #win schrift
                screen.blit(pygame.font.Font(None, 120).render("ROT GEWINNT", True, (0, 0, 0)), (PADDING, SCREEN_HEIGHT // 3))
            elif win_detection(2):
                screen.blit(pygame.font.Font(None, 120).render("GELB GEWINNT", True, (0, 0, 0)), (PADDING, SCREEN_HEIGHT // 3))
    pygame.display.update()


update_board()
# hauptloop
running = True
model: Model = Model.load(MODEL_NAME)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # click position
            x, y = event.pos

            # is in window?
            if (PADDING <= x <= SCREEN_WIDTH - PADDING and
                    PADDING <= y <= SCREEN_HEIGHT - PADDING):

                # row berechnen
                col = (x - PADDING) // SQUARESIZE

                player = change_value_row(col, player)
                print("player: " + str(player))
                win_detection(player)
                update_board()
                #print(board)
                if win_detection(player):
                    time.sleep(3)
                    pygame.quit()
                    exit()


                if PLAY_AGAINST_AI:
                    if player == 2: #ai player
                        pred = model.forward([cell for row in board for cell in row] + [player])
                        pred = [(i, pred[i]) for i in range(pred.__len__())]
                        pred = sorted(pred, key=lambda x: x[1], reverse=True) # sort in descending order
                        for (col, percentage) in pred:
                            if isFreeColumn(col):
                                player = change_value_row(col, player)
                                win_detection(player)
                                update_board()
                            #   print(board)
                                break
                        if win_detection(player):
                            time.sleep(3)
                            pygame.quit()
                            exit()