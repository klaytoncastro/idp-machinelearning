# Aprendizado por Reforço


## Exemplo de Implementação do Minimax

```python

import math
import random

# Função para verificar se alguém venceu
def check_winner(board):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),
                            (0, 4, 8), (2, 4, 6)]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != ' ':
            return board[combo[0]]
    return None

# Função para verificar se o tabuleiro está cheio
def is_board_full(board):
    return ' ' not in board

# Função Minimax
def minimax(board, is_maximizing):
    winner = check_winner(board)
    if winner == 'X':
        return 1
    if winner == 'O':
        return -1
    if is_board_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(board, False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(board, True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

# Função para encontrar o melhor movimento para o jogador X (Minimax)
def best_move(board):
    best_score = -math.inf
    move = None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'X'
            score = minimax(board, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    return move

# Função para imprimir o tabuleiro
def print_board(board):
    for row in [board[i:i+3] for i in range(0, 9, 3)]:
        print('|'.join(row))
        print("-" * 5)

# Função principal para jogar contra o Minimax com sorteio
def play_game():
    board = [' '] * 9
    print("Bem-vindo ao Jogo da Velha!")
    print_board(board)

    player = input("Você quer ser X ou O? ").upper()
    if player not in ['X', 'O']:
        print("Escolha inválida! O padrão será X.")
        player = 'X'

    ai = 'O' if player == 'X' else 'X'

    # Sorteio para decidir quem começa
    first_player = random.choice(['Você', 'IA Minimax'])
    print(f"{first_player} começa!")

    if first_player == 'Você':
        while True:
            # Sua jogada
            human_turn(board, player)
            if check_game_over(board, player):
                break

            # Minimax (IA) joga
            ai_turn(board, ai)
            if check_game_over(board, ai):
                break
    else:
        while True:
            # Minimax (IA) joga
            ai_turn(board, ai)
            if check_game_over(board, ai):
                break

            # Sua jogada
            human_turn(board, player)
            if check_game_over(board, player):
                break

# Função para a jogada do jogador humano
def human_turn(board, player):
    while True:
        try:
            move = int(input(f"Sua vez ({player}). Escolha uma posição (0-8): "))
            if board[move] == ' ':
                board[move] = player
                break
            else:
                print("Posição ocupada, escolha outra.")
        except (ValueError, IndexError):
            print("Escolha um número válido entre 0 e 8.")
    print_board(board)

# Função para a jogada do Minimax (IA)
def ai_turn(board, ai):
    print(f"Turno do {ai} (IA Minimax).")
    move = best_move(board)
    board[move] = ai
    print_board(board)

# Função para verificar se o jogo acabou
def check_game_over(board, player):
    winner = check_winner(board)
    if winner:
        print(f"{winner} venceu!")
        return True
    elif is_board_full(board):
        print("Empate!")
        return True
    return False

# Inicia o jogo
play_game()

```

## Exemplo de Implementação do Q-Learning

```python

import numpy as np
import random

# Parâmetros do Q-learning
alpha = 0.5  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
epsilon = 0.1  # Probabilidade de exploração inicial
decay_rate = 0.99  # Taxa de decaimento de epsilon
q_table = {}  # A tabela Q será inicializada conforme os estados aparecem

# Funções auxiliares para o jogo da velha
def initialize_board():
    return [' '] * 9

def print_board(board):
    for row in [board[i:i+3] for i in range(0, 9, 3)]:
        print('|'.join(row))
        print("-" * 5)

def is_winner(board, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),
                            (0, 4, 8), (2, 4, 6)]
    return any(board[a] == board[b] == board[c] == player for a, b, c in winning_combinations)

def is_draw(board):
    return ' ' not in board

def available_actions(board):
    return [i for i in range(9) if board[i] == ' ']

def next_state(board, action, player):
    new_board = board[:]
    new_board[action] = player
    return new_board

# Função que escolhe a próxima ação do agente
def choose_action(state, epsilon):
    actions = available_actions(state)
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        q_values = [q_table.get((tuple(state), a), 0) for a in actions]
        return actions[np.argmax(q_values)]

# Função que atualiza a tabela Q
def update_q_table(state, action, reward, next_state, epsilon):
    actions_next = available_actions(next_state)
    next_max = max([q_table.get((tuple(next_state), a), 0) for a in actions_next], default=0)
    q_table[(tuple(state), action)] = q_table.get((tuple(state), action), 0) + alpha * (reward + gamma * next_max - q_table.get((tuple(state), action), 0))

# Função para jogar uma partida completa com escolha aleatória de quem começa
def play_against_agent_random():
    board = initialize_board()

    # Sorteio para decidir quem começa
    first_player = random.choice(['Você', 'Agente'])
    print(f"{first_player} começa.")

    if first_player == 'Você':
        # Você começa como X
        play_turn(board, 'X', 'Você', 'Agente')
    else:
        # O agente começa como X
        play_turn(board, 'O', 'Agente', 'Você')

def play_turn(board, player, current_player, opponent):
    state = board[:]

    while True:
        if current_player == 'Você':
            # Jogador humano joga
            while True:
                try:
                    player_action = int(input("Sua jogada (0-8): "))
                    if player_action in available_actions(state):
                        next_board = next_state(state, player_action, player)
                        break
                    else:
                        print("Movimento inválido. Tente novamente.")
                except ValueError:
                    print("Por favor, insira um número válido.")

            print("\nVocê jogou:")
            print_board(next_board)

            if is_winner(next_board, player):
                print("Você venceu!")
                return
            elif is_draw(next_board):
                print("Empate!")
                return
        else:
            # Agente joga
            action = choose_action(state, 0)  # Epsilon = 0 para jogar com a política aprendida
            next_board = next_state(state, action, player)
            print(f"\n{opponent} jogou:")
            print_board(next_board)

            if is_winner(next_board, player):
                print(f"{opponent} venceu!")
                return
            elif is_draw(next_board):
                print("Empate!")
                return

        # Alterna os jogadores
        state = next_board[:]
        if current_player == 'Você':
            current_player, opponent = opponent, current_player
            player = 'O' if player == 'X' else 'X'  # Alterna entre X e O
        else:
            current_player, opponent = opponent, current_player
            player = 'O' if player == 'X' else 'X'

# Função para treinar o agente sem interação (simulação)
def play_game(epsilon):
    board = initialize_board()
    state = board[:]

    while True:
        # Jogador X (agente) joga
        action = choose_action(state, epsilon)
        next_board = next_state(state, action, 'X')

        if is_winner(next_board, 'X'):
            update_q_table(state, action, 1, next_board, epsilon)  # Vitória do X
            break
        elif is_draw(next_board):
            update_q_table(state, action, 0, next_board, epsilon)  # Empate
            break
        else:
            update_q_table(state, action, 0, next_board, epsilon)  # Jogo continua

        # Oponente joga (jogador O - aleatório)
        opponent_action = random.choice(available_actions(next_board))
        next_board = next_state(next_board, opponent_action, 'O')

        if is_winner(next_board, 'O'):
            update_q_table(state, action, -1, next_board, epsilon)  # Derrota de X
            break
        elif is_draw(next_board):
            update_q_table(state, action, 0, next_board, epsilon)  # Empate
            break

        # Passa para o próximo estado
        state = next_board[:]

# Treinamento do agente
def train_agent(num_games):
    global epsilon
    for i in range(num_games):
        play_game(epsilon)
        epsilon *= decay_rate  # Decaimento de epsilon para reduzir a exploração

# Treinando o agente
train_agent(10000)  # Treina o agente com 10.000 partidas

# Agora você pode jogar contra o agente com chance aleatória de começar
play_against_agent_random()

```