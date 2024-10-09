# Aprendizado por Reforço

## 1. Visão Geral

A **inteligência artificial** e o **aprendizado de máquina (Machine Learning)** desempenham um papel fundamental em sistemas modernos, permitindo que computadores aprendam e tomem decisões com base em dados e interações. Dentro desse domínio, um ramo específico se destaca por sua habilidade de tomar decisões em ambientes dinâmicos e incertos: o **Aprendizado por Reforço (Reinforcement Learning)**.

O aprendizado por reforço difere dos métodos tradicionais de aprendizado supervisionado, pois os algoritmos não são treinados com exemplos de entrada e saída corretos. Em vez disso, o agente aprende tomando ações em um ambiente e observando as consequências dessas ações por meio de **recompensas** ou **penalidades**. O objetivo do agente é aprender uma **política** que maximize a recompensa total ao longo do tempo.

Dois algoritmos populares dentro desse campo são:
- **Minimax**, uma abordagem baseada em busca e otimização para resolver jogos de soma zero com informações completas.
- **Q-learning**, um método de aprendizado por reforço que permite ao agente aprender a tomar decisões em ambientes desconhecidos.

Neste trabalho, vamos explorar as implementações de ambos os algoritmos e como eles podem ser aplicados para resolver o clássico **jogo da velha (Tic-Tac-Toe)**, comparando suas performances em diferentes condições.

---

## 2. Minimax: Algoritmo de Busca de Decisões Ótimas

O algoritmo **Minimax** é amplamente utilizado na teoria dos jogos e inteligência artificial para tomar decisões ótimas em situações competitivas de jogos de soma zero, onde dois jogadores estão em oposição direta (como no xadrez, gamão e jogo da velha). O objetivo do algoritmo é minimizar a perda máxima de um jogador, daí o nome "minimax" (mínimo do máximo). Além de ser especialmente útil em jogos de tabuleiro, o Minimax também é a base para muitos algoritmos de tomada de decisão em inteligência artificial.

### Princípios

- **Jogadores**: O algoritmo assume que um jogador (Max) tenta maximizar seu ganho, enquanto o adversário (Min) tenta minimizar esse ganho. Ambos jogam de maneira otimizada e conhecem todas as opções disponíveis.
- **Árvore de Decisão**: O algoritmo percorre uma árvore de decisões do jogo, onde cada nó representa o estado atual do jogo e os ramos representam as ações possíveis dos jogadores.
- **Folhas**: As folhas da árvore são estados finais do jogo, onde uma pontuação é atribuída dependendo de quem venceu ou se houve empate.

### Funcionamento

O algoritmo simula o jogo até o final para avaliar os resultados de todas as jogadas possíveis, funcionando de maneira recursiva:

- **Maximizar o ganho**: Quando é a vez do jogador Max, o algoritmo escolhe a jogada que maximiza o valor obtido, assumindo que o jogador Min também jogará da melhor maneira possível.
- **Minimizar o ganho**: Quando é a vez do jogador Min, o algoritmo escolhe a jogada que minimiza o valor, assumindo que Max jogará de forma ótima.

O Minimax continua alternando entre essas duas fases, avaliando todas as possibilidades até chegar ao resultado final de cada sequência de jogadas.

### Poda Alfa-Beta

Para melhorar a eficiência do Minimax, pode-se aplicar a técnica de **poda alfa-beta**, que elimina ramos da árvore de decisão que não influenciam no resultado final, economizando tempo de processamento.

### Exemplo de Implementação do Minimax

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
```

---

## 3. Q-Learning: Algoritmo de Aprendizado por Reforço

O **Q-learning** é um algoritmo de aprendizado por reforço onde o agente aprende a tomar decisões ótimas com base em uma função de recompensa. Ao contrário do aprendizado supervisionado, onde o modelo é treinado com exemplos rotulados, no Q-learning o agente aprende interagindo diretamente com o ambiente e ajustando seu comportamento para maximizar as recompensas futuras.

### Princípios Básicos

O Q-learning se baseia no conceito de uma **função de valor Q**, que estima a qualidade de uma ação em um determinado estado. Essa função é atualizada em cada interação com o ambiente, permitindo que o agente construa uma tabela de "melhores jogadas" para diferentes situações.

### Equação de Bellman e Atualização da Função Q

A equação de Bellman define como o valor de uma ação \( Q(s, a) \) é atualizado:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]

Onde:

- **Q(s, a)**: Valor da ação \( a \) no estado \( s \).
- **\( \alpha \)**: Taxa de aprendizado, que controla o quão rapidamente o algoritmo atualiza os valores Q.
- **\( r \)**: Recompensa recebida após executar a ação \( a \) no estado \( s \).
- **\( \gamma \)**: Fator de desconto, que define o quanto o agente valoriza recompensas futuras em comparação com recompensas imediatas.
- **\( \max_{a'} Q(s', a') \)**: Valor máximo de Q para o próximo estado \( s' \), estimando a melhor ação futura.

---

### Exploração vs Explotação

Um aspecto crucial do Q-learning é o equilíbrio entre **exploração** e **explotação**:

- **Exploração (Exploration)**: O agente toma ações aleatórias para descobrir novas possibilidades e aprender mais sobre o ambiente.
- **Explotação (Exploitation)**: O agente seleciona as ações com os maiores valores de Q já conhecidos para maximizar a recompensa.

Esse equilíbrio é controlado pelo parâmetro \( \epsilon \) (**epsilon**), que define a probabilidade de o agente explorar ou explorar. O Q-learning implementa uma estratégia chamada **\( \epsilon \)-greedy**, onde o agente explora com uma pequena probabilidade \( \epsilon \) e explora (ou seja, aproveita o conhecimento atual) no restante do tempo.

---

### Fases do Algoritmo Q-Learning

1. **Inicialização**: O agente começa com uma tabela Q vazia ou inicializada com valores arbitrários.
2. **Interação**: A cada iteração, o agente observa o estado atual, seleciona uma ação (com base na estratégia \( \epsilon \)-greedy), executa a ação e observa a recompensa.
3. **Atualização**: O agente usa a equação de Bellman para atualizar o valor Q da ação tomada com base na recompensa recebida.
4. **Repetição**: O processo se repete por várias interações, refinando continuamente a tabela de valores Q até que a política esteja otimizada.

### Exemplo de Implementação do Q-Learning

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

## 4. Tarefa

Vocês estão divididos em dois grupos, cada um responsável por um dos algoritmos principais deste projeto:

- **Grupo 1**: Responsável pelo algoritmo **Q-learning**
- **Grupo 2**: Responsável pelo algoritmo **Minimax**

**Objetivo Geral**: Implementar e ajustar os algoritmos designados para competir entre si no jogo da velha (**Tic-Tac-Toe**). Cada grupo deve refinar seu algoritmo para maximizar o desempenho e vencer o maior número possível de partidas.

### O que deve ser feito?

1. **Implementação e Ajustes**:
   - **Grupo 1**: Implementar o algoritmo de Q-learning e ajustar parâmetros como `α`, `γ` e `ε` para que o agente aprenda a jogar de maneira eficiente. Testem várias simulações para identificar os melhores ajustes que otimizem o desempenho do algoritmo ao longo do tempo. Refine os parâmetros do algoritmo para maximizar o desempenho ao longo do tempo. Ajustem `α` (taxa de aprendizado), `γ` (fator de desconto) e `ε` (exploração vs. explotação). Vocês precisam mostrar como esses ajustes impactam na evolução do aprendizado e quantas partidas foram necessárias para o algoritmo começar a reagir.

   - **Grupo 2**: Implementar o algoritmo Minimax e, se necessário, aplicar **poda alfa-beta** para otimizar a eficiência do algoritmo. Certifiquem-se de que o Minimax jogue de maneira perfeita, explorando todas as possíveis jogadas do jogo. Otimize o algoritmo Minimax para que ele jogue de forma eficiente e perfeita, aplicando poda alfa-beta se necessário para economizar tempo de processamento. Certifiquem-se de que o Minimax esteja pronto para jogar contra o Q-learning.

### Grupo 1: Q-Learning
- Luca
- Pedro
- Cláudio
- Lucas Narita
- Felipe Dutra
- Lucas Fiche
- Mateus
- João

### Grupo 2: Minimax
- Mariana
- Fábio
- Sara
- Arthur
- Felipe Barroso
- Igor
- Eduardo

---

### Formato de Apresentação e Análise

### Métricas de Desempenho:
- Qual foi a quantidade total de partidas jogadas?
- Quantas vitórias obteve cada algoritmo?
- Qual foi o número de empates?
- Qual a taxa de vitória do Q-learning após estabilização?
- A partir de quantas partidas o Q-learning começou a reagir?
- Parâmetros usados para Q-learning: \( \alpha \), \( \gamma \), \( \epsilon \).

### Análise Crítica:
- Qual algoritmo performou melhor e em quais condições?
- Quais ajustes precisaram ser feitos para o algoritmo do grupo vencer?
- Como o Q-learning reagiu ao longo do tempo?
- Quais ajustes fizeram a diferença no desempenho?

**Boa sorte!**