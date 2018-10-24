import pygame
from random import randint
from DQN import DDQNAgent, ExperienceReplay
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.utils.np_utils import to_categorical as one_hot
from collections import namedtuple

# Set options to activate or deactivate the game view, and its speed
display_option = True
speed = 0
pygame.font.init()


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food,agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:

            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move ,[1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move,[0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move,[0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width-40 or self.y < 20 or self.y > game.game_height-40 or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent, replay_buffer):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    replay_buffer.add(Transition(s=state_init1, a=np.argmax(action), r=reward1, next_s=state_init2, t=game.crash))
    #agent.remember(state_init1, action, reward1, state_init2, game.crash)
    #agent.replay_new(agent.memory)

def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()
def eps_greedy_policy(q_values, eps):
    '''
    Creates an epsilon-greedy policy
    :param q_values: set of Q-values of shape (num actions,)
    :param eps: probability of taking a uniform random action 
    :return: policy of shape (num actions,)
    '''
    # YOUR CODE HERE
    (num_actions,) = q_values.shape
    rand = np.random.uniform()
    if rand < eps:
        probability = 1/num_actions
        return np.ones(num_actions)*probability
    
    policy = np.zeros(num_actions)
    action = q_values.argmax()
    policy[action] = 1
    
    return policy

def calculate_td_targets(q1_batch, q2_batch, r_batch, t_batch, gamma=.99):
    '''
    Calculates the TD-target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network, shape (N, num actions)
    : param r_batch: Batch of rewards, shape (N, 1)
    : param t_batch: Batch of booleans indicating if state, s' is terminal, shape (N, 1)
    : return: TD-target, shape (N, 1)
    '''

    # YOUR CODE HERE
    N = len(q1_batch)
    Y = r_batch.copy()
    for i in range(N):
        if not int(t_batch[i]):
            a = np.argmax(q1_batch[i])
            Y[i] += gamma*q2_batch[i,a]
    
    return Y

def run():
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    pygame.init()
    agent = DDQNAgent()
    score_plot = []
    counter_plot = []
    counter_games = 0
    record = 0
    eps = .5
    eps_end = .05 
    eps_decay = .01
    batch_size = 128
    gamma = 0.9
    while counter_games < 3000:
        # Initialize classes
        game = Game(440, 440)
        ep_reward = 0
        player1 = game.player
        food1 = game.food
        q_buffer = []
        state = agent.get_state(game, player1, food1)
        #state.resize(12,)
        
        #state[11] = game.score
        print(state.shape)
        
        

        # Perform first move
        initialize_game(player1, game, food1, agent, replay_buffer)
        if display_option:
            display(player1, food1, game, record)

        while not game.crash:
            q_values = agent.get_q_values(state.reshape((1,12)))
            q_buffer.append(q_values[0])
            policy = eps_greedy_policy(q_values[0], eps) 
            action = np.random.choice(3, p=policy) # sample action from epsilon-greedy policy
            final_move = to_categorical(action, num_classes=3)[0]
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            new_state = agent.get_state(game, player1, food1)
            #new_state.resize(12,)
            #new_state[11] = game.score
            reward = agent.set_reward(player1, game.crash)
                 
            # store data to replay buffer
            replay_buffer.add(Transition(s=state, a=action, r=reward, next_s=new_state, t=game.crash))
            state = new_state
            record = get_record(game.score, record)
            # if buffer contains more than 1000 samples, perform one training step
            if replay_buffer.buffer_length > 1000:
                s, a, r, s_, t = replay_buffer.sample_minibatch(batch_size) # sample a minibatch of transitions
                q_1, q_2 = agent.get_q_values_for_both_models(np.squeeze(s_))
                td_target = calculate_td_targets(q_1, q_2, r, t, gamma)
                agent.update(s, td_target, a)
                
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)
                
        eps = max(eps - eps_decay, eps_end) # decrease epsilon        
        if replay_buffer.buffer_length > 1000:
            s, a, r, s_, t = replay_buffer.sample_minibatch(1000) # sample a minibatch of transitions
            q_1, q_2 = agent.get_q_values_for_both_models(np.squeeze(s_))
            td_target = calculate_td_targets(q_1, q_2, r, t, gamma)
            agent.update(s, td_target, a)
        else:
            s, a, r, s_, t = replay_buffer.sample_minibatch(replay_buffer.buffer_length) # sample a minibatch of transitions
            q_1, q_2 = agent.get_q_values_for_both_models(np.squeeze(s_))
            td_target = calculate_td_targets(q_1, q_2, r, t, gamma)
            agent.update(s, td_target, a)
        
        #agent.replay_new(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score, '      Epsilon:', eps)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    agent.model.save_weights('weights.hdf5')
    plot_seaborn(counter_plot, score_plot)


# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored 
# for training
replay_buffer = ExperienceReplay(state_size=12)
run()