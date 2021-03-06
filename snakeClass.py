import pygame
from random import randint
from DQN import AGENT, DQNAgent, DDQNAgent, ExperienceReplay
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import namedtuple
import timeit


# Set options to activate or deactivate the game view, and its speed
display_option = True
speed = 0                       # The speed we run the game, zero is fastest

pygame.font.init()


class Game:
    """
        Stuff related to the game eg. size of window images
        Player and Food class exist within this class
    """

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
    """
        Stuff related to where the snake are and function to move it etc
        :functions update_position, do_move, display_player
    """

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
        """
        Update the position of the snake
        :param x: the x-pos of the snake head
        :param y: the y-pos of the snake head
        :return:  None
        """
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food):
        """
        "Make a move": check if we eat, update the position, check if we crash
        :param move: the action
        :param x:    the x-pos of the snake head
        :param y:    the y-pos of the snake head
        :param game: the game class
        :param food: the food class
        :return:     None
        """
        move_array = [self.x_change, self.y_change]

        if self.eaten:

            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
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
        """

        :param x:       the x-pos of the snake head
        :param y:       the y-pos of the snake head
        :param food:    food class
        :param game:    game class
        :return:        None
        """
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash is False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):

    """
        Stuff related to the food i.e. its position and image
        :functions food_coord, display_food
    """

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food3.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    """
    Checks if we are eating the food in this move.
    This functions runs every time we do a move

    If we have eaten the player.eaten it set to True, the score increases with one and
    a new food position is generated
    :param player: player class
    :param food:   food class
    :param game:   game class
    :return: None
    """
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    """
    Update the high score if the current value is higher
    :param score: Current score
    :param record: high score
    :return: high score
    """

    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    """
    Display the game with the current score and high score
    :param game:    game class
    :param score:   the current score
    :param record:  high score
    :return: None
    """

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
    """
    Update the screen
    :return: None
    """
    pygame.display.update()


def initialize_game(player, game, food, agent, replay_buffer):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    replay_buffer.add(Transition(s=state_init1, a=np.argmax(action), r=reward1, next_s=state_init2, t=game.crash))


def plot_seaborn(array_counter, array_score, plot_name):
    """
    Plot a graph over all played games
    :param array_counter:   A vector with increasing numbers [1, 2, 3 ... N]
    :param array_score:     The scores for all played games
    :param plot_name:       Name of plot
    :return:                None
    """
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1, line_kws={'color':'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.savefig(plot_name, dpi=400)

    plt.show()


def evaluate_network(agent, name_weights):
    # Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
    # for training
    replay_buffer = ExperienceReplay(state_size=12)

    # Tuple subclass named Transition
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    pygame.init()  # Initialize all imported pygame modules

    # Initialize to zero
    counter_games = 0  # The number of games played
    score_plot = []  # The scores over time, used to plot
    counter_plot = []  # The game number over time [1 2 3 ... N], used to plot
    record = 0  # Highest score
    average_score = 0

    eps = 0
    number_episodes = 100  # Number of episodes (games) we train on

    # Load the weighs of the model from the computer
    agent.online_model.load_weights(name_weights)
    if agent.DDQN is True:
        agent.online_model.load_weights(name_weights)
    else:
        agent.target_model.load_weights(name_weights)

    # Run "number_episodes" games
    while counter_games < number_episodes:

        steps = 0

        # Initialize classes
        game = Game(440, 440)  # eg set game.crash to False
        player1 = game.player
        food1 = game.food
        state = agent.get_state(game, player1, food1)

        # Perform first move
        initialize_game(player1, game, food1, agent, replay_buffer)

        # Display window if display_option is True
        if display_option:
            display(player1, food1, game, record)

        # While the snake as not died
        while not (game.crash or steps > 950):
            steps += 1
            # Get the q-values w.r.t. the states
            q_values = agent.get_q_values(state.reshape((1, 12)))

            # Evaluate new policy w.r.t q-values and epsilon (epsilon-greedy policy)
            policy = agent.eps_greedy_policy(q_values[0], eps)

            # Choose an action w.r.t. the policy and converts it to one-hot format (eg 2 to [0, 0, 1])
            action = np.random.choice(3, p=policy)
            action_hot = to_categorical(action, num_classes=3)#[0] # CHANGE HERE

            # Let the player do the chosen action
            player1.do_move(action_hot, player1.x, player1.y, game, food1)

            # Update the state variables
            new_state = agent.get_state(game, player1, food1)

            # Give reward, 0 - alive, 10 - eaten, -10 - died
            reward = agent.set_reward(player1, game.crash)

            # Store data to the Tuple subclass Transition and then use the function add in replay_buffer to add
            # Transition to replay_buffer.__buffer
            replay_buffer.add(Transition(s=state, a=action, r=reward, next_s=new_state, t=game.crash))

            # Assign the "old" state as the new
            state = new_state

            # Update record (high score)
            record = get_record(game.score, record)

            # Update display if display_option == True
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        counter_games += 1

        print('Game %i      Score: %i      Epsilon: %.4f    Highest: %i' % (counter_games, game.score, round(eps, 5), record))

        score_plot.append(game.score)
        counter_plot.append(counter_games)

        average_score += game.score

    average_score = round(average_score/number_episodes)
    print('The highest score: %i, the average score: %i' % (record, average_score))

    return agent, counter_plot, score_plot


def run(agent, name_weights, name_plot):
    # Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
    # for training
    replay_buffer = ExperienceReplay(state_size=12)

    # Tuple subclass named Transition
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    pygame.init()                   # Initialize all imported pygame modules

    # Initialize to zero
    counter_games = 0               # The number of games played
    score_plot = []                 # The scores over time, used to plot
    counter_plot =[]                # The game number over time [1 2 3 ... N], used to plot
    record = 0                      # Highest score

    # Design parameters
    eps = .8                        # "start epsilon"
    eps_end = .01                   # "Final epsilon"
    eps_decay = .0005               # Parameter how slowly it decays

    batch_size = 128                # The size of the batch
    gamma = 0.99                    # How much the future rewards matter
    number_episodes = 2000          # Number of episodes (games) we train on

    tau_lim = 5                     # Number of iteration between target updates

    # Run "number_episodes" games
    while counter_games < number_episodes:

        # Initialize classes
        game = Game(440, 440)       # eg set game.crash to False
        player1 = game.player
        food1 = game.food
        state = agent.get_state(game, player1, food1)

        # Perform first move
        initialize_game(player1, game, food1, agent, replay_buffer)

        # Display window if display_option is True
        if display_option:
            display(player1, food1, game, record)

        # While the snake as not died
        while not game.crash:
            # Get the q-values w.r.t. the states
            q_values = agent.get_q_values(state.reshape((1, 12)))

            # Evaluate new policy w.r.t q-values and epsilon (epsilon-greedy policy)
            policy = agent.eps_greedy_policy(q_values[0], eps)

            # Choose an action w.r.t. the policy and converts it to one-hot format (eg 2 to [0, 0, 1])
            action = np.random.choice(3, p=policy)
            action_hot = to_categorical(action, num_classes=3) # [0] # CHANGE HERE

            # Let the player do the chosen action
            player1.do_move(action_hot, player1.x, player1.y, game, food1)

            # Update the state variables
            new_state = agent.get_state(game, player1, food1)

            # Give reward, 0 - alive, 10 - eaten, -10 - died
            reward = agent.set_reward(player1, game.crash)

            # Store data to the Tuple subclass Transition and then use the function add in replay_buffer to add
            # Transition to replay_buffer.__buffer
            replay_buffer.add(Transition(s=state, a=action, r=reward, next_s=new_state, t=game.crash))

            # Assign the "old" state as the new
            state = new_state

            # Update record (high score)
            record = get_record(game.score, record)

            # If we have done more than 1000 steps in total (over multiple games)
            # then !perform one training step!
            if replay_buffer.buffer_length > 1000:
                # Get batch_size many random samples of state, action, reward, t, next_state from previous
                # experience
                s, a, r, s_, t = replay_buffer.sample_minibatch(batch_size)

                if agent.DDQN is True:

                    # Get q_values for both online and offline network
                    q_1, q_2 = agent.get_q_values_for_both_models(np.squeeze(s_))

                    # Calculate the td-targets
                    td_target = agent.calculate_td_targets(q_1, q_2, r, t, gamma)

                    # Perform one update step on the model and 50% chance to switch between online and offline network
                    # Almost the same thing as model.fit in Keras
                    agent.update(s, td_target, a)
                    
                    # Update q-target if agent.targetIt is greater than 10
                    agent.target_update(tau_lim)

                else:

                    # Get q_values for network
                    q_values = agent.get_q_values(np.squeeze(s_))

                    # Calculate the td-targets
                    td_target = agent.calculate_td_targets(q_values, r, t, gamma) #agent.Q_target

                    # Perform one update step on the model and 50% chance to switch between online and offline network
                    # Almost the same thing as model.fit in Keras
                    agent.update(s, td_target, a)

                    # Update q-target if agent.targetIt is greater than 10
                    agent.target_update(tau_lim)

            # Update display if display_option == True
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)
                
        eps = max(eps - eps_decay, eps_end)  # decrease epsilon
        if replay_buffer.buffer_length < 1000:
            s, a, r, s_, t = replay_buffer.sample_minibatch(replay_buffer.buffer_length) # sample a minibatch of transitions
            if agent.DDQN is True:
                # Get q_values for both online and offline network
                q_1, q_2 = agent.get_q_values_for_both_models(np.squeeze(s_))
                # Calculate the td-targets
                td_target = agent.calculate_td_targets(q_1, q_2, r, t, gamma)

            else:
                # Get q_values for network
                q_values = agent.get_q_values(np.squeeze(s_))

                # Calculate the td-targets
                td_target = agent.calculate_td_targets(q_values, r, t, gamma)

                # Update q-target if agent.targetIt is greater than 10
                agent.target_update(tau_lim)

            agent.update(s, td_target, a)
        
        # agent.replay_new(agent.memory)
        counter_games += 1
        print('Game %i      Score: %i      Epsilon: %.4f    Highest: %i' % (counter_games,game.score ,round(eps, 5), record))
        score_plot.append(game.score)
        counter_plot.append(counter_games)

    # Save the weighs of the model to the computer
    agent.online_model.save_weights(name_weights)

    # Plot the score to the number of game
    plot_seaborn(counter_plot, score_plot, name_plot)

    return agent, counter_plot, score_plot

# ################ CHANGE THESE!!!!! ##############
DDQN = False                # Do we run DDQN, else we run DQN
target = False              # Do run target DQN or not
# #################################################

# Start timer
start = timeit.default_timer()
if DDQN is True:

    # Name of weights + Initialize DDQN class + run training + evaluate
    name_of_weights_DDQN = 'weights_DDQN.h5'
    name_of_plot = 'train_plot_DDQN'
    pre_agent = DDQNAgent()
    aft_agent = run(pre_agent, name_of_weights_DDQN, name_of_plot)
    evaluate_network(aft_agent, name_of_weights_DDQN)
else:
    # Name of weights + Initialize DQN target class + run training + evaluatn
    if target:
        name_of_weights_DQN = 'weights_DQN_target.h5'
        name_of_plot = 'train_plot_DQN_target'
    else:
        name_of_weights_DQN = 'weights_DQN_not_target_2.h5'
        name_of_plot = 'train_plot_DQN_not_target_2'

    pre_agent = DQNAgent(target)
    aft_agent = run(pre_agent, name_of_weights_DQN, name_of_plot)
    evaluate_network(aft_agent, name_of_weights_DQN)

stop = timeit.default_timer()
print('Total run time: %i min' % ((stop - start)/60))



