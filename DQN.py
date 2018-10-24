from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
from collections import deque
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input
from keras.optimizers import RMSprop
from keras import backend as K
from keras.initializers import RandomUniform
from keras.models import Model


class ExperienceReplay:

    def __init__(self, buffer_size=1e+6, state_size=4):
        self.__buffer = deque(maxlen=int(buffer_size))
        self._state_size = state_size

    @property
    def buffer_length(self):
        return len(self.__buffer)

    def add(self, transition):
        '''
        Adds a transition <s, a, r, s', t > to the replay buffer
        :param transition:
        :return:
        '''
        self.__buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        '''
        :param batch_size:
        :return:
        '''
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self._state_size])
        action_batch = np.zeros([batch_size, 1])
        reward_batch = np.zeros([batch_size, 1])
        t_batch = np.zeros([batch_size, 1])
        next_state_batch = np.zeros([batch_size, self._state_size])
        for i, index in zip(range(batch_size), ids):
            state_batch[i] = self.__buffer[index].s
            action_batch[i] = self.__buffer[index].a
            reward_batch[i] = self.__buffer[index].r
            t_batch[i] = self.__buffer[index].t
            next_state_batch[i] = self.__buffer[index].next_s

        return state_batch, action_batch, reward_batch, next_state_batch, t_batch

class DDQNAgent(object):
    
    def __init__(self):
        self.reward = 0
        self.dataframe = pd.DataFrame()
        self._lr = 0.001
        self._state_dim = 12
        self._action_dim = 3
        # define the two deep Q-networks
        self._online_model = self.__build_model()
        self._offline_model = self.__build_model()
        # define ops for updating the networks
        self._update = self.__mse() 

    def get_state(self, game, player, food):

        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left


            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y,  # food down
            game.score # Score(length of snake)
            
            ]

        for i in range(len(state)-1):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def __build_model(self):
        '''
        Define all the layers in the network
        :return: Keras model
        '''
        x = Input(shape=(self._state_dim,))
        fc = Dense(120, activation='relu', kernel_initializer='VarianceScaling', input_dim=self._state_dim)(x)
        fc = Dense(120, activation='relu', kernel_initializer='VarianceScaling')(fc)
        fc = Dense(60, activation='relu', kernel_initializer='VarianceScaling')(fc)
        Q_initializer = RandomUniform(minval=-1e-6, maxval=1e-6, seed=None)
        q_value = Dense(self._action_dim, activation='softmax',  kernel_initializer=Q_initializer)(fc)
        model = Model(inputs=x, outputs=q_value)
        
        return model

    def __mse(self):
        '''
        Mean squared error loss
        :return: Keras function
        '''
        q_values = self._online_model.output
        # trace of taken actions
        target = K.placeholder(shape=(None, ), name='target_value')
        a_1_hot = K.placeholder(shape=(None, self._action_dim), name='chosen_actions')

        q_value = K.sum(q_values * a_1_hot, axis=1)
        squared_error = K.square(target - q_value)
        mse = K.mean(squared_error)
        optimizer = RMSprop(lr=self._lr)
        updates = optimizer.get_updates(loss=mse, params=self._online_model.trainable_weights)

        return K.function(inputs=[self._online_model.input, target, a_1_hot], outputs=[], updates=updates)

    def get_q_values_for_both_models(self, states):
        '''
        Calcuates Q-values for both models
        :param states: set of states
        :return: Q-values for online network, Q-values for offline network
        '''
        return self._online_model.predict(states), self._offline_model.predict(states)

    def get_q_values(self, state):
        '''
        Predict all Q-values for the current state
        :param state:
        :return:
        '''
        return self._online_model.predict(state)

    def update(self, states, td_target, actions):
        '''
        Performes one update step on the model and switches between online and offline network
        :param states: batch of states
        :param td_target: batch of temporal difference targets
        :param actions: batch of actions
        :return:
        '''
        actions_one_hot = to_categorical(np.squeeze(actions), self._action_dim)
        self._update([states, np.squeeze(td_target), actions_one_hot])
        if np.random.uniform() > .5:
            self.__switch_weights()

    def __switch_weights(self):
        '''
        Switches between online and offline networks
        '''
        offline_params = self._offline_model.get_weights()
        online_params = self._online_model.get_weights()
        self._online_model.set_weights(offline_params)
        self._offline_model.set_weights(online_params)
