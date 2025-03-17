import gym
import gym.spaces
from gym.utils.play import play
import sys

game = sys.argv[1]
play(gym.make(game), fps=60, zoom=3.)
