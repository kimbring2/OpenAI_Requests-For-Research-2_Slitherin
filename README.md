# OpenAI_Requests-For-Research-2_Slitherin
![Slither.io Game Play](https://github.com/kimbring2/OpenAI_Requests-For-Research-2_Slitherin/blob/master/image/slitherio.gif)

Research for improving self-play instability

# Research Introduction
<p>⭐⭐ <strong>Slitherin'.</strong> Implement and solve a multiplayer clone of the classic <a href="https://www.youtube.com/watch?v=wDbTP0B94AM">Snake</a> game (see <a href="https://slither.io">slither.io</a> for inspiration) as a <a href="https://github.com/openai/gym">Gym</a> environment.</p>
<ul>
<li>Environment: have a reasonably large field with multiple snakes; snakes grow when eating randomly-appearing fruit; a snake dies when colliding with another snake, itself, or the wall; and the game ends when all snakes die. Start with two snakes, and scale from there.</li>
<li>Agent: solve the environment using self-play with an RL algorithm of <a href="https://blog.openai.com/competitive-self-play/">your</a> <a href="https://deepmind.com/blog/alphago-zero-learning-scratch/">choice</a>. You'll need to experiment with various approaches to overcome self-play instability (which resembles the instability people see with GANs). For example, try training your current policy against a distribution of past policies. Which approach works best?</li>
<li>Inspect the learned behavior: does the agent learn to competently pursue food and avoid other snakes? Does the agent learn to attack, trap, or gang up against the competing snakes? Tweet us videos of the learned policies!</li>
</ul>

# Single Agent Warm-Up
Before playing multiple snake games, I first solved a single snake game on dqn, but was able to tweet and generate decent results.

https://twitter.com/kimbring2/status/963671596610326528

# Reference
In this research, there was no given code for environment and algorithm, so I need to create and find a multi-agent snake game and self-play algorithm. Many excellent other researchers have published environment and algorithm on github, so I can use them as a reference to conduct research presented by OpenAI.

https://github.com/bhairavmehta95/slitherin-gym - Multi Snake Env  
https://github.com/ingkanit/multi-snake-RL - PPO2 for Multi Snake playing 
https://web.stanford.edu/~surag/posts/alphazero.html - Self Play Algorithm

Once again, thanks for these researchers.

# Reference
