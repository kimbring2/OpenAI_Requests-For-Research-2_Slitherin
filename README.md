# OpenAI_Requests-For-Research-2_Slitherin
Research for improving self-play instability

# Research Introduction
<p>⭐⭐ <strong>Slitherin'.</strong> Implement and solve a multiplayer clone of the classic <a href="https://www.youtube.com/watch?v=wDbTP0B94AM">Snake</a> game (see <a href="https://slither.io">slither.io</a> for inspiration) as a <a href="https://github.com/openai/gym">Gym</a> environment.</p>
<ul>
<li>Environment: have a reasonably large field with multiple snakes; snakes grow when eating randomly-appearing fruit; a snake dies when colliding with another snake, itself, or the wall; and the game ends when all snakes die. Start with two snakes, and scale from there.</li>
<li>Agent: solve the environment using self-play with an RL algorithm of <a href="https://blog.openai.com/competitive-self-play/">your</a> <a href="https://deepmind.com/blog/alphago-zero-learning-scratch/">choice</a>. You'll need to experiment with various approaches to overcome self-play instability (which resembles the instability people see with GANs). For example, try training your current policy against a distribution of past policies. Which approach works best?</li>
<li>Inspect the learned behavior: does the agent learn to competently pursue food and avoid other snakes? Does the agent learn to attack, trap, or gang up against the competing snakes? Tweet us videos of the learned policies!</li>
</ul>

# Reference


https://github.com/bhairavmehta95/slitherin-gym
https://github.com/ingkanit/multi-snake-RL
