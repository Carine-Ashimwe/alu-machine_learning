import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value function estimation.

    Args:
        env: OpenAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over (default: 5000)
        max_steps: maximum number of steps per episode (default: 100)
        alpha: learning rate (default: 0.1)
        gamma: discount rate (default: 0.99)
    
    Returns:
        V: updated value estimate
    """
    
    for episode in range(episodes):
        # Reset environment and get initial state
        state = env.reset()
        # Handle case where reset() returns tuple (observation, info)
        if isinstance(state, tuple):
            state = state[0]
        
        # Store episode trajectory: (state, reward) pairs
        episode_trajectory = []
        
        # Generate episode following the policy
        for step in range(max_steps):
            # Get action from policy
            action = policy(state)
            
            # Take step in environment
            next_state, reward, done, *_ = env.step(action)
            
            # Store state and reward
            episode_trajectory.append((state, reward))
            
            # Update state
            state = next_state
            
            # Break if episode is done
            if done:
                break
        
        # Calculate returns and update value function
        G = 0  # Return (cumulative discounted reward)
        
        # Process trajectory backwards to calculate returns
        for t in reversed(range(len(episode_trajectory))):
            state_t, reward_t = episode_trajectory[t]
            
            # Update return
            G = gamma * G + reward_t
            
            # Update value function using incremental average
            # V(s) = V(s) + alpha * (G - V(s))
            V[state_t] = V[state_t] + alpha * (G - V[state_t])
    
    return V
