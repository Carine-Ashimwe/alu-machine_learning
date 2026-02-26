import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, 
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) algorithm for Q-learning with eligibility traces.
    
    Args:
        env: OpenAI environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor (lambda)
        episodes: total number of episodes to train over (default: 5000)
        max_steps: maximum number of steps per episode (default: 100)
        alpha: learning rate (default: 0.1)
        gamma: discount rate (default: 0.99)
        epsilon: initial threshold for epsilon greedy (default: 1)
        min_epsilon: minimum value that epsilon should decay to (default: 0.1)
        epsilon_decay: decay rate for updating epsilon between episodes (default: 0.05)
    
    Returns:
        Q: updated Q table
    """
    
    def epsilon_greedy_action(state, Q, epsilon):
        """
        Choose action using epsilon-greedy policy.
        """
        if np.random.random() < epsilon:
            # Explore: choose random action
            return np.random.randint(Q.shape[1])
        else:
            # Exploit: choose action with highest Q-value
            return np.argmax(Q[state])
    
    for episode in range(episodes):
        # Reset environment and get initial state
        state = env.reset()
        # Handle case where reset() returns tuple (observation, info)
        if isinstance(state, tuple):
            state = state[0]
        
        # Initialize eligibility traces for this episode
        eligibility_traces = np.zeros_like(Q)
        
        # Choose initial action using epsilon-greedy policy
        action = epsilon_greedy_action(state, Q, epsilon)
        
        # Run episode
        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, done, *_ = env.step(action)
            
            # Choose next action using epsilon-greedy policy
            if not done:
                next_action = epsilon_greedy_action(next_state, Q, epsilon)
            else:
                next_action = None
            
            # Calculate TD error
            if done:
                # Terminal state: Q(s',a') = 0
                td_error = reward - Q[state, action]
            else:
                # SARSA update: δ = r + γ * Q(s',a') - Q(s,a)
                td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]
            
            # Update eligibility trace for current state-action pair
            eligibility_traces[state, action] += 1
            
            # Update Q-values and decay eligibility traces for all state-action pairs
            for s in range(Q.shape[0]):
                for a in range(Q.shape[1]):
                    # Update Q-value: Q(s,a) = Q(s,a) + α * δ * e(s,a)
                    Q[s, a] = Q[s, a] + alpha * td_error * eligibility_traces[s, a]
                    
                    # Decay eligibility trace: e(s,a) = γ * λ * e(s,a)
                    eligibility_traces[s, a] = gamma * lambtha * eligibility_traces[s, a]
            
            # Move to next state and action
            state = next_state
            action = next_action
            
            # Break if episode is done
            if done:
                break
        
        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    
    return Q
