import numpy as np

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for value function estimation using eligibility traces.
    
    Args:
        env: OpenAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        lambtha: eligibility trace factor (lambda)
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
        
        # Initialize eligibility traces for this episode
        eligibility_traces = np.zeros_like(V)
        
        # Run episode
        for step in range(max_steps):
            # Get action from policy
            action = policy(state)
            
            # Take step in environment
            next_state, reward, done, *_ = env.step(action)
            
            # Calculate TD error
            # δ = r + γ * V(s') - V(s)
            if done:
                # If terminal state, V(s') = 0
                td_error = reward - V[state]
            else:
                td_error = reward + gamma * V[next_state] - V[state]
            
            # Update eligibility trace for current state
            eligibility_traces[state] += 1
            
            # Update value function and decay eligibility traces
            for s in range(len(V)):
                # Update value function: V(s) = V(s) + α * δ * e(s)
                V[s] = V[s] + alpha * td_error * eligibility_traces[s]
                
                # Decay eligibility trace: e(s) = γ * λ * e(s)
                eligibility_traces[s] = gamma * lambtha * eligibility_traces[s]
            
            # Move to next state
            state = next_state
            
            # Break if episode is done
            if done:
                break
    
    return V
