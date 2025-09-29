# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.
## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.
## VALUE ITERATION ALGORITHM
Step 1: Set the value of each state to 0 (initial guess).
Step 2: Look at all the actions you can take from that state (like moving up, down, left, or right).
Step 3: Calculate the expected value of each action (i.e., how good that action is based on its possible results).
Step 4: Pick the action that gives the highest value and update the value of the state with that number.
Step 5: Keep updating the values for all states until the difference between the old and new values is very small.
Step 6: Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state.
## VALUE ITERATION FUNCTION
### Name: JEEVANESH S
### Register Number: 212222243002

```PYTHON
def value_iteration(P, gamma=1.0, theta=1e-10):
  V = np.zeros(len(P), dtype=np.float64)
  while True:
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
      break

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    
    return V, pi
```
## OUTPUT:
<img width="787" height="130" alt="image" src="https://github.com/user-attachments/assets/0a60a025-34ee-44d7-9a62-a26940d505a2" />
<img width="687" height="26" alt="image" src="https://github.com/user-attachments/assets/6c222e66-1ef2-495b-bfa3-4f3af714ad88" />
<img width="658" height="103" alt="image" src="https://github.com/user-attachments/assets/b9e60e94-d364-4f04-81b5-f6f6e78ea321" />

## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.

