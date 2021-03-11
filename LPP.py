import time
import torch

class LPP():
    
    def predict(self,state,future_horizon=12,velocity=None,goal=None,history=None):
        assert state.ndim == 2
        trajects = []
        if velocity is not None:
            assert velocity.ndim == 2
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)

        if goal is not None:
            assert goal.ndim == 2
            velocity = goal-state
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)

        if history is not None:
            assert history.ndim == 3
            velocity = state-history[:,0] # may be -1
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)
        
        assert len(trajects)>0
        trajects = torch.stack(trajects)
        return trajects
                
        
        
        assert True


if __name__ == '__main__':
    future_horizon = 12
    bs = 1000
    dev = 'cpu'
    # if torch.cuda.is_available():
    #     dev = torch.device(0)
    lpp = LPP()
    agent_state = torch.rand((bs, 2),device =dev)
    agent_vel = torch.rand((bs, 2),device =dev)
    agent_goal = torch.rand((bs, 2),device =dev)
    agent_history = torch.rand((bs,8, 2),device =dev)

    # by velocy
    start = time.time()
    trajects = lpp.predict(agent_state, velocity=agent_vel,future_horizon=12)
    print("working time "+str(time.time()-start))
    assert trajects.ndim==3
    print("by velocy ok")
    # by goal
    start = time.time()
    trajects = lpp.predict(agent_state, goal=agent_goal,future_horizon=12)
    print("working time "+str(time.time()-start))
    assert trajects.ndim==3
    print("by goal ok")
    # by history
    start = time.time()
    trajects = lpp.predict(agent_state, history=agent_history,future_horizon=12)
    print("working time "+str(time.time()-start))
    assert trajects.ndim==3
    print("by history ok")

    exit()

