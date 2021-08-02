import time
import torch



def calc_speeds(poses, avail):
    batches_to_change = (torch.sum(avail, axis=1) > 1).nonzero()[:,0]
    bs = len(poses)
    full_speeds = torch.zeros(bs,2)
    if len(batches_to_change) == 0:
        return full_speeds
    poses_non_zero = poses[batches_to_change]
    avail_non_zero = avail[batches_to_change]
    indexes_last = [(avail_non_zero[i,1:].cumsum(0) == 1).nonzero()[0].item()+1 for i in range(len(batches_to_change))]
#     print("indexes_last", indexes_last)
    last_visible_pose = poses[[ i for i in batches_to_change],torch.tensor(indexes_last)]
    current_pose = poses_non_zero[:,0]
    # delta from last visible pose
    delta_step = torch.tensor(indexes_last).reshape(-1,1)
    speeds = (current_pose - last_visible_pose) / delta_step
    full_speeds = torch.zeros(bs,2)
    full_speeds[batches_to_change] = speeds
    return full_speeds

class LPP():
    
    def predict(self,state,future_horizon=12,velocity=None,goal=None,history=None, avail=None):
        assert state.ndim == 2
        trajects = []
        if velocity is not None:
            assert velocity.ndim == 2
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)

        if goal is not None:
            assert goal.ndim == 2
            velocity = (goal-state)/future_horizon
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)

        if history is not None:
            assert history.ndim == 3
            velocity = calc_speeds(torch.cat((state.unsqueeze(1),history),dim=1), avail)
            # velocity = torch.mean(history[:,0:-1]-history[:,1:],dim=1) #
            # velocity = (history[:,0]-history[:,-1])/len(history[0]) #
            # velocity = history[:,0]-history[:,1] #
            trajects.append(state+velocity)
            for _ in range(future_horizon-1):
                trajects.append(trajects[-1]+velocity)
        
        assert len(trajects)>0
        trajects = torch.stack(trajects)
        return trajects.permute(1,0,2)
                

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

