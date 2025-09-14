import argparse
import collections
import logging
import matplotlib.pyplot as plt
from random import expovariate, sample, seed, choice
from discrete_event_sim import Simulation, Event
from workloads import weibull_generator
class MMN(Simulation):
    def __init__(self, lambd, mu, n, d, max_t,plot_interval,weibull_shape):
        super().__init__()
        self.running = [None] * n
        self.queues: list[collections.deque] = [collections.deque() for _ in range(n)]  # FIFO queues of the system  
        self.arrivals: dict[int, float] = {} 
        self.completions: dict[int, float] = {} 
        self.service_times: dict[int, float] = {} ####
        self.start_times:dict[int, float] = {} ####
        self.lambd = lambd
        self.n = n
        self.d = d
        self.max_t = max_t
        self.mu = mu
        self.times = []
        self.arrival_rate = lambd * n
        self.completion_rate = mu
        self.array_len = [0] * self.n
        self.arrival_gen=weibull_generator(weibull_shape,1/self.arrival_rate)
        self.service_gen=weibull_generator(weibull_shape,1/self.completion_rate)
        self.schedule(self.arrival_gen(), Arrival(0))  # schedule the first arrival
        self.schedule(0, MonitoringQueueSize(plot_interval))

    def schedule_arrival(self, job_id):
        self.schedule(self.arrival_gen(), Arrival(job_id))

    def schedule_completion(self, job_id, queue_index):
        service_time = self.service_times[job_id] ####
        self.start_times[job_id] = self.t ####
        self.schedule(service_time, Completion(job_id, queue_index))

    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queues[i])
    
    ##########################################
    def calculate_workload(self, queue_index):
        if self.running[queue_index] is not None:
            job_id = self.running[queue_index]
            total_service_time = self.service_times[job_id]
            start_time = self.start_times[job_id]
            elapsed_time = self.t - start_time
            remaining_service_time=total_service_time - elapsed_time  #the remaining work of the currently running job
        else:
            remaining_service_time = 0
        # the total service time of jobs waiting in the queue
        queue_workload = sum(self.service_times[job] for job in self.queues[queue_index]) 
        adjusted_workload = queue_workload + remaining_service_time
        return adjusted_workload  #Returns the total remaining workload of the queue

class MonitoringQueueSize(Event):
    def __init__(self, interval=10):
        self.interval = interval
    def process(self, sim: MMN):
        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))
        sim.schedule(self.interval, self)

class Arrival(Event):
    def __init__(self, job_id):
        self.id = job_id
    def process(self, sim: MMN):
        sim.arrivals[self.id] = sim.t
        # empty_queues = [q for q in sim.queues if len(q) == 0 and sim.running[sim.queues.index(q)] is None]
        # if empty_queues:
        #     selected_list = choice(empty_queues)
        # else:
        samples = sample(sim.queues, sim.d)
        selected_list = min(samples, key=lambda lst: sim.calculate_workload(sim.queues.index(lst)))
        queue_index = sim.queues.index(selected_list) 
        sim.service_times[self.id] = sim.service_gen()
        if sim.running[queue_index] is None:
            sim.running[queue_index] = self.id
            sim.schedule_completion(self.id, queue_index)
        else:
            sim.queues[queue_index].append(self.id)
        sim.schedule_arrival(self.id + 1)

class Completion(Event):
    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index
    def process(self, sim: MMN): 
        queue_index = self.queue_index
        assert sim.running[queue_index] == self.job_id
        sim.completions[self.job_id] = sim.t
        if sim.queues[queue_index]:
            new_job = sim.queues[queue_index].popleft()
            sim.running[queue_index] = new_job
            sim.schedule_completion(new_job, queue_index)
        else: 
            sim.running[queue_index] = None
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5,0.6,0.77,0.8,0.85,0.9,0.95,0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, nargs='+', default=[5,10,15,20])
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot_interval", type=float, default=10, help="how often to collect data points for the plot")
    parser.add_argument('--weibull-shape', type=float, default= 0.8, help="Shape parameter for Weibull distribution")
    args = parser.parse_args()
    if args.seed:
        seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(10, 6))

    for j, d_value in enumerate(args.d):
        print(f"d:{d_value}")
        W_values = [] 
        for lambd_value in args.lambd:
            sim = MMN(lambd_value, args.mu, args.n, d_value, args.max_t, args.plot_interval,args.weibull_shape)
            sim.run(args.max_t)
            W = (sum(sim.completions.values()) - sum(sim.arrivals[job_id] for job_id in sim.completions)) / len(sim.completions)
            print(f"Average time spent in the system: {W}")
            W_values.append(W)
        plt.plot(args.lambd, W_values, label=f'mu={args.mu}, max_t={args.max_t}, n={args.n}, d={d_value}', linestyle=line_styles[j % len(line_styles)], color=colors[j % len(colors)])
    plt.xlabel('λ')
    plt.ylabel('W')
    plt.yscale("log")  
    plt.yscale("log")  
    plt.title(f"System Performance: W vs. λ - LWL Model")
    plt.legend()
    plt.grid(True)
    plt.show() 
if __name__ == '__main__':
    main()
