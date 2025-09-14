# MMN Queue Simulation

This project is part of the **Distributed Computing** course and implements and analyzes different queueing models through **Discrete Event Simulation (DES)** with a focus on **M/M/N systems** and their extensions.
---

## 📌 Project Overview
The goal is to move beyond simple M/M/1 models and study more realistic distributed systems.  
The implementation covers:
- **M/M/1 → M/M/N extension**  
- **Supermarket Model optimization** for queue selection  
- **Weibull distribution** for service times (generalization of exponential distribution)  
- **Job priority handling** (priority queues replacing FIFO)  
- **Least Work Left (LWL) model** compared with the supermarket approach  

---

## 📂 Project Structure
- `discrete_event_sim.py` → Core simulation engine  
- `queue_sim.py` → M/M/1 and M/M/N implementation  
- `queue_sim_weibull.py` → Weibull distribution extension  
- `workloads.py` → Arrival & service time generators  
- `sir.py` & `workloads.py` → Additional experiments  

