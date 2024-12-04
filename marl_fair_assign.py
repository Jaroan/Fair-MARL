import numpy as np
import pyomo.environ as pe
import scipy.spatial.distance as dist

def build_base_model(costs):
    assert np.ndim(costs) == 2
    n, nj = costs.shape
    m = pe.ConcreteModel()
    m.ns = pe.RangeSet(0,n-1)
    m.ms = pe.RangeSet(0,nj-1)
    m.x = pe.Var(m.ns, m.ms, domain=pe.Binary)
    m.coverage = pe.Constraint(m.ms, rule=lambda m,j: sum(m.x[i,j] for i in m.ns) == 1)   # eacg task must be performed by exactly one agent
    m.assignment = pe.Constraint(m.ns, rule=lambda m,i: sum(m.x[i,j] for j in m.ms) == 1) # each agent must perform exactly one task
    return m

def solve_fair_assignment(costs):
    '''Solves lexifair assignment.

    costs - matrix of costs where entry [i,j] is the cost for agent i to perform task j
    '''
    n, nj = costs.shape
    cost_helper = np.copy(costs)
    m = build_base_model(costs)
    m.z = pe.Var()
    m.aux = pe.Constraint(m.ns, m.ms, rule=lambda m,i,j: cost_helper[i,j]*m.x[i,j] <= m.z)  # maximum cost of each agent
    m.assigned = pe.ConstraintList()
    m.obj = pe.Objective(expr=m.z, sense=pe.minimize)

    solver = pe.SolverFactory('gurobi_persistent')
    solver.set_instance(m)
    objs = []

    for iter in range(n):
        solver.solve(options={'TimeLimit': 60}, save_results=True)
        x = np.array(pe.value(m.x[:,:])).reshape((n,nj)).astype(int)
        obj = pe.value(m.obj)

        objs.append(obj)
        r,c = np.unravel_index(np.argmin(np.abs(costs - obj)), (n,nj)) # find the agent/task that incurred max cost

        # update costs + constraints
        cost_helper[r,c] = 0
        for constr in m.aux.values():
            solver.remove_constraint(constr)
        m.del_component(m.aux)
        m.del_component(m.aux_index)
        m.aux = pe.Constraint(m.ns, m.ms, rule=lambda m,i,j: cost_helper[i,j]*m.x[i,j] <= m.z)
        for constr in m.aux.values():
            solver.add_constraint(constr)
        for j in m.ms:
            m.assigned.add(expr=m.x[r,j] == x[r,j])
            solver.add_constraint(m.assigned[iter*nj + j + 1])

    objs = np.sort(np.sum(costs*x, axis=1))[::-1]
    return x, objs

if __name__=='__main__':
    # n = 10
    # rng = np.random.default_rng(seed=667143)
    # goals = rng.random((n,2))
    # agents = rng.random((n,2))

    goals = np.array([[0.,-0.5],[0.45,-0.5],[0.9,-0.5]])
    agents = np.array([[-0.9,-0.9],[-0.9,0.],[-0.9,0.9]])
    print(goals.shape)
    # goals = # FILL ME (n-by-2 np.ndarray)
    # agents = # FILL ME (n-by-2 np.ndarray)

    costs = dist.cdist(agents, goals)
    x, objs = solve_fair_assignment(costs)
    # print(x)
    # print(objs)
    