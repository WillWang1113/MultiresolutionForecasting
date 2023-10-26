import pickle
import numpy as np

import cvxpy as cp
from src.utils import setup_seed
from copy import deepcopy

setup_seed(9)

Cap = 20000
SOC_ini = 0.4
eta_c = 0.95
eta_d = 0.9
N_g = 4
degrade_price = 800
P_battery = 400

reserve = 0

a = np.array([0.12, 0.17, 0.15, 0.19])
b = np.array([14.8, 16.57, 15.55, 16.21])
c = np.array([89, 83, 100, 70])
Pg_min = np.array([28, 20, 30, 20])
Pg_max = np.array([200, 290, 190, 260])
RU = np.array([40, 30, 30, 50])
RD = np.array([40, 30, 30, 50])


def daed(load, delta_t):
    N_t = len(load)
    P = cp.Variable((N_g, N_t), name='P')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')
    phi = cp.Variable(N_t, name='phi', boolean=True)

    P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
        P[:, t] >= Pg_min for t in range(N_t)
    ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)
         ] + [P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)]
    SOC_constraints = [
        Cap * SOC[0] == (Cap * SOC_ini +
                         (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
    ] + [
        Cap * SOC[t] == (Cap * SOC[t - 1] +
                         (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
        for t in range(1, N_t)
    ] + [
        SOC >= 0.05, SOC <= 0.95, Pc >= 0, Pd >= 0, Pc <= (P_battery * phi), Pd <=
        (P_battery * (1 - phi))
    ]
    balance_constraints = [((sum(P[:, t])) + Pd[t] == (load[t] + Pc[t]))
                           for t in range(N_t)]
    constraints = P_constraints + balance_constraints + SOC_constraints
    # print(constraints)
    objective = cp.Minimize(
        sum((a @ P**2 + b @ P + c.sum())) + 
        degrade_price * sum(((Pc * eta_c + Pd / eta_d))))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    da_opt_P = P.value
    print(sum((a @ da_opt_P**2 + b @ da_opt_P + c.sum())))

    return result, da_opt_P

def ipb(load, da_opt_P, delta_t):
    N_t = len(load)

    P = cp.Variable((N_g, N_t), name='P')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')
    phi = cp.Variable(N_t, name='phi', boolean=True)

    P_constraints = [P == da_opt_P]
    SOC_constraints = [
        Cap * SOC[0] == (Cap * SOC_ini +
                         (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
    ] + [
        Cap * SOC[t] == (Cap * SOC[t - 1] +
                         (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
        for t in range(1, N_t)
    ] + [
        SOC >= 0.05, SOC <= 0.95, Pc >= 0, Pd >= 0, Pc <=
        (P_battery * phi), Pd <= (P_battery * (1 - phi))
    ]
    balance_constraints = [((sum(P[:, t]) + Pd[t]) == (load[t] + Pc[t]))
                           for t in range(N_t)]
    constraints = P_constraints + SOC_constraints + balance_constraints
    # print(constraints)
    objective = cp.Minimize(
        sum((a @ P**2 + b @ P + c.sum())) + 
        sum(((Pc * eta_c * degrade_price  + Pd / eta_d * degrade_price))))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"


    return result, (P.value, SOC.value, Pc.value, Pd.value)

if __name__ == "__main__":
    dataset = "mfred"

    with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
        proposed_preds = pickle.load(handle)
    with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
        bench_raw_preds = pickle.load(handle)
    with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
        bench_bu_preds = pickle.load(handle)
    with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
        bench_opt_preds = pickle.load(handle)

    all_opt_results = []
    all_ipb_schedule = []
    df_all_opt_results = []
    for i in range(len(proposed_preds)):
        # for i in range(1):
        print(i, "/", len(proposed_preds) - 1)
        proposed_model = {"Proposed": proposed_preds[i]}
        all_preds = {
            **proposed_model,
            **bench_raw_preds[i],
            **bench_bu_preds[i],
            **bench_opt_preds[i]
        }
        opt_result = {name: {} for name in sorted(all_preds.keys())}
        ipb_result = {name: {} for name in sorted(all_preds.keys())}

        # loop for models
        for name in sorted(all_preds.keys()):
            print(name)
            if name == "Persistence-BU" or name == "Persistence-OPT":
                continue
            all_term_preds = all_preds[name]

            # loop for avg terms
            for avg_terms in sorted(all_term_preds.keys()):
                # avg_terms = 12
                if name.__contains__("BU") and avg_terms == 1:
                    ipb_result[name][avg_terms] = deepcopy(
                        ipb_result[name.split("-")[0]][avg_terms])
                    opt_result[name][avg_terms] = deepcopy(
                        opt_result[name.split("-")[0]][avg_terms])
                    print(f"skip {name} for {avg_terms} avg_terms")
                    continue
                preds, labels = all_term_preds[avg_terms]

                preds *= 1.5e3
                labels *= 1.5e3


                avg_cost = 0
                count = 0
                # break
                # loop for samples
                ipb_samples, opt_cost = [], []
                for n in range(23, 23+7*24, 24):

                    Load = preds[n].squeeze()
                    True_load = labels[n].squeeze()
                    delta_t = avg_terms / 12


                    # DAED
                    # print("ideal schedule")
                    ideal_cost, ideal_P = daed(load=True_load, delta_t=delta_t)

                    # DAED
                    # print("daed schedule")
                    daed_cost, daed_P = daed(load=Load, delta_t=delta_t)

                    # IPB
                    ipb_cost, ipb_out = ipb(load=True_load,
                                            da_opt_P=daed_P,
                                            delta_t=delta_t)
                    # print(ipb_cost)

                    # error = True_load - daed_P.sum(axis=0)
                    # diffs = np.where(error > 0, error / eta_d * degrade_price,
                    #                  error * -eta_c * degrade_price).sum()


                    print("cost diff:")
                    diffs = ipb_cost - ideal_cost
                    print(diffs)
                    # print(diff)
                    ipb_samples.append(ipb_out)
                    opt_cost.append(diffs)
                    count += 1
                    print("--" * 30)

                avg_cost /= count
                ipb_result[name][avg_terms] = deepcopy(ipb_samples)
                opt_result[name][avg_terms] = deepcopy(opt_cost)


        all_opt_results.append(opt_result)
        all_ipb_schedule.append(ipb_result)

        with open(f'savings/large_ramp_opt_20_{dataset}_new.pickle', 'wb') as handle:
            pickle.dump(all_opt_results,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'savings/large_ramp_opt_20_{dataset}_schedule_new.pickle',
                  'wb') as handle:
            pickle.dump(all_ipb_schedule,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
