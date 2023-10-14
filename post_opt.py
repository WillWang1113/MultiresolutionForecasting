import pickle
import numpy as np

import cvxpy as cp
import pandas as pd
from utils import setup_seed
from copy import deepcopy
import matplotlib.pyplot as plt

setup_seed(9)

Cap = 30000000
SOC_ini = 0.4
# SOC_exp = 0.8
eta_c = 0.95
eta_d = 0.9
N_g = 4
degrade_price = 800
# shortage_price = 10
# surplus_price = 500

P_battery = 180000
# scale_min = 500.0
# scale_max = 800.0
reserve = 0

a = np.array([0.12, 0.17, 0.15, 0.19])
b = np.array([14.8, 16.57, 15.55, 16.21])
c = np.array([89, 83, 100, 70])
Pg_min = np.array([28, 20, 30, 20])
Pg_max = np.array([200, 290, 190, 260])
RU = np.array([40, 30, 30, 50])
RD = np.array([40, 30, 30, 50])


# def daed(load, delta_t):

#     N_t = len(load)
#     P = cp.Variable((N_g, N_t), name='P')

#     P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
#         P[:, t] >= Pg_min for t in range(N_t)
#     ] + [P[:, t + 1] - P[:, t] <= RU for t in range(N_t - 1)
#          ] + [P[:, t] - P[:, t + 1] <= RD for t in range(N_t - 1)]

#     balance_constraints = [((sum(P[:, t])) == (load[t])) for t in range(N_t)]
#     constraints = P_constraints + balance_constraints
#     # print(constraints)
#     objective = cp.Minimize(
#         sum([a @ P[:, t]**2 + b @ P[:, t] + c.sum() for t in range(N_t)]))
#     prob = cp.Problem(objective, constraints)
#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     da_opt_P = P.value
#     # print(np.sum(da_opt_P, axis=0))
#     # print(SOC.value)
#     # print(load[0])
#     # print(Pd[0].value)
#     return result, da_opt_P


# def ipb(load, da_opt_P, delta_t):

#     N_t = len(load)

#     P = cp.Variable((N_g, N_t), name='P')
#     SOC = cp.Variable(N_t, name='SOC')
#     Pc = cp.Variable(N_t, name='Pc')
#     Pd = cp.Variable(N_t, name='Pd')
#     phi = cp.Variable(N_t, name='phi', boolean=True)

#     P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
#         P[:, t] >= Pg_min for t in range(N_t)
#     ] + [P[:, t + 1] - P[:, t] <= RU for t in range(N_t - 1)] + [
#         P[:, t] - P[:, t + 1] <= RD for t in range(N_t - 1)
#     ] + [P <= da_opt_P + reserve, da_opt_P - reserve <= P]
#     SOC_constraints = [
#         Cap * SOC[0] == (Cap * SOC_ini +
#                          (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
#     ] + [
#         Cap * SOC[t] == (Cap * SOC[t - 1] +
#                          (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
#         for t in range(1, N_t)
#     ] + [
#         SOC >= 0.05, SOC <= 0.95, Pc >= 0, Pd >= 0, Pc <=
#         (P_battery * phi), Pd <= (P_battery * (1 - phi))
#     ]
#     balance_constraints = [((sum(P[:, t]) + Pd[t]) == (load[t] + Pc[t]))
#                            for t in range(N_t)]
#     constraints = P_constraints + SOC_constraints + balance_constraints
#     # print(constraints)
#     objective = cp.Minimize(
#         sum([a @ P[:, t]**2 + b @ P[:, t] + c.sum() for t in range(N_t)]) +
#         sum(((Pc * eta_c * degrade_price + Pd / eta_d * degrade_price))))
#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     # print(phi.value[index])
#     # print(SOC.value)
#     # print(Pd.value)
#     # print(Pc.value)
#     # print(Pb.value)
#     # print(Ps.value)
#     return result, (P.value, SOC.value, Pc.value, Pd.value)


def daed(load, delta_t):
    # print(np.diff(load).min())
    # print(np.diff(load).max())

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
        delta_t * sum((a @ P**2 + b @ P + c.sum())) + delta_t *
        degrade_price * sum(((Pc * eta_c + Pd / eta_d))))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    da_opt_P = P.value
    # print(np.sum(da_opt_P, axis=0))
    # print(SOC.value)
    # print(load[0])
    # print(Pd[0].value)
    # print((Pd - Pc).value)
    # print(np.diff(da_opt_P[0]).min())
    # print(np.diff(da_opt_P[0]).max())
    return result, da_opt_P

def ipb(load, da_opt_P, delta_t):
    N_t = len(load)

    P = cp.Variable((N_g, N_t), name='P')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')
    phi = cp.Variable(N_t, name='phi', boolean=True)
    # Pb = cp.Variable(N_t, name='Pb')
    # Ps = cp.Variable(N_t, name='Ps')
    # bs = cp.Variable(N_t, name='phi', boolean=True)

    # P_constraints = [P == da_opt_P]
    P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
        P[:, t] >= Pg_min for t in range(N_t)
    ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)] + [
        P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)
    ] + [P <= da_opt_P + reserve * delta_t, da_opt_P - reserve * delta_t <= P]
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
        delta_t * sum((a @ P**2 + b @ P + c.sum())) + delta_t *
        sum(((Pc * eta_c * degrade_price  + Pd / eta_d * degrade_price))))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    # print(phi.value[index])
    # print(SOC.value)
    # print(Pd.value)
    # print(Pc.value)
    # print(Pb.value)
    # print(Ps.value)
    # print((Pd - Pc).value)

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

                preds *= 1e3
                labels *= 1e3

                # preds = (preds - preds.min()) / (preds.max() - preds.min()) * (
                #     scale_max - scale_min) + scale_min
                # labels = (labels - labels.min()) / (labels.max() - labels.min(
                # )) * (scale_max - scale_min) + scale_min

                avg_cost = 0
                count = 0
                # break
                # loop for samples
                ipb_samples, opt_cost = [], []
                for n in range(23, len(preds), 24):
                    # for n in range(2):
                    # for n in range(0, len(preds), 24):
                    Load = preds[n].squeeze()
                    True_load = labels[n].squeeze()
                    delta_t = avg_terms / 12
                    # if abs(np.diff(Load).min()) > (RU * delta_t).sum() or abs(np.diff(Load).max()) > (RU * delta_t).sum():
                    #     print("111111111")
                    # print(np.diff(Load).min())
                    # print(np.diff(Load).max())
                    # print("load deviation:")
                    # index = (Load - True_load).argmin()
                    # print(np.mean((Load - True_load)**2))
                    # print((Load - True_load).max())
                    # print((Load - True_load).min())
                    # print((Load - True_load))

                    # DAED
                    # print("ideal schedule")
                    # ideal_cost, ideal_P = daed(load=True_load, delta_t=delta_t)
                    # plt.plot(ideal_P.T)

                    # DAED
                    # print("daed schedule")
                    daed_cost, daed_P = daed(load=Load, delta_t=delta_t)
                    # plt.plot(daed_P.T)
                    # plt.savefig("test.png")
                    # plt.close()
                    # IPB
                    ipb_cost, ipb_out = ipb(load=True_load,
                                            da_opt_P=daed_P,
                                            delta_t=delta_t)
                    # print(ipb_cost)

                    # error = True_load - Load
                    # diffs = np.where(error > 0, error / eta_d * degrade_price,
                    #                  error * -eta_c * degrade_price).sum()
                    # print(((ipb_out[3] - ipb_out[2]) - error).sum())
                    # # print(error)
                    # # print((ipb_out[0].sum(axis=0) - daed_P.sum(axis=0)).sum())
                    # print(diffs)
                    # print(ideal_cost)
                    # print(ideal_P)
                    # print(daed_cost)
                    # print(daed_P)
                    # print(ipb_cost)
                    # print(ipb_P)

                    print("cost diff:")
                    # print(diffs)
                    print(ipb_cost)
                    # # print("ideal diff:")
                    # # print(ideal_cost)
                    ipb_samples.append(ipb_out)
                    # # opt_cost.append(diffs)
                    # # avg_cost += diffs
                    opt_cost.append(ipb_cost)
                    # avg_cost += ipb_cost - ideal_cost
                    count += 1
                    print("--" * 30)
                    # break

                avg_cost /= count
                ipb_result[name][avg_terms] = deepcopy(ipb_samples)
                opt_result[name][avg_terms] = deepcopy(opt_cost)
                # break
            # print("--" * 30)
            # break
        # print(1111)
        # break

        all_opt_results.append(opt_result)
        all_ipb_schedule.append(ipb_result)

        with open(f'savings/large2_ramp_opt_20_{dataset}.pickle', 'wb') as handle:
            pickle.dump(all_opt_results,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'savings/large2_ramp_opt_20_{dataset}_schedule.pickle',
                  'wb') as handle:
            pickle.dump(all_ipb_schedule,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
