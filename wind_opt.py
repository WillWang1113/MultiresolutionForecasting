import pickle
import numpy as np

import cvxpy as cp
import pandas as pd
from utils import setup_seed
from copy import deepcopy
import matplotlib.pyplot as plt
# setup_seed(9)

a = np.array([0.12, 0.17, 0.15, 0.19])
b = np.array([14.8, 16.57, 15.55, 16.21])
c = np.array([89, 83, 100, 70])

eta_c = 0.95
eta_d = 0.9
SOC_ini = 0.5
Cap = 3000
degrade_price = 50
shortage_price = 500
surplus_price = 80
P_battery = 100
reserve = 0


N_g = 4
# N_t = 24
VWC = 100
Pg_min = np.array([28, 20, 30, 20])
Pg_max = np.array([200, 290, 190, 260])
RU = np.array([40, 30, 30, 50])
RD = np.array([40, 30, 30, 50])

load = np.array([
    510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, 754, 700, 686,
    720, 714, 761, 727, 714, 618, 584, 578, 544
])

load_scale_max = load.max() * 0.8
load_scale_min = load.min() / 0.8

W = np.array([
    44.1, 48.5, 65.7, 144.9, 202.3, 317.3, 364.4, 317.3, 271, 306.9, 424.1,
    398, 487.6, 521.9, 541.3, 560, 486.8, 372.6, 367.4, 314.3, 316.6, 311.4,
    405.4, 470.4
])

wind_scale_max = W.max()
wind_scale_min = W.min()


def daed(load, wind, delta_t):
    N_t = len(load)
    # print((load - wind).min())
    # print((load - wind).max())
    P = cp.Variable((N_g, N_t), name='P')
    Pw = cp.Variable(N_t, name='Pw')
    Pwc = cp.Variable(N_t, name='Pwc')

    P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
        P[:, t] >= Pg_min for t in range(N_t)
    ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)
         ] + [P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)]
    W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]

    balance_constraints = [
        ((sum(P[:, t]) + Pw[t]) == (load[t])) for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]

    constraints = P_constraints + balance_constraints + W_constraints
    objective = cp.Minimize(
        sum([a @ P[:, t]**2 + b @ P[:, t] + c.sum()
             for t in range(N_t)]) + VWC * sum(Pwc))

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    print(Pwc.value.min())
    print(Pwc.value.max())

    return result, P.value


def ipb(load, wind, delta_t, da_schedule):
    N_t = len(load)
    # print((load - wind).min())
    # print((load - wind).max())
    P = cp.Variable((N_g, N_t), name='P')
    Pw = cp.Variable(N_t, name='Pw')
    Pwc = cp.Variable(N_t, name='Pwc')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')
    Pb = cp.Variable(N_t, name='Pb')
    phi = cp.Variable(N_t, name='phi', boolean=True)
    P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
        P[:, t] >= Pg_min for t in range(N_t)
    ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)
         ] + [P[:, t] - P[:, t + 1] <= RD * delta_t
              for t in range(N_t - 1)] + [
                  P <= da_schedule + reserve * delta_t,
                  da_schedule - reserve * delta_t <= P
              ]
    W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]
    SOC_constraints = [
        Cap * SOC[0] == (Cap * SOC_ini +
                         (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
    ] + [
        Cap * SOC[t] == (Cap * SOC[t - 1] +
                         (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
        for t in range(1, N_t)
    ] + [
        SOC >= 0., SOC <= 1., Pc >= 0, Pd >= 0, Pc <= (P_battery * phi), Pd <=
        (P_battery * (1 - phi)), SOC[-1] == SOC_ini
    ] + [Pb >= 0]
    balance_constraints = [
        ((sum(P[:, t]) + Pd[t] + Pw[t] + Pb[t]) == (load[t] + Pc[t]))
        for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]
    constraints = P_constraints + SOC_constraints + balance_constraints + W_constraints

    objective = cp.Minimize(
        sum([a @ P[:, t]**2 + b @ P[:, t] + c.sum() for t in range(N_t)]) +
        degrade_price * sum(((Pc * eta_c+ Pd / eta_d) * delta_t)) +
        VWC * sum(Pwc) + sum(shortage_price * Pb))

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    # print(Pb.value)
    # print(Pwc.value)
    # print(SOC.value)

    # return result, (P.value, SOC.value, Pwc.value, Pb.value, Pc.value, Pd.value)
    return result, (P.value, SOC.value, Pc.value, Pd.value, Pwc.value,Pw.value, Pb.value)


# result, da_schedule = daed(load=load, wind=W, delta_t=1)
# print(result)
# result, da_schedule = daed(load=load,
#                            wind=W + np.random.normal(-10, 1, size=W.shape),
#                            delta_t=1)
# print(result)
# result, _ = ipb(load=load, wind=W, delta_t=1, da_schedule=da_schedule)
# print(result)

if __name__ == "__main__":
    dataset = "nrel"

    with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
        proposed_preds = pickle.load(handle)
    with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
        bench_raw_preds = pickle.load(handle)
    with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
        bench_bu_preds = pickle.load(handle)
    with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
        bench_opt_preds = pickle.load(handle)
    load = pd.read_csv("datasets/MFRED_wiztemp.csv",
                       index_col=0,
                       parse_dates=True)[["AGs01To26_kW"]]
    load = load["2019-12-13":"2019-12-31"]
    print(load)
    load = (load - load.min()) / (load.max() - load.min()) * (
        load_scale_max - load_scale_min) + load_scale_min
    print(load.shape)

    all_opt_results = []
    all_ipb_schedule = []
    df_all_opt_results = []
    # for i in range(1):
    for i in range(len(proposed_preds)):
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
            if name.__contains__("Persistence"):
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
                preds = (preds - preds.min()) / (preds.max() - preds.min()) * (
                    wind_scale_max - wind_scale_min) + wind_scale_min
                labels = (labels - labels.min()) / (labels.max() - labels.min(
                )) * (wind_scale_max - wind_scale_min) + wind_scale_min

                avg_load = load.resample(f"{avg_terms * 5}min").mean()
                # print(avg_load.shape)

                avg_cost = 0
                count = 0
                start = 0
                ipb_samples, opt_cost = [], []
                # loop for samples
                for n in range(2, len(preds), 24):
                    # for n in range(1):
                    print(n, "/", len(preds))
                    wind = preds[n].squeeze()
                    true_wind = labels[n].squeeze()
                    daily_load = avg_load.values[start:start +
                                                 288 // avg_terms].squeeze()
                    # print(daily_load)
                    # plt.plot(daily_load)
                    # plt.savefig("wind.png")
                    # plt.close()
                    # break

                    # DAED
                    ideal_cost, ideal_P = daed(load=daily_load,
                                               wind=true_wind,
                                               delta_t=avg_terms / 12)

                    # DAED
                    daed_cost, daed_P = daed(load=daily_load,
                                             wind=wind,
                                             delta_t=avg_terms / 12)
                    # IPB
                    ipb_cost, ipb_out = ipb(load=daily_load,
                                            wind=true_wind,
                                            da_schedule=daed_P,
                                            delta_t=avg_terms / 12)

                    print("cost diff:")
                    print(ipb_cost - ideal_cost)
                    print("ideal diff:")
                    print(ideal_cost)
                    ipb_samples.append(ipb_out)
                    opt_cost.append(ipb_cost - ideal_cost)

                    avg_cost += ipb_cost - ideal_cost
                    count += 1
                    start += 288 // avg_terms
                    # print("--" * 30)
                    # break
                
                avg_cost /= count
                # opt_result[name][avg_terms] = avg_cost
                ipb_result[name][avg_terms] = deepcopy(ipb_samples)
                opt_result[name][avg_terms] = deepcopy(opt_cost)
            print("--" * 30)
            # break
        # df_opt_result = pd.DataFrame(opt_result)
        # df_all_opt_results.append(df_opt_result)
        print(1111)
        all_opt_results.append(opt_result)
        all_ipb_schedule.append(ipb_result)

        with open(f'savings/ramp_opt_20_{dataset}.pickle', 'wb') as handle:
            pickle.dump(all_opt_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'savings/ramp_opt_20_{dataset}_schedule.pickle', 'wb') as handle:
            pickle.dump(all_ipb_schedule, handle, protocol=pickle.HIGHEST_PROTOCOL)

