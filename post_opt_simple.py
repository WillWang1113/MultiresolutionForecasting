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
degrade_price = 50
shortage_price = 200
surplus_price = 100

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


def day_ahead(load, delta_t):
    # print(np.diff(load).min())
    # print(np.diff(load).max())

    N_t = len(load)
    P = cp.Variable((N_g, N_t), name='P')

    P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
        P[:, t] >= Pg_min for t in range(N_t)
    ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)
         ] + [P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)]

    balance_constraints = [((sum(P[:, t])) == (load[t])) for t in range(N_t)]
    constraints = P_constraints + balance_constraints
    # print(constraints)
    objective = cp.Minimize(sum((a @ P**2 + b @ P + c.sum())))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    # print(P.value)
    return result, P.value


def intra_day(load, da_opt_P, delta_t, SOC_ini=0.5):
    N_t = len(load)
    # print(load.shape)
    # print(da_opt_P.shape)
    P = cp.Variable((N_g, N_t), name='P')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')
    phi = cp.Variable(N_t, name='phi', boolean=True)
    # Pb = cp.Variable(N_t, name='Pb')
    # Ps = cp.Variable(N_t, name='Ps')
    # bs = cp.Variable(N_t, name='phi', boolean=True)

    P_constraints = [P == da_opt_P]
    # P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
    #     P[:, t] >= Pg_min for t in range(N_t)
    # ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)] + [
    #     P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)
    # ] + [P <= da_opt_P + reserve * delta_t, da_opt_P - reserve * delta_t <= P]
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
    objective = cp.Minimize(delta_t * sum(
        (a @ P**2 + b @ P + c.sum())) + delta_t * sum((
            (Pc * eta_c * degrade_price + Pd / eta_d * degrade_price))))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    # print(P.value)
    return result, (P.value, SOC.value, Pc.value, Pd.value)


def real_time(real_load, scheduled_load, delta_t):
    error = real_load - scheduled_load
    diffs = np.where(error > 0, error * shortage_price * delta_t,
                     error * -surplus_price * delta_t).sum()
    return diffs


# load = np.array([
#     510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, 754, 700, 686,
#     720, 714, 761, 727, 714, 618, 584, 578, 544
# ])

# high_res_load = load + np.random.normal(0, scale=10, size=load.shape)

# _, da_schedule = daed(load, 1)
# print(da_schedule)
# cost, d = ipb(load=high_res_load,
#     da_opt_P=da_schedule,
#     delta_t=1,
#     SOC_ini=0.5
#     )
# print(cost)

# start = 0
# all = 0
# SOC_ini=0.5
# for i in range(4):
#     end = start + 6
#     temp, d = ipb(load=high_res_load[start:end],
#     da_opt_P=da_schedule[:,start:end],
#     delta_t=1,
#     SOC_ini=SOC_ini
#     )
#     start = end
#     SOC_ini = d[1][-1]
#     all+=temp
# print(all)
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
            all_term_preds = all_preds[name]

            # hourly preds
            da_preds, da_labels = all_term_preds[12]

            # 5min preds
            # ipb_preds, ipb_labels = all_term_preds[1]

            # 15min preds
            ipb_preds, ipb_labels = all_term_preds[3]

            da_preds *= 1e3
            da_labels *= 1e3
            ipb_preds *= 1e3
            ipb_labels *= 1e3
            samples_hour = len(da_preds)

            objectives, schedules = [], []
            # day loop
            for day in range(23, len(da_preds)-24, 24):

                # day ahead schedule
                da_fcst = da_preds[day].squeeze()
                da_cost, da_schedule = day_ahead(da_fcst, 1)
                print(da_cost)
                # interpolate(repeat)
                da_schedule = np.repeat(da_schedule, 4, axis=1)
                # da_schedule = np.repeat(da_schedule, 12, axis=1)

                # ipb schedule
                # THE SAME AS ROLLING, DECOUPLED
                ipb_fcst = np.concatenate([
                    ipb_preds[hour].squeeze()[:4 * 4]
                    for hour in range(day, day + 24, 4)
                ])
                ipb_cost, ipb_schedule = intra_day(ipb_fcst,
                                                   da_schedule,
                                                   delta_t=1 / 4)
                # ipb_fcst = np.concatenate([
                #     ipb_preds[hour].squeeze()[:12 * 4]
                #     for hour in range(day, day + 24, 4)
                # ])
                # ipb_cost, ipb_schedule = intra_day(ipb_fcst,
                #                                    da_schedule,
                #                                    delta_t=1 / 12)
                print(ipb_cost)

                # imbalance penalty
                final_cost = ipb_cost + real_time(
                    real_load=ipb_labels[day].squeeze(),
                    scheduled_load=ipb_fcst,
                    delta_t=1 / 4)
                # final_cost = ipb_cost + imbalance_penalty(
                #     real_load=ipb_labels[day].squeeze(),
                #     scheduled_load=ipb_fcst,
                #     delta_t=1 / 12)

                print(final_cost)

                # break
                print("--" * 20)
                objectives.append((da_cost, ipb_cost, final_cost))
                schedules.append(ipb_schedule)
            ipb_result[name] = deepcopy(objectives)
            opt_result[name] = deepcopy(schedules)

        all_opt_results.append(opt_result)
        all_ipb_schedule.append(ipb_result)

        with open(f'savings/daid_opt_20_{dataset}.pickle', 'wb') as handle:
            pickle.dump(all_opt_results,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'savings/daid_opt_20_{dataset}_schedule.pickle',
                  'wb') as handle:
            pickle.dump(all_ipb_schedule,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
