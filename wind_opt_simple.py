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
Cap = 300000
degrade_price = 50
shortage_price = 200
surplus_price = 100
P_battery = 500
reserve = 0
id_interval = 4

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


def post_process(preds):
    preds = np.where(preds < 0, 0, preds)
    preds = np.where(preds > 16, 16, preds)
    return preds * 3e1


def day_ahead(load, wind, delta_t):
    N_t = len(load)
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
    objective = cp.Minimize(sum((a @ P**2 + b @ P + c.sum())) + VWC * sum(Pwc))

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"

    return result, P.value


def intra_day(load, wind, delta_t, da_schedule):
    N_t = len(load)

    P = cp.Variable((N_g, N_t), name='P')
    Pw = cp.Variable(N_t, name='Pw')
    Pwc = cp.Variable(N_t, name='Pwc')
    SOC = cp.Variable(N_t, name='SOC')
    Pc = cp.Variable(N_t, name='Pc')
    Pd = cp.Variable(N_t, name='Pd')

    phi = cp.Variable(N_t, name='phi', boolean=True)
    P_constraints = [P == da_schedule]
    W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]
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
    balance_constraints = [
        ((sum(P[:, t]) + Pd[t] + Pw[t]) == (load[t] + Pc[t]))
        for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]
    constraints = P_constraints + SOC_constraints + balance_constraints + W_constraints

    objective = cp.Minimize(
        delta_t * sum((a @ P**2 + b @ P + c.sum())) +
        degrade_price * sum(((Pc * eta_c + Pd / eta_d) * delta_t)) +
        VWC * sum(Pwc))

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"

    return result, (P.value, SOC.value, Pc.value, Pd.value, Pwc.value,
                    Pw.value)


def real_time(load, wind, P_schedule, Pd_schedule, Pc_schedule, delta_t):
    N_t = len(load)

    P = cp.Variable((N_g, N_t), name='P')
    Pw = cp.Variable(N_t, name='Pw')
    Pwc = cp.Variable(N_t, name='Pwc')
    Pp = cp.Variable(N_t, name='Pp')
    Pn = cp.Variable(N_t, name='Pn')
    phi = cp.Variable(N_t, name='phi', boolean=True)

    P_constraints = [P == P_schedule]
    W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]

    balance_constraints = [
        ((sum(P[:, t]) + Pd_schedule[t] + Pw[t] + Pp[t]) == (load[t] + Pc_schedule[t] + Pn[t]))
        for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)] +[
        Pp >= 0, Pn >= 0, Pp <=
        (P_battery * phi), Pn <= (P_battery * (1 - phi))
    ]
    constraints = P_constraints + balance_constraints + W_constraints

    objective = cp.Minimize(
       delta_t *  sum((a @ P**2 + b @ P + c.sum())) + degrade_price * sum((
            (Pc_schedule * eta_c + Pd_schedule / eta_d) * delta_t)) +
        VWC * sum(Pwc)+ delta_t *degrade_price*sum(5*Pp+Pn)) 

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"
    return result, (P.value, Pd_schedule, Pc_schedule, Pwc.value, Pw.value)



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

            all_term_preds = all_preds[name]

            # hourly preds
            da_preds, da_labels = all_term_preds[12]

            # 5min preds
            ipb_preds, ipb_labels = all_term_preds[1]

            # 15min preds
            # ipb_preds, ipb_labels = all_term_preds[3]

            da_preds = post_process(da_preds)
            da_labels = post_process(da_labels)
            ipb_preds = post_process(ipb_preds)
            ipb_labels = post_process(ipb_labels)
            samples_hour = len(da_preds)

            objectives, schedules = [], []
            hourly_load = load.resample(f"60min").mean().values.reshape(-1, 24)
            # highres_load = load.resample(f"15min").mean().values.reshape(
                # -1, 96)
            highres_load = load.resample(f"5min").mean().values.reshape(-1, 288)

            start = 0
            # day loop
            for i, day in enumerate(range(2, len(da_preds) - 24, 24)):
                da_load = hourly_load[i]
                ipb_load = highres_load[i]

                # day ahead schedule
                da_fcst = da_preds[day].squeeze()
                da_cost, da_schedule = day_ahead(da_load, da_fcst, 1)
                print(da_cost)
                # interpolate(repeat)
                # da_schedule = np.repeat(da_schedule, 4, axis=1)
                da_schedule = np.repeat(da_schedule, 12, axis=1)

                # ipb schedule
                # THE SAME AS ROLLING, DECOUPLED
                ipb_fcst = np.concatenate([
                    ipb_preds[hour].squeeze()[:12 * id_interval]
                    for hour in range(day, day + 24, id_interval)
                ])
                # ipb_fcst = ipb_preds[day].squeeze()

                ipb_cost, ipb_schedule = intra_day(ipb_load,
                                                   ipb_fcst,
                                                   da_schedule=da_schedule,
                                                   delta_t=1 / 12)
                # ipb_fcst = np.concatenate([
                #     ipb_preds[hour].squeeze()[:4 * 4]
                #     for hour in range(day, day + 24, 4)
                # ])

                # ipb_cost, ipb_schedule = intra_day(ipb_load,
                #                                    ipb_fcst,
                #                                    da_schedule=da_schedule,
                #                                    delta_t=1 / 4)
                print(ipb_cost)
                (P, SOC, Pc, Pd, Pwc, Pw) = ipb_schedule

                rt_cost, rt_schedule = real_time(ipb_load,
                                                 ipb_labels[day].squeeze(),
                                                 P_schedule=P,
                                                 Pd_schedule=Pd,
                                                 Pc_schedule=Pc,
                                                 delta_t=1 / 12)
                # rt_cost, rt_schedule = real_time(ipb_load,
                #                                  ipb_labels[day].squeeze(),
                #                                  P_schedule=P,
                #                                  Pd_schedule=Pd,
                #                                  Pc_schedule=Pc,
                #                                  delta_t=1 / 4)
                print(rt_cost)
                print("--" * 20)
                objectives.append((da_cost, ipb_cost, rt_cost))
                schedules.append(rt_schedule)
            ipb_result[name] = deepcopy(objectives)
            opt_result[name] = deepcopy(schedules)

        all_opt_results.append(opt_result)
        all_ipb_schedule.append(ipb_result)

        with open(f'savings/daid_5min_20_{dataset}.pickle', 'wb') as handle:
            pickle.dump(all_opt_results,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'savings/daid_5min_20_{dataset}_schedule.pickle',
                  'wb') as handle:
            pickle.dump(all_ipb_schedule,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)