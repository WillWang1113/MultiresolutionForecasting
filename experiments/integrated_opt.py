import pickle
import numpy as np
import cvxpy as cp
from copy import deepcopy
import argparse

a = np.array([0.12, 0.17, 0.15, 0.19])
b = np.array([14.8, 16.57, 15.55, 16.21])
c = np.array([89, 83, 100, 70])

eta_c = 0.95
eta_d = 0.9
SOC_ini = 0.5
Cap = 300000
degrade_price = 100
shortage_price = 200
surplus_price = 200
P_battery = 40
reserve = 0
id_interval = 4 # intra-day interval: 4hr

N_g = 4
# N_t = 24
VWC = 200
Pg_min = np.array([28, 20, 30, 20])
Pg_max = np.array([200, 290, 190, 260])
RU = np.array([40, 30, 30, 50])
RD = np.array([40, 30, 30, 50])


def wind_post_process(preds, peneration_rate):
    preds = np.where(preds < 0, 0, preds)
    preds = np.where(preds > 16, 16, preds)
    preds *= peneration_rate * 600 / 16
    return preds


def load_post_process(preds):
    return preds * 1e3


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
    objective = cp.Minimize(
        sum((a @ P**2 + b @ P + c.sum())) * delta_t + VWC * sum(Pwc) * delta_t)

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
    Pp = cp.Variable(N_t, name='Pp')
    Pn = cp.Variable(N_t, name='Pn')

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
        (P_battery * phi), Pd <= (P_battery * (1 - phi)), Pp >= 0, Pn >= 0
    ]
    balance_constraints = [
        ((sum(P[:, t]) + Pd[t] + Pw[t] + Pp[t]) == (load[t] + Pc[t] + Pn[t]))
        for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]
    constraints = P_constraints + SOC_constraints + balance_constraints + W_constraints

    objective = cp.Minimize(delta_t * sum((a @ P**2 + b @ P + c.sum())) +
                            degrade_price * sum((
                                (Pc * eta_c + Pd / eta_d) * delta_t)) +
                            delta_t * VWC * sum(Pwc) +
                            shortage_price * sum(Pp) * delta_t +
                            surplus_price * sum(Pn) * delta_t)

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
    # Pn =

    P_constraints = [P == P_schedule]
    W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]

    balance_constraints = [
        ((sum(P[:, t]) + Pd_schedule[t] + Pw[t] + Pp[t])
         == (load[t] + Pc_schedule[t] + Pn[t])) for t in range(N_t)
    ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)] + [Pp >= 0, Pn >= 0]
    constraints = P_constraints + balance_constraints + W_constraints

    objective = cp.Minimize(
        delta_t * sum((a @ P**2 + b @ P + c.sum())) + degrade_price * sum((
            (Pc_schedule * eta_c + Pd_schedule / eta_d) * delta_t)) +
        delta_t * VWC * sum(Pwc) + delta_t * sum(shortage_price * Pp) +
        surplus_price * sum(Pn) * delta_t)

    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.GUROBI)
    assert prob.status == "optimal"

    # print((Pn.value * Pp.value).max())

    return result, (P.value, Pd_schedule, Pc_schedule, Pwc.value, Pw.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run for multi-resolution schedules")
    parser.add_argument("--peneration_rate", type=float, default=0.5)
    args = parser.parse_args()

    # load wind
    dataset = "nrel"
    with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
        proposed_wind_preds = pickle.load(handle)
    with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
        bench_wind_raw_preds = pickle.load(handle)
    with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
        bench_wind_bu_preds = pickle.load(handle)
    with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
        bench_wind_opt_preds = pickle.load(handle)

    # load load
    dataset = "mfred"
    with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
        proposed_load_preds = pickle.load(handle)
    with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
        bench_load_raw_preds = pickle.load(handle)
    with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
        bench_load_bu_preds = pickle.load(handle)
    with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
        bench_load_opt_preds = pickle.load(handle)

    all_seed_costs = []
    for i in range(20):
        # for i in range(len(proposed_wind_preds)):
        print(i, "/", len(proposed_wind_preds) - 1)
        proposed_wind_model = {"Proposed": proposed_wind_preds[i]}
        all_wind_preds = {
            **proposed_wind_model,
            **bench_wind_raw_preds[i],
            **bench_wind_bu_preds[i],
            **bench_wind_opt_preds[i]
        }
        for key in ["Persistence-BU", "Persistence-OPT"]:
            all_wind_preds.pop(key, None)

        proposed_load_model = {"Proposed": proposed_load_preds[i]}
        all_load_preds = {
            **proposed_load_model,
            **bench_load_raw_preds[i],
            **bench_load_bu_preds[i],
            **bench_load_opt_preds[i]
        }
        for key in ["Persistence-BU", "Persistence-OPT"]:
            all_load_preds.pop(key, None)
        # print(all_load_preds.keys())
        # print(all_wind_preds.keys())
        assert all_load_preds.keys() == all_wind_preds.keys()

        opt_result = {name: {} for name in sorted(all_load_preds.keys())}
        ipb_result = {name: {} for name in sorted(all_load_preds.keys())}
        model_num = len(all_load_preds.keys())
        # # loop for models
        cost_results = np.zeros((model_num, model_num, 3))
        for ll, load_name in enumerate(sorted(all_load_preds.keys())):
            print(load_name)
            if load_name == "Persistence-BU" or load_name == "Persistence-OPT":
                continue

            load_preds = all_load_preds[load_name]
            da_load_preds, da_load_labels = load_preds[12]
            da_load_preds, da_load_labels = da_load_preds[23:], da_load_labels[
                23:]

            # 5min preds
            ipb_load_preds, ipb_load_labels = load_preds[1]
            ipb_load_preds, ipb_load_labels = ipb_load_preds[
                23:], ipb_load_labels[23:]

            # 15min preds
            # ipb_load_preds, ipb_load_labels = load_preds[3]

            da_load_preds = load_post_process(da_load_preds)
            da_load_labels = load_post_process(da_load_labels)
            ipb_load_preds = load_post_process(ipb_load_preds)
            ipb_load_labels = load_post_process(ipb_load_labels)

            for ww, wind_name in enumerate(sorted(all_wind_preds.keys())):
                print(wind_name)
                if wind_name == "Persistence-BU" or wind_name == "Persistence-OPT":
                    continue

                wind_preds = all_wind_preds[wind_name]
                da_wind_preds, da_wind_labels = wind_preds[12]
                da_wind_preds, da_wind_labels = da_wind_preds[
                    2:], da_wind_labels[2:]
                # 5min preds
                ipb_wind_preds, ipb_wind_labels = wind_preds[1]
                ipb_wind_preds, ipb_wind_labels = ipb_wind_preds[
                    2:], ipb_wind_labels[2:]

                # 15min preds
                # ipb_wind_preds, ipb_wind_labels = wind_preds[3]

                da_wind_preds = wind_post_process(da_wind_preds,
                                                  args.peneration_rate)
                da_wind_labels = wind_post_process(da_wind_labels,
                                                   args.peneration_rate)
                ipb_wind_preds = wind_post_process(ipb_wind_preds,
                                                   args.peneration_rate)
                ipb_wind_labels = wind_post_process(ipb_wind_labels,
                                                    args.peneration_rate)

                avg_da_cost, avg_ipb_cost, avg_rt_cost = 0, 0, 0
                # day loop
                for i, day in enumerate(range(0, 7 * 24, 24)):
                    # day ahead schedule
                    da_cost, da_schedule = day_ahead(
                        da_load_preds[day].squeeze(),
                        da_wind_preds[day].squeeze(), 1)
                    # print(da_cost)

                    # interpolate(repeat)
                    # da_schedule = np.repeat(da_schedule, 4, axis=1)
                    da_schedule = np.repeat(da_schedule, 12, axis=1)

                    # ipb schedule
                    # THE SAME AS ROLLING, DECOUPLED
                    ipb_wind_fcst = np.concatenate([
                        ipb_wind_preds[hour].squeeze()[:12 * id_interval]
                        for hour in range(day, day + 24, id_interval)
                    ])
                    ipb_load_fcst = np.concatenate([
                        ipb_load_preds[hour].squeeze()[:12 * id_interval]
                        for hour in range(day, day + 24, id_interval)
                    ])

                    # ipb_fcst = ipb_preds[day].squeeze()

                    ipb_cost, ipb_schedule = intra_day(ipb_load_fcst,
                                                       ipb_wind_fcst,
                                                       da_schedule=da_schedule,
                                                       delta_t=1 / 12)
                    print(ipb_cost)
                    (P, SOC, Pc, Pd, Pwc, Pw) = ipb_schedule

                    rt_cost, rt_schedule = real_time(
                        ipb_load_labels[day].squeeze(),
                        ipb_wind_labels[day].squeeze(),
                        P_schedule=P,
                        Pd_schedule=Pd,
                        Pc_schedule=Pc,
                        delta_t=1 / 12)
 
                    avg_da_cost += da_cost
                    avg_ipb_cost += ipb_cost
                    avg_rt_cost += rt_cost
                    print("--" * 20)

                cost_results[ll, ww, 0] = avg_da_cost
                cost_results[ll, ww, 1] = avg_ipb_cost
                cost_results[ll, ww, 2] = avg_rt_cost

        all_seed_costs.append(deepcopy(cost_results))

        with open(f'savings/daid_all_strict_pn_{args.peneration_rate}.pickle',
                  'wb') as handle:
            pickle.dump((all_seed_costs, sorted(all_load_preds.keys())),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
