# import pickle
# import numpy as np

# import cvxpy as cp
# import pandas as pd
# from utils import setup_seed
# from copy import deepcopy

# # setup_seed(9)

# a = np.array([0.12, 0.17, 0.15, 0.19])
# b = np.array([14.8, 16.57, 15.55, 16.21])
# c = np.array([89, 83, 100, 70])

# eta_c = 0.95
# eta_d = 0.9
# SOC_ini = 0.5
# Cap = 200
# P_battery = 50
# P_line_max = 150
# kwf = 30
# kb = 30
# price = np.array([
#     30, 49, 49, 40, 45, 75, 85, 90, 90, 89, 89, 92, 95, 95, 89, 80, 105, 115,
#     102, 85, 68, 60, 58, 50
# ])
# price = pd.DataFrame(price,
#                      index=pd.date_range("2000-01-01 00:00:00",
#                                          "2000-01-02 00:00:00",
#                                          periods=24))

# load = np.array([
#     510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, 754, 700, 686,
#     720, 714, 761, 727, 714, 618, 584, 578, 544
# ])

# load_scale_max = load.max() / 0.8
# load_scale_min = load.min() * 0.8

# W = np.array([
#     44.1, 48.5, 65.7, 144.9, 202.3, 317.3, 364.4, 317.3, 271, 306.9, 424.1,
#     398, 487.6, 521.9, 541.3, 560, 486.8, 372.6, 367.4, 314.3, 316.6, 311.4,
#     405.4, 470.4
# ])

# wind_scale_max = 160
# wind_scale_min = 10

# scaled_W = (W - W.min()) / (W.max() - W.min()) * (200 - 10) + 10

# def wind_mkt(wind, price, delta_t):
#     N_t = len(wind)

#     Pw = cp.Variable(N_t, name='Pw')
#     Pwc = cp.Variable(N_t, name='Pwc')
#     E = cp.Variable(N_t, name='E')
#     Pc = cp.Variable(N_t, name='Pc')
#     Pd = cp.Variable(N_t, name='Pd')
#     phi = cp.Variable(N_t, name='phi', boolean=True)

#     W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0
#                      ] + [Pw + Pwc == wind]
#     SOC_constraints = [
#         E[0] == (Cap * SOC_ini + (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
#     ] + [
#         E[t] == (E[t - 1] + (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
#         for t in range(1, N_t)
#     ] + [E[-1] == Cap * SOC_ini] + [
#         E >= 0. * Cap, E <= 1. * Cap, Pc >= 0, Pd >= 0, Pc <=
#         (P_battery * phi), Pd <= (P_battery * (1 - phi)), Pc <= wind
#     ]
#     balance_constraints = [(Pd - Pc + Pw) <= P_line_max]
#     constraints = SOC_constraints + balance_constraints + W_constraints
#     revenue = price.T @ (Pd - Pc + Pw)
#     p_curtail = kwf * sum(Pwc)
#     p_degrate = kb * sum(((Pc * eta_c + Pd / eta_d) * delta_t))
#     objective = cp.Maximize(revenue - p_curtail - p_degrate)

#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     # print(Pc.value)
#     # print(Pd.value)
#     # print(E.value)
#     # print(Pwc.value.sum())
#     # print((Pd - Pc + Pw).value)
#     # print(revenue.value)
#     # print(p_curtail.value)
#     # print(p_degrate.value)

#     return result, (E.value, Pc.value, Pd.value, Pwc.value)

# def ipb(wind, price, delta_t, schedule):
#     N_t = len(wind)

#     Pw = cp.Variable(N_t, name='Pw')
#     Pwc = cp.Variable(N_t, name='Pwc')
#     E = cp.Variable(N_t, name='E')
#     Pc = cp.Variable(N_t, name='Pc')
#     Pd = cp.Variable(N_t, name='Pd')
#     phi = cp.Variable(N_t, name='phi', boolean=True)

#     W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0
#                      ] + [Pw + Pwc == wind]
#     SOC_constraints = [
#         E[0] == (Cap * SOC_ini + (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
#     ] + [
#         E[t] == (E[t - 1] + (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
#         for t in range(1, N_t)
#     ] + [E[-1] == Cap * SOC_ini] + [
#         E >= 0. * Cap, E <= 1. * Cap, Pc >= 0, Pd >= 0, Pc <=
#         (P_battery * phi), Pd <= (P_battery * (1 - phi)), Pc <= wind
#     ]
#     balance_constraints = [(Pd - Pc + Pw) <= P_line_max]
#     constraints = SOC_constraints + balance_constraints + W_constraints
#     revenue = cp.sum(cp.multiply(price, (Pd - Pc + Pw)))
#     p_curtail = kwf * cp.sum(Pwc)
#     p_degrate = kb * cp.sum(((Pc * eta_c + Pd / eta_d) * delta_t))
#     objective = cp.Maximize(revenue - p_curtail - p_degrate)

#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.MOSEK)
#     assert prob.status == "optimal"

#     return result

# # print(scaled_W)
# # ideal = wind_mkt(wind=scaled_W, price=price.values.flatten(), delta_t=1)
# # print(ideal)
# # W_pred = W + np.random.normal(100, 20, size=W.shape)
# # scaled_W += np.random.normal(20, 5, size=W.shape)
# # # scaled_W += np.random.normal(-10, 1, size=W.shape)

# # print("--" * 30)
# # print(scaled_W)
# # real = wind_mkt(wind=scaled_W, price=price.values.flatten(), delta_t=1)
# # print(real)
# # print("--" * 30)
# # print(ideal - real)
# if __name__ == "__main__":
#     dataset = "nrel"
#     # dataset = "australia"

#     with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
#         proposed_preds = pickle.load(handle)
#     with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
#         bench_raw_preds = pickle.load(handle)
#     with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
#         bench_bu_preds = pickle.load(handle)
#     with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
#         bench_opt_preds = pickle.load(handle)

#     all_opt_results = []
#     all_ipb_schedule = []
#     df_all_opt_results = []
#     # for i in range(1):
#     for i in range(len(proposed_preds)):
#         print(i, "/", len(proposed_preds) - 1)
#         proposed_model = {"Proposed": proposed_preds[i]}
#         all_preds = {
#             **proposed_model,
#             **bench_raw_preds[i],
#             **bench_bu_preds[i],
#             **bench_opt_preds[i]
#         }
#         opt_result = {name: {} for name in sorted(all_preds.keys())}
#         ipb_result = {name: {} for name in sorted(all_preds.keys())}

#         # loop for models
#         for name in sorted(all_preds.keys()):
#             print(name)
#             if name.__contains__("Persistence"):
#                 continue
#             all_term_preds = all_preds[name]

#             # loop for avg terms
#             for avg_terms in sorted(all_term_preds.keys()):
#                 # avg_terms = 12
#                 if name.__contains__("BU") and avg_terms == 1:
#                     ipb_result[name][avg_terms] = deepcopy(
#                         ipb_result[name.split("-")[0]][avg_terms])
#                     opt_result[name][avg_terms] = deepcopy(
#                         opt_result[name.split("-")[0]][avg_terms])
#                     print(f"skip {name} for {avg_terms} avg_terms")
#                     continue
#                 preds, labels = all_term_preds[avg_terms]
#                 preds = np.maximum(preds, 0)
#                 preds = np.minimum(preds, 16)
#                 preds *= 20
#                 labels *= 20

#                 avg_price = price.resample(
#                     f"{avg_terms * 5}min",
#                     closed="left").interpolate().values.flatten()[:288 //
#                                                                   avg_terms]
#                 # print(avg_price)

#                 avg_cost = 0
#                 count = 0
#                 start = 0
#                 ipb_samples, opt_cost = [], []
#                 # loop for samples
#                 for n in range(2, len(preds), 24):
#                     print(n, "/", len(preds))
#                     wind = preds[n].squeeze()
#                     true_wind = labels[n].squeeze()

#                     # DAED
#                     ideal_cost, ideal_p = wind_mkt(price=avg_price,
#                                                    wind=true_wind,
#                                                    delta_t=avg_terms / 12)

#                     # DAED
#                     daed_cost, daed_p = wind_mkt(price=avg_price,
#                                                  wind=wind,
#                                                  delta_t=avg_terms / 12)

#                     print("cost diff:")
#                     print(ideal_cost - daed_cost)
#                     print("ideal diff:")
#                     print(ideal_cost)
#                     ipb_samples.append(daed_p)
#                     opt_cost.append(np.abs(ideal_cost - daed_cost))

#                     avg_cost += np.abs(ideal_cost - daed_cost)
#                     count += 1
#                     start += 288 // avg_terms
#                     print("--" * 30)

#                 avg_cost /= count
#                 # opt_result[name][avg_terms] = avg_cost
#                 ipb_result[name][avg_terms] = deepcopy(ipb_samples)
#                 opt_result[name][avg_terms] = deepcopy(opt_cost)
#             # break
#             print("--" * 30)
#         # df_opt_result = pd.DataFrame(opt_result)
#         # df_all_opt_results.append(df_opt_result)
#         all_opt_results.append(opt_result)
#         all_ipb_schedule.append(ipb_result)

#     with open(f'savings/new_mix_mkt_opt_20_{dataset}.pickle', 'wb') as handle:
#         pickle.dump(all_opt_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     with open(f'savings/new_mix_mkt_opt_20_{dataset}_schedule.pickle',
#               'wb') as handle:
#         pickle.dump(all_ipb_schedule, handle, protocol=pickle.HIGHEST_PROTOCOL)

# import pickle
# import numpy as np

# import cvxpy as cp
# import pandas as pd
# from utils import setup_seed
# from copy import deepcopy
# import matplotlib.pyplot as plt
# # setup_seed(9)

# a = np.array([0.12, 0.17, 0.15, 0.19])
# b = np.array([14.8, 16.57, 15.55, 16.21])
# c = np.array([89, 83, 100, 70])

# eta_c = 0.95
# eta_d = 0.9
# SOC_ini = 0.5
# Cap = 300000
# degrade_price = 100
# shortage_price = 200
# surplus_price = 200
# P_battery = 40
# reserve = 0
# id_interval = 4

# N_g = 4
# # N_t = 24
# VWC = 100
# Pg_min = np.array([28, 20, 30, 20])
# Pg_max = np.array([200, 290, 190, 260])
# RU = np.array([40, 30, 30, 50])
# RD = np.array([40, 30, 30, 50])

# load = np.array([
#     510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, 754, 700, 686,
#     720, 714, 761, 727, 714, 618, 584, 578, 544
# ])

# load_scale_max = load.max() * 0.8
# load_scale_min = load.min() / 0.8

# # W = np.array([
# #     44.1, 48.5, 65.7, 144.9, 202.3, 317.3, 364.4, 317.3, 271, 306.9, 424.1,
# #     398, 487.6, 521.9, 541.3, 560, 486.8, 372.6, 367.4, 314.3, 316.6, 311.4,
# #     405.4, 470.4
# # ])

# # wind_scale_max = W.max()
# # wind_scale_min = W.min()


# def wind_post_process(preds, peneration_rate=0.8):
#     preds = np.where(preds < 0, 0, preds)
#     preds = np.where(preds > 16, 16, preds)
#     preds *= peneration_rate * 600 / 16
#     return preds


# def load_post_process(preds):
#     # print(preds.max())
#     # preds = (preds - preds.min()) / (preds.max() - preds.min()) * (
#     # load_scale_max - load_scale_min) + load_scale_min
#     return preds * 1e3


# def day_ahead(load, wind, delta_t):
#     N_t = len(load)
#     P = cp.Variable((N_g, N_t), name='P')
#     Pw = cp.Variable(N_t, name='Pw')
#     Pwc = cp.Variable(N_t, name='Pwc')

#     P_constraints = [P[:, t] <= Pg_max for t in range(N_t)] + [
#         P[:, t] >= Pg_min for t in range(N_t)
#     ] + [P[:, t + 1] - P[:, t] <= RU * delta_t for t in range(N_t - 1)
#          ] + [P[:, t] - P[:, t + 1] <= RD * delta_t for t in range(N_t - 1)]
#     W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]

#     balance_constraints = [
#         ((sum(P[:, t]) + Pw[t]) == (load[t])) for t in range(N_t)
#     ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]

#     constraints = P_constraints + balance_constraints + W_constraints
#     objective = cp.Minimize(
#         sum((a @ P**2 + b @ P + c.sum())) * delta_t + VWC * sum(Pwc) * delta_t)

#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     # print("-----test-----")
#     # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
#     # print((delta_t * VWC * sum(Pwc)).value)
#     # print("-----test-----")

#     return result, P.value


# def intra_day(load, wind, delta_t, da_schedule):
#     # print(da_schedule.sum(axis=0))
#     # print(load)
#     # print(wind)
#     N_t = len(load)

#     P = cp.Variable((N_g, N_t), name='P')
#     Pw = cp.Variable(N_t, name='Pw')
#     Pwc = cp.Variable(N_t, name='Pwc')
#     SOC = cp.Variable(N_t, name='SOC')
#     Pc = cp.Variable(N_t, name='Pc')
#     Pd = cp.Variable(N_t, name='Pd')
#     Pp = cp.Variable(N_t, name='Pp')
#     Pn = cp.Variable(N_t, name='Pn')

#     phi = cp.Variable(N_t, name='phi', boolean=True)
#     P_constraints = [P == da_schedule]
#     W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]
#     SOC_constraints = [
#         Cap * SOC[0] == (Cap * SOC_ini +
#                          (Pc[0] * eta_c - Pd[0] / eta_d) * delta_t)
#     ] + [
#         Cap * SOC[t] == (Cap * SOC[t - 1] +
#                          (Pc[t] * eta_c - Pd[t] / eta_d) * delta_t)
#         for t in range(1, N_t)
#     ] + [
#         SOC >= 0.05, SOC <= 0.95, Pc >= 0, Pd >= 0, Pc <=
#         (P_battery * phi), Pd <= (P_battery * (1 - phi)), Pp >= 0, Pn >= 0
#     ]
#     balance_constraints = [
#         ((sum(P[:, t]) + Pd[t] + Pw[t] + Pp[t]) == (load[t] + Pc[t] + Pn[t]))
#         for t in range(N_t)
#     ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)]
#     constraints = P_constraints + SOC_constraints + balance_constraints + W_constraints

#     objective = cp.Minimize(delta_t * sum((a @ P**2 + b @ P + c.sum())) +
#                             degrade_price * sum((
#                                 (Pc * eta_c + Pd / eta_d) * delta_t)) +
#                             delta_t * VWC * sum(Pwc) +
#                             shortage_price * sum(Pp) * delta_t +
#                             surplus_price * sum(Pn) * delta_t)

#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     # print("-----test-----")
#     # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
#     # print((degrade_price * sum((
#     #                             (Pc * eta_c + Pd / eta_d) * delta_t))).value)
#     # print((delta_t * VWC * sum(Pwc)).value)
#     # print((shortage_price * sum(Pp) * delta_t).value)
#     # print("-----test-----")
#     print((Pn.value * Pp.value).max())

#     # assert (np.zeros_like(Pn.value * Pp.value) == Pn.value * Pp.value).any()

#     return result, (P.value, SOC.value, Pc.value, Pd.value, Pwc.value,
#                     Pw.value)


# def real_time(load, wind, P_schedule, Pd_schedule, Pc_schedule, delta_t):
#     N_t = len(load)

#     P = cp.Variable((N_g, N_t), name='P')
#     Pw = cp.Variable(N_t, name='Pw')
#     Pwc = cp.Variable(N_t, name='Pwc')
#     Pp = cp.Variable(N_t, name='Pp')
#     Pn = cp.Variable(N_t, name='Pn')
#     # Pn =

#     P_constraints = [P == P_schedule]
#     W_constraints = [Pw <= wind, Pwc <= wind, Pw >= 0, Pwc >= 0]

#     balance_constraints = [
#         ((sum(P[:, t]) + Pd_schedule[t] + Pw[t] + Pp[t])
#          == (load[t] + Pc_schedule[t] + Pn[t])) for t in range(N_t)
#     ] + [Pw[t] + Pwc[t] == wind[t] for t in range(N_t)] + [Pp >= 0, Pn >= 0]
#     constraints = P_constraints + balance_constraints + W_constraints

#     objective = cp.Minimize(
#         delta_t * sum((a @ P**2 + b @ P + c.sum())) + degrade_price * sum((
#             (Pc_schedule * eta_c + Pd_schedule / eta_d) * delta_t)) +
#         delta_t * VWC * sum(Pwc) + delta_t * sum(shortage_price * Pp) +
#         surplus_price * sum(Pn) * delta_t)

#     prob = cp.Problem(objective, constraints)

#     result = prob.solve(solver=cp.GUROBI)
#     assert prob.status == "optimal"
#     # print("-----test-----")
#     # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
#     # print(degrade_price * sum((
#     #         (Pc_schedule * eta_c + Pd_schedule / eta_d) * delta_t)))
#     # print((delta_t * VWC * sum(Pwc)).value)
#     # print((delta_t * sum(shortage_price * Pp)).value)
#     # print("-----test-----")
#     print((Pn.value * Pp.value).max())

#     # assert (np.zeros_like(Pn.value * Pp.value) == Pn.value * Pp.value).any()

#     return result, (P.value, Pd_schedule, Pc_schedule, Pwc.value, Pw.value)


# if __name__ == "__main__":
#     # load wind
#     dataset = "nrel"
#     with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
#         proposed_wind_preds = pickle.load(handle)
#     with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
#         bench_wind_raw_preds = pickle.load(handle)
#     with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
#         bench_wind_bu_preds = pickle.load(handle)
#     with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
#         bench_wind_opt_preds = pickle.load(handle)

#     # load load
#     dataset = "mfred"
#     with open(f'savings/proposed_20_{dataset}.pickle', 'rb') as handle:
#         proposed_load_preds = pickle.load(handle)
#     with open(f'savings/bench_20_{dataset}.pickle', 'rb') as handle:
#         bench_load_raw_preds = pickle.load(handle)
#     with open(f'savings/bench_bu_20_{dataset}.pickle', 'rb') as handle:
#         bench_load_bu_preds = pickle.load(handle)
#     with open(f'savings/bench_opt_20_{dataset}.pickle', 'rb') as handle:
#         bench_load_opt_preds = pickle.load(handle)

#     all_seed_costs = []
#     for i in range(20):
#         # for i in range(len(proposed_wind_preds)):
#         print(i, "/", len(proposed_wind_preds) - 1)
#         proposed_wind_model = {"Proposed": proposed_wind_preds[i]}
#         all_wind_preds = {
#             **proposed_wind_model,
#             **bench_wind_raw_preds[i],
#             **bench_wind_bu_preds[i],
#             **bench_wind_opt_preds[i]
#         }
#         for key in ["Persistence-BU", "Persistence-OPT"]:
#             all_wind_preds.pop(key, None)

#         proposed_load_model = {"Proposed": proposed_load_preds[i]}
#         all_load_preds = {
#             **proposed_load_model,
#             **bench_load_raw_preds[i],
#             **bench_load_bu_preds[i],
#             **bench_load_opt_preds[i]
#         }
#         for key in ["Persistence-BU", "Persistence-OPT"]:
#             all_load_preds.pop(key, None)
#         # print(all_load_preds.keys())
#         # print(all_wind_preds.keys())
#         assert all_load_preds.keys() == all_wind_preds.keys()

#         opt_result = {name: {} for name in sorted(all_load_preds.keys())}
#         ipb_result = {name: {} for name in sorted(all_load_preds.keys())}
#         model_num = len(all_load_preds.keys()) + 1
#         names = list(sorted(all_load_preds.keys())) + ["Real"]
#         print(names)
#         # # loop for models
#         cost_results = np.zeros((model_num, model_num, 3))
#         for ll, load_name in enumerate(names):
#             print(load_name)
#             if load_name == "Persistence-BU" or load_name == "Persistence-OPT":
#                 continue

#             if load_name == "Real":
#                 load_preds = all_load_preds[names[0]]
#             else:
#                 load_preds = all_load_preds[load_name]
#             da_load_preds, da_load_labels = load_preds[12]
#             da_load_preds, da_load_labels = da_load_preds[23:], da_load_labels[
#                 23:]

#             # 5min preds
#             ipb_load_preds, ipb_load_labels = load_preds[1]
#             ipb_load_preds, ipb_load_labels = ipb_load_preds[
#                 23:], ipb_load_labels[23:]

#             # 15min preds
#             # ipb_load_preds, ipb_load_labels = load_preds[3]

#             da_load_preds = load_post_process(da_load_preds)
#             da_load_labels = load_post_process(da_load_labels)
#             ipb_load_preds = load_post_process(ipb_load_preds)
#             ipb_load_labels = load_post_process(ipb_load_labels)

#             for ww, wind_name in enumerate(names):
#                 print(wind_name)
#                 if wind_name == "Persistence-BU" or wind_name == "Persistence-OPT":
#                     continue

#                 if wind_name == "Real":
#                     wind_preds = all_wind_preds[names[0]]
#                 else:
#                     wind_preds = all_wind_preds[wind_name]
#                 da_wind_preds, da_wind_labels = wind_preds[12]
#                 da_wind_preds, da_wind_labels = da_wind_preds[
#                     2:], da_wind_labels[2:]
#                 # 5min preds
#                 ipb_wind_preds, ipb_wind_labels = wind_preds[1]
#                 ipb_wind_preds, ipb_wind_labels = ipb_wind_preds[
#                     2:], ipb_wind_labels[2:]

#                 # 15min preds
#                 # ipb_wind_preds, ipb_wind_labels = wind_preds[3]

#                 da_wind_preds = wind_post_process(da_wind_preds)
#                 da_wind_labels = wind_post_process(da_wind_labels)
#                 ipb_wind_preds = wind_post_process(ipb_wind_preds)
#                 ipb_wind_labels = wind_post_process(ipb_wind_labels)

#                 # if load_name == "Real":
#                 #     da_load_preds = da_load_labels
#                 #     ipb_load_preds = ipb_load_labels
#                 # if wind_name == "Real":
#                 #     da_wind_preds = da_wind_labels
#                 #     ipb_wind_preds = ipb_wind_labels

#                 avg_da_cost, avg_ipb_cost, avg_rt_cost = 0, 0, 0
#                 # day loop
#                 for i, day in enumerate(range(0, 7 * 24, 24)):
#                     da_load = da_load_preds[
#                         day] if load_name != "Real" else da_load_labels[day]
#                     da_wind = da_wind_preds[
#                         day] if wind_name != "Real" else da_wind_labels[day]
#                     # day ahead schedule
#                     da_cost, da_schedule = day_ahead(da_load.squeeze(),
#                                                      da_wind.squeeze(), 1)

#                     # interpolate(repeat)
#                     # da_schedule = np.repeat(da_schedule, 4, axis=1)
#                     da_schedule = np.repeat(da_schedule, 12, axis=1)

#                     # ipb schedule
#                     # THE SAME AS ROLLING, DECOUPLED

#                     ipb_wind = ipb_wind_preds if wind_name != "Real" else ipb_wind_labels
#                     ipb_load = ipb_load_preds if load_name != "Real" else ipb_load_labels

#                     ipb_wind_fcst = np.concatenate([
#                         ipb_wind[hour].squeeze()[:12 * id_interval]
#                         for hour in range(day, day + 24, id_interval)
#                     ])
#                     ipb_load_fcst = np.concatenate([
#                         ipb_load[hour].squeeze()[:12 * id_interval]
#                         for hour in range(day, day + 24, id_interval)
#                     ])

#                     ipb_cost, ipb_schedule = intra_day(ipb_load_fcst,
#                                                        ipb_wind_fcst,
#                                                        da_schedule=da_schedule,
#                                                        delta_t=1 / 12)
#                     (P, SOC, Pc, Pd, Pwc, Pw) = ipb_schedule
#                     rt_cost, rt_schedule = real_time(
#                         ipb_load_labels[day].squeeze(),
#                         ipb_wind_labels[day].squeeze(),
#                         P_schedule=P,
#                         Pd_schedule=Pd,
#                         Pc_schedule=Pc,
#                         delta_t=1 / 12)
#                     # rt_cost, rt_schedule = real_time(ipb_load,
#                     #                                  ipb_labels[day].squeeze(),
#                     #                                  P_schedule=P,
#                     #                                  Pd_schedule=Pd,
#                     #                                  Pc_schedule=Pc,
#                     #                                  delta_t=1 / 4)
#                     print(rt_cost)
#                     avg_da_cost += da_cost
#                     avg_ipb_cost += ipb_cost
#                     avg_rt_cost += rt_cost
#                     print("--" * 20)

#                 cost_results[ll, ww, 0] = avg_da_cost
#                 cost_results[ll, ww, 1] = avg_ipb_cost
#                 cost_results[ll, ww, 2] = avg_rt_cost

#         all_seed_costs.append(deepcopy(cost_results))

#         with open(f'savings/daid_pure_test_strict_pn.pickle', 'wb') as handle:
#             pickle.dump((all_seed_costs, sorted(all_load_preds.keys())),
#                         handle,
#                         protocol=pickle.HIGHEST_PROTOCOL)



import pickle
import numpy as np

import cvxpy as cp
import pandas as pd
from utils import setup_seed
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
# setup_seed(9)

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
id_interval = 4

N_g = 4
# N_t = 24
VWC = 200
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

# W = np.array([
#     44.1, 48.5, 65.7, 144.9, 202.3, 317.3, 364.4, 317.3, 271, 306.9, 424.1,
#     398, 487.6, 521.9, 541.3, 560, 486.8, 372.6, 367.4, 314.3, 316.6, 311.4,
#     405.4, 470.4
# ])

# wind_scale_max = W.max()
# wind_scale_min = W.min()


def wind_post_process(preds, peneration_rate=0.8):
    preds = np.where(preds < 0, 0, preds)
    preds = np.where(preds > 16, 16, preds)
    preds *= peneration_rate * 600 / 16
    return preds


def load_post_process(preds):
    # print(preds.max())
    # preds = (preds - preds.min()) / (preds.max() - preds.min()) * (
    # load_scale_max - load_scale_min) + load_scale_min
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
    # print("-----test-----")
    # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
    # print((delta_t * VWC * sum(Pwc)).value)
    # print("-----test-----")

    return result, P.value


def intra_day(load, wind, delta_t, da_schedule):
    # print(da_schedule.sum(axis=0))
    # print(load)
    # print(wind)
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
    # print("-----test-----")
    # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
    # print((degrade_price * sum((
    #                             (Pc * eta_c + Pd / eta_d) * delta_t))).value)
    # print((delta_t * VWC * sum(Pwc)).value)
    # print((shortage_price * sum(Pp) * delta_t).value)
    # print("-----test-----")
    # print((Pn.value * Pp.value).max())

    # assert (np.zeros_like(Pn.value * Pp.value) == Pn.value * Pp.value).any()

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
    # print("-----test-----")
    # print(delta_t * sum((a @ P**2 + b @ P + c.sum())).value)
    # print(degrade_price * sum((
    #         (Pc_schedule * eta_c + Pd_schedule / eta_d) * delta_t)))
    # print((delta_t * VWC * sum(Pwc)).value)
    # print((delta_t * sum(shortage_price * Pp)).value)
    # print("-----test-----")
    print((Pn.value * Pp.value).max())

    # assert (np.zeros_like(Pn.value * Pp.value) == Pn.value * Pp.value).any()

    return result, (P.value, Pd_schedule, Pc_schedule, Pwc.value, Pw.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run for multi-resolution schedules")
    parser.add_argument("--fixed_power", type=str, choices=["load","wind"])
    args = parser.parse_args()
    fixed_power = args.fixed_power

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
        print(all_load_preds.keys())
        print(all_wind_preds.keys())
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
            
            if fixed_power == "load":
                da_load_preds = da_load_labels
                ipb_load_preds = ipb_load_labels


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
                if fixed_power == "wind":
                    da_wind_preds = da_wind_labels
                    ipb_wind_preds = ipb_wind_labels
                da_wind_preds = wind_post_process(da_wind_preds)
                da_wind_labels = wind_post_process(da_wind_labels)
                ipb_wind_preds = wind_post_process(ipb_wind_preds)
                ipb_wind_labels = wind_post_process(ipb_wind_labels)

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
                    # print(ipb_cost)
                    (P, SOC, Pc, Pd, Pwc, Pw) = ipb_schedule

                    rt_cost, rt_schedule = real_time(
                        ipb_load_labels[day].squeeze(),
                        ipb_wind_labels[day].squeeze(),
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
                    # print(rt_cost)
                    avg_da_cost += da_cost
                    avg_ipb_cost += ipb_cost
                    avg_rt_cost += rt_cost
                    print("--" * 20)

                cost_results[ll, ww, 0] = avg_da_cost
                cost_results[ll, ww, 1] = avg_ipb_cost
                cost_results[ll, ww, 2] = avg_rt_cost
                if fixed_power == "wind":
                    break

            if fixed_power == "load":
                break
        all_seed_costs.append(deepcopy(cost_results))

        with open(f'savings/daid_strict_pn_fixed_{fixed_power}.pickle',
                  'wb') as handle:
            pickle.dump((all_seed_costs, sorted(all_load_preds.keys())),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
