import gurobipy as gp
import os
import glob
import pandas as pd
import math
import time
import csv
import datetime
import multiprocessing
from joblib import Parallel, delayed

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import numpy as np
from poisson_binomial import PoissonBinomial

opts = {'K': 1, 'T': 7, 'S': 24}

timeInMaint = {'generators': [1, 2], 'lines': [1, 2]}  # pred, corr

lineScen = {'threshold': 100, 'theta_mu': 15, 'theta_sigma': 5,
            'nu_mu': 3, 'nu_sigma': 0.3, 'e_sigma': 1}
genScen = {'threshold': 100, 'theta_mu': 20, 'theta_sigma': 10,
           'nu_mu': 5, 'nu_sigma': 0.3, 'e_sigma': 3}

sampleSize = 200
history = 100  # historical data

eps = 0.10
divLine = 20
divGen = 10


def evaluateChanceConstr(generators, genValues, lines, lineValues):
    probs_Gen = []
    # find probs of failures before (or on) the maintenance day P (zeta <= t)
    rho_Gen = max(1, int(len(generators) / divGen))
    for gen in generators:
        if gen.inSubset == 1:
            t = min(genValues[gen.listIndex].index(1) + 1, gen.T)
            probs_Gen.append(stats.invgauss.cdf(
                t, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape))
        else:
            probs_Gen.append(stats.invgauss.cdf(
                gen.T, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape))

    probs_Line = []
    rho_Line = max(1, int(len(lines) / divLine))
    for line in lines:
        if line.inSubset == 1:
            t = min(lineValues[line.listIndex].index(1) + 1, line.T)
            probs_Line.append(stats.invgauss.cdf(
                t, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape))
        else:
            probs_Line.append(stats.invgauss.cdf(
                line.T, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape))

    # create a PB random variable with probs
    PB_Gen = PoissonBinomial(probs_Gen)
    PB_Line = PoissonBinomial(probs_Line)

    if PB_Gen.x_or_less(rho_Gen) * ((1 - includeLine) + includeLine * PB_Line.x_or_less(rho_Line)) >= 1 - eps:
        flag = 'feasible'
    else:
        flag = 'infeasible'

    return flag


def getSignalData(n, theta_mu, theta_sigma, nu_mu, nu_sigma, e_sigma, threshold):
    time = np.array([t for t in range(n)])

    # initial amplitude
    theta = np.random.normal(theta_mu, theta_sigma)

    # drift parameter
    nu = np.random.normal(nu_mu, nu_sigma)

    # Brownian motion
    W = np.array([np.random.normal(0, t ** (0.5)) for t in time])

    # Signal at time t
    S = np.array([theta + nu * t + e_sigma * W[t] for t in range(n)])

    failureTime = None
    for t in range(n):
        if S[t] > threshold:
            failureTime = t
            break
        else:
            continue

    if failureTime == None:
        failureTime = n - 1

    # Increments
    increments = [S[0]]
    for t in range(1, len(S)):
        increments.append(S[t] - S[t - 1])

    return S, failureTime, increments, int(theta)


def getIncrements(signalData):
    # Increments
    E = [signalData[0]]
    for t in range(1, len(signalData)):
        E.append(signalData[t] - signalData[t - 1])

    return np.array(E)


def create(t, n, threshold, theta_mu, theta_sigma, nu_mu, nu_sigma, e_sigma):
    data = [{'degrSignal': None, 'failureTime': None, 'incrSignal': None}
            for i in range(t)]

    for i in range(t):
        data[i]['degrSignal'], data[i]['failureTime'], data[i]['incrSignal'], data[i][
            'initialAmplitude'] = getSignalData(n, theta_mu, theta_sigma, nu_mu,
                                                nu_sigma, e_sigma, threshold)

    return data


def Priors(data):
    mu = np.mean([data[i]['degrSignal'][0] for i in range(len(data))])

    mu_ = np.mean([(data[i]['degrSignal'][data[i]['failureTime']] - data[i]
    ['degrSignal'][0]) / (data[i]['failureTime'] + 1) for i in range(len(data))])

    return mu, mu_


def Posteriors(degrSignal, priorTheta_mu, priorTheta_var, priorNu_mu, priorNu_var, e_sigma, t1, tk):
    numeratorMean = (degrSignal[t1] * priorTheta_var + priorTheta_mu * (e_sigma ** 2) * t1) * (
                priorNu_var * tk + e_sigma ** 2) \
                    - priorTheta_var * t1 * \
                    (priorNu_var *
                     (degrSignal[tk] + priorNu_mu * e_sigma ** 2))

    denominatorMean = (priorTheta_var + e_sigma ** 2 * t1) * \
                      (priorNu_var * tk + e_sigma ** 2) - \
                      priorTheta_var * priorNu_var * t1

    numeratorMean_ = (priorNu_var * degrSignal[tk] + priorNu_mu * (e_sigma ** 2)) * (
                priorTheta_var + (e_sigma ** 2) * t1) \
                     - priorNu_var * (degrSignal[t1] * priorTheta_var +
                                      priorTheta_mu * (e_sigma ** 2) * t1)

    denominatorMean_ = (priorTheta_var + (e_sigma ** 2) * t1) * (priorNu_var * tk + e_sigma ** 2) \
                       - priorTheta_var * priorNu_var * t1

    return numeratorMean / denominatorMean, priorTheta_var, numeratorMean_ / denominatorMean_, priorNu_var


def RemainingCDF(compSignal, threshold, priorTheta_mu, priorTheta_var, priorNu_mu, priorNu_var, e_sigma, plot=0):
    s = compSignal

    obsUpper = (threshold - priorTheta_mu) / \
               (priorNu_mu + 3 * (priorNu_var ** 0.5))

    observedTime = 1e5
    while observedTime > s['failureTime'] - 1:
        observedTime = np.random.randint(0, obsUpper)

    posteriorTheta_mu, posteriorTheta_var, posteriorNu_mu, posteriorNu_var \
        = Posteriors(s['degrSignal'], priorTheta_mu, priorTheta_var, priorNu_mu, priorNu_var, e_sigma, 1, observedTime)

    mean = (threshold - compSignal['degrSignal']
    [observedTime]) / posteriorNu_mu

    shape = ((threshold - compSignal['degrSignal']
    [observedTime]) ** 2) / e_sigma ** 2

    fig = None
    if plot == 1:
        rv = stats.invgauss.rvs(mean / shape, 0, shape, size=10000)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(rv, 1000, density=True, histtype='step', cumulative=True)
        ax.set_xlabel('Remaining lifetime')
        ax.set_ylabel('Cumulative Distribution Function')
        # plt.show()
        plt.close()

    return mean, shape, observedTime, fig


class Bus:

    def __init__(self, data):

        # K: #scenarios (equiprobable), T: #days, S: #hours
        self.K, self.T, self.S, self.barT = set_horizon(opts)

        self.index = int(data[0])

        self.deltaMax = math.pi
        self.deltaMin = -self.deltaMax

        self.curtailmentCost = 100  # 100*0.1 # 100*10 # 100

        self.file_demand = float(data[2]) / 100
        real_data = Demand('weekly-demand.csv')
        # 7 days - 24 hours data
        scaled_demand = Rescale(real_data, self.file_demand, self.file_demand / 2)

        self.demand = {}
        for t in range(self.T):
            self.demand[t] = [0] * self.S
            for s in range(self.S):
                if (float(data[2]) / 100) != 0:
                    self.demand[t][s] = scaled_demand[t][s]

        # all assigned later
        self.outgoing_arcs, self.incoming_arcs, self.generators = [], [], []
        self.listIndex = None
        self.delta = None
        self.q = None
        self.C_q = None
        self.dMax = {t: [] for t in range(self.T)}

    def add_operation_variables(self, model, solveBy, scenNo=-1, dayNo=-1):

        self.rangeK, self.rangeT, self.rangeS = method(solveBy, scenNo, dayNo)
        self.solveBy = solveBy

        # Voltage angle (delta)
        self.delta = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=self.deltaMin, ub=self.deltaMax,
                                   vtype=gp.GRB.CONTINUOUS,
                                   name=[
                                       'delta' + str(self.listIndex) + '[' + str(i) + ',' + str(j) + ',' + str(k) + ']'
                                       for i in self.rangeK for j in
                                       self.rangeT for k in self.rangeS])

        # Demand curtailment (q)
        self.q = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                               name=['q' + str(self.listIndex) + '[' + str(i) + ',' + str(j) + ',' + str(k) + ']' for i
                                     in self.rangeK for j in self.rangeT for
                                     k in self.rangeS])

    def add_operation_constraints(self, model):

        # Flow balance
        model.addConstrs(gp.quicksum(gen.p[k, t, s] for gen in self.generators) + self.q[k, t, s] - self.demand[t][s]
                         == gp.quicksum(outgoing.f[k, t, s] for outgoing in self.outgoing_arcs)
                         - gp.quicksum(incoming.f[k, t, s] for incoming in self.incoming_arcs) for k in self.rangeK for
                         t in self.rangeT for s in self.rangeS)

    def add_operation_costs(self, scenNo):

        # Demand curtailment
        self.operation_cost = gp.quicksum(
            self.curtailmentCost * 100 * self.q[scenNo, t, s] for t in self.rangeT for s in self.rangeS)

        return self.operation_cost


class Line:

    def __init__(self, data, buses, generators):

        # K: #scenarios (equiprobable), T: #days, S: #hours
        self.K, self.T, self.S, self.barT = set_horizon(opts)

        # not in L'
        self.inSubset = 0

        # maintenance length in periods
        self.predTime, self.corrTime = timeInMaint['lines'][0], timeInMaint['lines'][1]

        self.f_bar = float(data[5]) / 100  # rate_A

        self.f_barUB = {
            t: {s: self.f_bar for s in range(self.S)} for t in range(self.T)}
        self.f_barLB = {
            t: {s: -self.f_bar for s in range(self.S)} for t in range(self.T)}

        self.B = 1 / float(data[3])  # susceptance

        self.switch_cost = 0

        self.tail, self.head = int(data[0]), int(data[1])  # (i, j)

        # node i of arc (i, j)
        for i in range(len(buses)):
            if buses[i].index == self.tail:
                buses[i].outgoing_arcs.append(self)
                self.from_bus = buses[i]
                break

        # node j of arc (i, j)
        for i in range(len(buses)):
            if buses[i].index == self.head:
                buses[i].incoming_arcs.append(self)
                self.to_bus = buses[i]
                break

        self.M = self.B * (self.from_bus.deltaMax - self.to_bus.deltaMin)

        self.fix_pred_cost = 0.1 * \
                             sum(gen.fix_pred_cost for gen in generators) / len(generators)
        self.corr_cost = 0.1 * \
                         sum(gen.corr_cost for gen in generators) / len(generators)

        # all assigned later
        self.statusList = []
        for k in range(self.K):
            self.statusList.append([])
        self.upperBounds = {}
        self.listIndex = None
        self.z = None
        self.f = None
        self.y = None
        self.zeta = None
        self.C_p = None
        self.C_c = None
        self.C_y = None
        self.total = None

    def add_maintenance_variables(self, model, solveAs):

        varType = program(solveAs)

        # Maintenance decision (z)
        self.z = model.addVars(
            self.barT, lb=0, ub=1, vtype=varType, name='z_(' + str(self.listIndex) + ').')

    def add_operation_variables(self, model, solveAs, solveBy, scenNo=-1, dayNo=-1):

        self.rangeK, self.rangeT, self.rangeS = method(solveBy, scenNo, dayNo)
        varType = program(solveAs)

        # Power flow (f)
        self.f = model.addVars(self.rangeK, self.rangeT, self.rangeS, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY,
                               ub=gp.GRB.INFINITY,
                               name='f_(' + str(self.listIndex) + ').')

        if self.inSubset == 1:
            # Switch status (y)
            self.y = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=0,
                                   ub=1, vtype=varType, name='y_(' + str(self.listIndex) + ').')

    def add_maintenance_constraints(self, model):
        # At most 1 maintenance
        model.addConstr(self.z.sum('*') == 1)

    def add_type_maintenance_constraints(self, model):

        # Under predictive maintenance
        for k in self.rangeK:
            for t in range(min(self.zeta[k] + self.predTime, self.T)):
                from_ = max(0, t - self.predTime + 1)
                to_ = t + 1
                model.addConstrs(self.y[k, t, s] <= 1 - gp.quicksum(self.z[e]
                                                                    for e in range(from_, to_)) for s in self.rangeS)

        # Under corrective maintenance
        model.addConstrs(self.y[k, t, s] <= gp.quicksum(self.z[t_] for t_ in range(self.zeta[k])) for k in self.rangeK
                         for t in range(self.zeta[k], min(self.zeta[k] + self.corrTime, self.T)) for s in self.rangeS)

    def add_operation_constraints(self, model):

        if self.f_bar != 9900 / 100:
            for t in self.rangeT:
                for s in self.rangeS:
                    if self.f_barUB[t][s] != 9900 / 100:
                        # Upper bound on power flow
                        model.addConstrs(
                            self.f_barUB[t][s] >= self.f[k, t, s] for k in self.rangeK)
                    if self.f_barLB[t][s] != - 9900 / 100:
                        # Lower bound on power flow
                        model.addConstrs(
                            self.f_barLB[t][s] <= self.f[k, t, s] for k in self.rangeK)

        if self.inSubset == 1:
            # Upper bound - balance
            model.addConstrs(
                self.B * (self.from_bus.delta[k, t, s] - self.to_bus.delta[k, t, s]) + self.M * (1 - self.y[k, t, s])
                >= self.f[k, t, s] for k in self.rangeK for t in self.rangeT for s in self.rangeS)

            # Lower bound -  balance
            model.addConstrs(
                self.B * (self.from_bus.delta[k, t, s] - self.to_bus.delta[k, t, s]) - self.M * (1 - self.y[k, t, s])
                <= self.f[k, t, s] for k in self.rangeK for t in self.rangeT for s in self.rangeS)

            # Upper bound on power flow
            model.addConstrs(self.f_barUB[t][s] * self.y[k, t, s] >= self.f[k, t, s]
                             for k in self.rangeK for t in self.rangeT for s in self.rangeS)

            # Lower bound on power flow
            model.addConstrs(self.f_barLB[t][s] * self.y[k, t, s] <= self.f[k, t, s]
                             for k in self.rangeK for t in self.rangeT for s in self.rangeS)

        if self.inSubset == 0:
            # flow definition
            model.addConstrs(self.B * (self.from_bus.delta[k, t, s] - self.to_bus.delta[k, t, s]) == self.f[k, t, s]
                             for k in self.rangeK for t in self.rangeT for s in self.rangeS)

    def add_maintenance_costs(self, scenNo, type='Dynamic'):

        if type == 'Dynamic':
            self.C_p = gp.quicksum(
                self.dynamic_pred_cost[t] * self.z[t] for t in range(self.zeta[scenNo]))
            self.C_c = gp.quicksum(self.dynamic_corr_cost * self.z[t] for t in range(
                self.zeta[scenNo], self.T + 1) if self.zeta[scenNo] != self.T)

        elif type == 'Fixed':
            self.C_p = gp.quicksum(
                self.fix_pred_cost * self.z[t] for t in range(self.zeta[scenNo]))
            self.C_c = gp.quicksum(self.corr_cost * self.z[t] for t in range(
                self.zeta[scenNo], self.T + 1) if self.zeta[scenNo] != self.T)

        self.maintenance_cost = self.C_p + self.C_c

        return self.maintenance_cost

    def resetScenarioUpperBounds(self, scenNo):
        if self.inSubset == 1:
            self.upperBounds[scenNo] = [1] * self.barT


class Generator:

    def __init__(self, data, buses, costs):

        # K: #scenarios (equiprobable), T: #days, S: #hours
        self.K, self.T, self.S, self.barT = set_horizon(opts)

        # not in G'
        self.inSubset = 0

        # maintenance length in periods
        self.predTime, self.corrTime = timeInMaint['generators'][0], timeInMaint['generators'][1]

        # bus index
        self.index = int(data[0])

        self.Pmax, self.Pmin = float(data[-13]) / 100, float(data[-12]) / 100


        self.RU, self.RD = self.Pmax, -self.Pmax

        self.MU, self.MD = 1, 1

        # from m. file
        self.Cost = costs

        self.fix_pred_cost = (self.Pmax * self.Pmax * self.Cost.quad *
                              (100) ** 2 + self.Pmax * self.Cost.lin * 100) * self.S

        self.corr_cost = 3 * self.fix_pred_cost  # 10 * self.fix_pred_cost # 3 * self.fix_pred_cost   #

        self.commit_cost = self.Cost.fixed

        # buses, at the same node with the generator
        for i in range(len(buses)):
            if buses[i].index == self.index:
                buses[i].generators.append(self)
                break

        # all assigned later
        self.statusList = []
        for k in range(self.K):
            self.statusList.append([])

        self.upperBounds = {}
        self.listIndex = None
        self.w = None
        self.p = None
        self.x = None
        self.u = None
        self.v = None
        self.C_p = None
        self.C_c = None
        self.C_prod = None
        self.C_x = None
        self.C_u = None
        self.C_v = None
        self.total = None
        self.zeta = None

    def add_maintenance_variables(self, model, solveAs):

        varType = program(solveAs)

        # Maintenance decision (w)
        self.w = model.addVars(
            self.barT, lb=0, ub=1, vtype=varType, name='w_(' + str(self.listIndex) + ').')

    def add_operation_variables(self, model, solveAs, solveBy, scenNo=-1, dayNo=-1):

        self.rangeK, self.rangeT, self.rangeS = method(solveBy, scenNo, dayNo)
        varType = program(solveAs)

        # Power generation (p)
        self.p = model.addVars(self.rangeK, self.rangeT, self.rangeS,
                               vtype=gp.GRB.CONTINUOUS, name='p_(' + str(self.listIndex) + ').')

        # Commitment status (x)
        self.x = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=0,
                               ub=1, vtype=varType, name='x_(' + str(self.listIndex) + ').')

        # Start up (u)
        self.u = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=0,
                               ub=1, vtype=gp.GRB.CONTINUOUS, name='u_(' + str(self.listIndex) + ').')

        # Shut down (v)
        self.v = model.addVars(self.rangeK, self.rangeT, self.rangeS, lb=0,
                               ub=1, vtype=varType, name='v_(' + str(self.listIndex) + ').')

    def add_maintenance_constraints(self, model):
        # At most 1 maintenance
        model.addConstr(self.w.sum('*') == 1)

    def add_type_maintenance_constraints(self, model):

        # Under predictive maintenance
        for k in self.rangeK:
            for t in range(min(self.zeta[k] + self.predTime, self.T)):
                from_ = max(0, t - self.predTime + 1)
                to_ = t + 1
                model.addConstrs(self.x[k, t, s] <= 1 - gp.quicksum(self.w[e]
                                                                    for e in range(from_, to_)) for s in self.rangeS)

        # Under corrective maintenance
        model.addConstrs(self.x[k, t, s] <= gp.quicksum(self.w[t_] for t_ in range(self.zeta[k])) for k in self.rangeK
                         for t in range(self.zeta[k], min(self.zeta[k] + self.corrTime, self.T)) for s in self.rangeS)

    def add_operation_constraints(self, model):

        # Upper bound on power generation
        model.addConstrs(self.Pmax * self.x[k, t, s] >= self.p[k, t, s]
                         for k in self.rangeK for t in self.rangeT for s in self.rangeS)

        # Lower bound on power generation
        model.addConstrs(self.Pmin * self.x[k, t, s] <= self.p[k, t, s]
                         for k in self.rangeK for t in self.rangeT for s in self.rangeS)

        # Start-up
        model.addConstrs(self.x[k, t, s - 1] - self.x[k, t, s] + self.u[k, t, s]
                         >= 0 for k in self.rangeK for t in self.rangeT for s in range(1, self.S))

        # Shut-down
        model.addConstrs(self.x[k, t, s] - self.x[k, t, s - 1] + self.v[k, t, s]
                         >= 0 for k in self.rangeK for t in self.rangeT for s in range(1, self.S))

        # Ramp-up
        model.addConstrs(self.RU >= self.p[k, t, s] - self.p[k, t, s - 1]
                         for k in self.rangeK for t in self.rangeT for s in range(1, self.S))

        # Ramp-down
        model.addConstrs(self.RD <= self.p[k, t, s] - self.p[k, t, s - 1]
                         for k in self.rangeK for t in self.rangeT for s in range(1, self.S))

        # Min-up
        model.addConstrs(
            self.x[k, t, s] - self.x[k, t, s - 1] <= self.x[k, t, s_] for k in self.rangeK for t in self.rangeT for s in
            range(1, self.S) for s_ in
            range(s + 1, min(s + self.MU, self.S)))

        # Min-down
        model.addConstrs(
            self.x[k, t, s - 1] - self.x[k, t, s] <= 1 - self.x[k, t, s_] for k in self.rangeK for t in self.rangeT for
            s in range(1, self.S) for s_ in
            range(s + 1, min(s + self.MD, self.S)))

    def add_maintenance_costs(self, scenNo, type='Dynamic'):

        if type == 'Dynamic':
            self.C_p = gp.quicksum(
                self.dynamic_pred_cost[t] * self.w[t] for t in range(self.zeta[scenNo]))

            self.C_c = gp.quicksum(self.dynamic_corr_cost * self.w[t] for t in range(
                self.zeta[scenNo], self.T + 1) if self.zeta[scenNo] != self.T)

        if type == 'Fixed':
            self.C_p = gp.quicksum(
                self.fix_pred_cost * self.w[t] for t in range(self.zeta[scenNo]))

            self.C_c = gp.quicksum(self.corr_cost * self.w[t] for t in range(
                self.zeta[scenNo], self.T + 1) if self.zeta[scenNo] != self.T)

        self.maintenance_cost = self.C_p + self.C_c

        return self.maintenance_cost

    def add_operation_costs(self, scenNo):

        # Power generation
        self.C_prod = gp.quicksum(self.Cost.quad * (100 ** 2) * self.p[scenNo, t, s] * self.p[scenNo, t, s]
                                  + self.Cost.lin * 100 * self.p[scenNo, t, s] for t in self.rangeT for s in
                                  self.rangeS)

        # Commitment status
        self.C_x = gp.quicksum(
            self.commit_cost * self.x[scenNo, t, s] for t in self.rangeT for s in self.rangeS)

        # Start up
        self.C_u = gp.quicksum(
            self.Cost.startup * self.u[scenNo, t, s] for t in self.rangeT for s in self.rangeS)

        # Shut down
        self.C_v = gp.quicksum(
            self.Cost.shutdown * self.v[scenNo, t, s] for t in self.rangeT for s in self.rangeS)

        self.operation_cost = self.C_prod + self.C_x + self.C_u + self.C_v

        return self.operation_cost

    def resetScenarioUpperBounds(self, scenNo):
        if self.inSubset == 1:
            self.upperBounds[scenNo] = [1] * self.barT


class Costs:

    def __init__(self, data):

        self.quad = 0  # float(data[4])

        # lin > 1 - fixed > 100 - startup > 1000 // (same if exists in m.file)

        self.lin = float(data[5])
        power = 1
        while self.lin < 1:
            self.lin = self.lin * 10 ** power
            power += 1

        self.fixed = float((data[6].split(';'))[0])
        power = 0
        while self.fixed < 100:
            self.fixed = self.lin * 10 ** power
            power += 1

        self.startup = float(data[1])
        power = 0
        while self.startup < 1000:
            self.startup = self.lin * 10 ** power
            power += 1

        self.shutdown = 0


class MasterProblem:

    def __init__(self, lines, generators, L, solveAs, solveBy, case):

        # K: #scenarios (equiprobable), T: #days, S: #hours
        self.K, self.T, self.S, self.barT = set_horizon(opts)
        self.prob = scenario_prob()
        self.solveAs = solveAs
        self.solveBy = solveBy
        self.case = case

        self.rangeK = range(0, self.K)
        self.rangeT = range(0, 1)
        if solveBy == 'KT':
            self.rangeT = range(0, self.T)

        self.lines = lines
        self.generators = generators

        self.model = gp.Model('Master-Problem')
        self.model.setParam('OutputFlag', 0)

        for gen in generators:
            gen.add_maintenance_variables(self.model, self.solveAs)

        for line in lines:
            line.add_maintenance_variables(self.model, self.solveAs)

        self.theta = self.model.addVars(self.rangeK, self.rangeT, name='theta')

        for gen in generators:
            gen.add_maintenance_constraints(self.model)

        for line in lines:
            line.add_maintenance_constraints(self.model)

        if solveBy == 'KT':
            self.model.addConstrs(self.theta[k, t] >= L[k][t]
                                  for k in self.rangeK for t in self.rangeT)

        elif solveBy == 'K':
            self.model.addConstrs(self.theta.sum(
                k, '*') >= sum(L[k]) for k in self.rangeK)

        if withDynamic == True:
            type = 'Dynamic'
        if withFixed == True:
            type = 'Fixed'

        self.gen_maintenance_cost = gp.quicksum(self.prob[k] * gen.add_maintenance_costs(
            k, type) for k in self.rangeK for gen in generators if gen.inSubset == 1)

        self.line_maintenance_cost = gp.quicksum(self.prob[k] * line.add_maintenance_costs(
            k, type) for k in self.rangeK for line in lines if line.inSubset == 1)

        self.theta_cost = gp.quicksum(
            self.prob[k] * self.theta[k, t] for k in self.rangeK for t in self.rangeT)

        self.totalCost = (self.gen_maintenance_cost +
                          self.line_maintenance_cost) + (self.theta_cost)

        self.model.setObjective(self.totalCost, sense=gp.GRB.MINIMIZE)

        self.model.update()

    def optimize(self):


        self.model.setParam('MIPGap', mipgap)
        self.model.setParam('Threads', num_cores)
        self.model.optimize()

        self.opt = self.model.objBound

        self.gen_cost_value = self.gen_maintenance_cost.getValue()
        self.line_cost_value = self.line_maintenance_cost.getValue()
        self.theta_cost_value = self.theta_cost.getValue()

    def store_first_stage(self):

        self.lineValues = {}
        for line in self.lines:
            if line.inSubset == 1:
                self.lineValues[line.listIndex] = [0] * self.barT
                for t in range(self.barT):
                    self.lineValues[line.listIndex][t] = int(
                        self.lines[line.listIndex].z[t].x + 0.5)

        self.genValues = {}
        for gen in self.generators:
            if gen.inSubset == 1:
                self.genValues[gen.listIndex] = [0] * self.barT
                for t in range(self.barT):
                    self.genValues[gen.listIndex][t] = int(
                        self.generators[gen.listIndex].w[t].x + 0.5)

        return self.lineValues, self.genValues

    def set_hatT(self, lineValues, genValues, scenNo):

        # assign hatT indices
        for line in self.lines:
            if line.inSubset == 1:

                line.hatT = []

                if lineValues[line.listIndex].index(1) < line.zeta[scenNo]:
                    # under PREDICTIVE maintenance - singleton
                    line.hatT.append(lineValues[line.listIndex].index(1))
                else:
                    if line.zeta != line.T:
                        # under CORRECTIVE maintenance - singleton / list
                        for t in range(line.zeta[scenNo], line.barT):
                            line.hatT.append(t)
                    else:
                        # no failure occurs case
                        line.hatT.append(line.zeta)


        # assign hatT indices
        for gen in self.generators:
            if gen.inSubset == 1:

                gen.hatT = []

                if genValues[gen.listIndex].index(1) < gen.zeta[scenNo]:
                    # under PREDICTIVE maintenance - singleton
                    gen.hatT.append(genValues[gen.listIndex].index(1))
                else:
                    if gen.zeta != gen.T:
                        # under CORRECTIVE maintenance - singleton / list
                        for t in range(gen.zeta[scenNo], gen.barT):
                            gen.hatT.append(t)
                    else:
                        gen.hatT.append(gen.zeta)


    def set_hatTt(self, scenNo, dayNo):

        for gen in self.generators:
            if gen.inSubset == 1:
                gen.hatTt = []
                for i in range(gen.barT):
                    if gen.statusList[scenNo][i][dayNo] == gen.upperBounds[scenNo][dayNo]:
                        gen.hatTt.append(i)

        for line in self.lines:
            if line.inSubset == 1:
                line.hatTt = []
                for i in range(line.barT):
                    if line.statusList[scenNo][i][dayNo] == line.upperBounds[scenNo][dayNo]:
                        line.hatTt.append(i)

    def add_cut(self, lineValues, genValues, Q, L):

        self.expr1 = gp.quicksum((line.z[t] - 1) for t in range(self.barT)
                                 for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] > 0.5)
        self.expr2 = gp.quicksum(line.z[t] for t in range(
            self.barT) for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] < 0.5)

        self.expr3 = gp.quicksum((gen.w[t] - 1) for t in range(self.barT)
                                 for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] > 0.5)
        self.expr4 = gp.quicksum(gen.w[t] for t in range(
            self.barT) for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] < 0.5)

        if self.case == 'std-L':
            sumQ = sum(sum(val) for val in Q)
            sumL = sum(sum(val) for val in L)
            self.model.addConstr(self.theta.sum('*', '*') >= (sumQ - sumL) * gp.LinExpr(
                (self.expr1 + self.expr3) - (self.expr2 + self.expr4)) + sumQ)

        if self.case == 'std-LK':
            for k in self.rangeK:
                sumQ = sum(Q[k])
                sumL = sum(L[k])
                self.model.addConstr(self.theta.sum(k, '*') >= (sumQ - sumL) * gp.LinExpr(
                    (self.expr1 + self.expr3) - (self.expr2 + self.expr4)) + sumQ)

        if self.case == 'std-LKT':
            for k in self.rangeK:
                for t in self.rangeT:
                    sumQ = Q[k][t]
                    sumL = L[k][t]
                    self.model.addConstr(self.theta.sum(k, t) >= (
                            sumQ - sumL) * gp.LinExpr((self.expr1 + self.expr3) - (self.expr2 + self.expr4)) + sumQ)

        if self.case == 'K+':
            self.expr1 = gp.quicksum(line.z[t] for t in range(
                self.barT) for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] > 0.5)
            self.expr2 = gp.quicksum(gen.w[t] for t in range(
                self.barT) for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] > 0.5)

            h = len(lineValues) + len(genValues)

            for k in self.rangeK:
                sumQ = sum(Q[k])
                sumL = sum(L[k])
                self.model.addConstr(self.theta.sum(
                    k, '*') >= (sumQ - sumL) * gp.LinExpr((self.expr1 + self.expr2) - h) + sumQ)

        if self.case == 'single_K+':
            self.expr1 = gp.quicksum(line.z[t] for t in range(
                self.barT) for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] > 0.5)
            self.expr2 = gp.quicksum(gen.w[t] for t in range(
                self.barT) for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] > 0.5)

            h = len(lineValues) + len(genValues)
            sumL = sum(sum(val) for val in L)
            sumQ = sum(sum(val) for val in Q)

            self.model.addConstr(self.theta.sum(
                '*', '*') >= (sumQ - sumL) * gp.LinExpr((self.expr1 + self.expr2) - h) + sumQ)

        if self.case == 'KT+':
            self.expr1 = gp.quicksum(line.z[t] for t in range(
                self.barT) for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] > 0.5)
            self.expr2 = gp.quicksum(gen.w[t] for t in range(
                self.barT) for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] > 0.5)

            h = len(lineValues) + len(genValues)

            for k in self.rangeK:
                for t in self.rangeT:
                    sumQ = Q[k][t]
                    sumL = L[k][t]

                    self.model.addConstr(self.theta[k, t] >= (
                            sumQ - sumL) * gp.LinExpr((self.expr1 + self.expr2) - h) + sumQ)

        if self.case == 'single_KT+':
            self.expr1 = gp.quicksum(line.z[t] for t in range(
                self.barT) for line in self.lines if line.inSubset == 1 if lineValues[line.listIndex][t] > 0.5)
            self.expr2 = gp.quicksum(gen.w[t] for t in range(
                self.barT) for gen in self.generators if gen.inSubset == 1 if genValues[gen.listIndex][t] > 0.5)

            h = len(lineValues) + len(genValues)
            sumQ = sum(sum(val) for val in Q)
            sumL = sum(sum(val) for val in L)

            self.model.addConstr(self.theta.sum(
                '*', '*') >= (sumQ - sumL) * gp.LinExpr((self.expr1 + self.expr2) - h) + sumQ)

        # hatT cut: (8) or (11)
        if self.case == 'K++' or self.case == 'KT++':
            for k in self.rangeK:
                self.set_hatT(lineValues, genValues, k)
                self.expr1 = gp.quicksum(
                    line.z[t] for line in self.lines if line.inSubset == 1 for t in line.hatT) - len(lineValues)
                self.expr2 = gp.quicksum(
                    gen.w[t] for gen in self.generators if gen.inSubset == 1 for t in gen.hatT) - len(genValues)

                if self.case == 'K++':
                    sumQ = sum(Q[k])
                    sumL = sum(L[k])

                self.model.addConstr(self.theta.sum(
                    k, '*') >= (sumQ - sumL) * (self.expr1 + self.expr2) + sumQ)

        if self.case == 'single_K++':
            self.expr3 = 0
            for k in self.rangeK:
                self.set_hatT(lineValues, genValues, k)
                self.expr1 = gp.quicksum(
                    line.z[t] for line in self.lines if line.inSubset == 1 for t in line.hatT) - len(lineValues)
                self.expr2 = gp.quicksum(
                    gen.w[t] for gen in self.generators if gen.inSubset == 1 for t in gen.hatT) - len(genValues)

                sumQ = sum(Q[k])
                sumL = sum(L[k])

                self.expr3 += (sumQ - sumL) * (self.expr1 + self.expr2)

            sumQ = sum(sum(val) for val in Q)

            self.model.addConstr(self.theta.sum(
                '*', '*') >= (self.expr3) + sumQ)

        if self.case == 'KT++T' or self.case == 'combine-KT':
            for k in self.rangeK:
                self.set_hatT(lineValues, genValues, k)
                self.expr1 = gp.quicksum(
                    line.z[t] for line in self.lines if line.inSubset == 1 for t in line.hatT) - len(lineValues)
                self.expr2 = gp.quicksum(
                    gen.w[t] for gen in self.generators if gen.inSubset == 1 for t in gen.hatT) - len(genValues)

                sumQ = sum(Q[k])
                sumL = sum(L[k])
                self.model.addConstr(self.theta.sum(
                    k, '*') >= (sumQ - sumL) * (self.expr1 + self.expr2) + sumQ)

        if self.case == 'KT+++' or self.case == 'combine-KT':
            for k in self.rangeK:
                for t in self.rangeT:
                    self.set_hatTt(k, t)

                    self.expr1 = gp.quicksum(
                        line.z[t] for line in self.lines if line.inSubset == 1 for t in line.hatTt) - len(lineValues)
                    self.expr2 = gp.quicksum(
                        gen.w[t] for gen in self.generators if gen.inSubset == 1 for t in gen.hatTt) - len(genValues)

                    sumQ = Q[k][t]
                    sumL = L[k][t]
                    self.model.addConstr(self.theta[k, t] >= (
                            sumQ - sumL) * (self.expr1 + self.expr2) + sumQ)

        if self.case == 'single_KT+++':
            self.expr3 = 0
            for k in self.rangeK:
                for t in self.rangeT:
                    self.set_hatTt(k, t)
                    self.expr1 = gp.quicksum(
                        line.z[t] for line in self.lines if line.inSubset == 1 for t in line.hatTt) - len(lineValues)
                    self.expr2 = gp.quicksum(
                        gen.w[t] for gen in self.generators if gen.inSubset == 1 for t in gen.hatTt) - len(genValues)

                    sumQ = Q[k][t]
                    sumL = L[k][t]

                    self.expr3 += (sumQ - sumL) * (self.expr1 + self.expr2)

            sumQ = sum(sum(val) for val in Q)

            self.model.addConstr(self.theta.sum(
                '*', '*') >= (self.expr3) + sumQ)

        self.model.update()

    def add_chance_constr(self, type=None):

        if type == 'socp':
            epsi = self.model.addVars(2, lb=0, ub=1, name='eps')

            self.model.addConstr(epsi[0] * epsi[1] >= 1 - eps)

        for gen in self.generators:
            if gen.inSubset == 1:
                gen.chanceCoef = [stats.invgauss.cdf(
                    t, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape) for t in range(1, self.T + 1)]
                gen.chanceCoef.append(stats.invgauss.cdf(
                    self.T, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape))
            else:
                gen.chanceCoef = stats.invgauss.cdf(
                    self.T, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape)

        for line in self.lines:
            if line.inSubset == 1:
                line.chanceCoef = [stats.invgauss.cdf(
                    t, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape) for t in range(1, self.T + 1)]
                line.chanceCoef.append(stats.invgauss.cdf(
                    self.T, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape))
            else:
                line.chanceCoef = stats.invgauss.cdf(
                    self.T, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape)

        n = len(self.generators)
        ro = max(1, int(len(self.generators) / divGen))

        if type == 'socp':
            q = ro * (1 - epsi[0])
        else:
            val = Bisection(n, eps, ro)
            q = max(ro * eps, val)
        self.model.addConstr((gp.quicksum(
            gen.chanceCoef[t] * gen.w[t] for t in range(self.barT) for gen in self.generators if gen.inSubset == 1)
                              + gp.quicksum(gen.chanceCoef for gen in self.generators if gen.inSubset == 0))
                             <= q)

        if includeLine == 1:
            n = len(self.lines)
            ro = max(1, int(len(self.lines) / divLine))

            if type == 'socp':
                q = ro * (1 - epsi[1])
            else:
                val = Bisection(n, eps, ro)
                q = max(ro * eps, val)

            self.model.addConstr((gp.quicksum(
                line.chanceCoef[t] * line.z[t] for t in range(self.barT) for line in self.lines if line.inSubset == 1)
                                  + gp.quicksum(line.chanceCoef for line in self.lines if line.inSubset == 0))
                                 <= q)



    def add_cover_inequality(self, genValues, lineValues, generators, lines, type=''):
        if type == 'stronger':
            expr = gp.quicksum(
                generators[key].w[t] for key in genValues.keys() for t in range(genValues[key].index(1), self.barT)) \
                   + includeLine * gp.quicksum(lines[key].z[t] for key in lineValues.keys()
                                               for t in range(lineValues[key].index(1), self.barT))

            self.model.addConstr(expr <= len(genValues) +
                                 includeLine * len(lineValues) - 1)

        else:
            expr = gp.quicksum(generators[key].w[genValues[key].index(1)] for key in genValues.keys()) \
                   + includeLine * gp.quicksum(lines[key].z[lineValues[key].index(1)]
                                               for key in lineValues.keys())

            self.model.addConstr(expr <= len(genValues) +
                                 includeLine * len(lineValues) - 1)

    def add_initial_cover_inequality(self, generators, lines):

        dict = {'generators': [gen.listIndex for gen in generators if gen.inSubset == 1],
                'lines': [line.listIndex for line in lines if line.inSubset == 1]}

        genValues = {}
        for gen in generators:
            if gen.inSubset == 1:
                genValues[gen.listIndex] = [0] * self.barT
                if gen.mostProbFail == 1:
                    genValues[gen.listIndex][0] = 1
                else:
                    genValues[gen.listIndex][0] = 1

        for t_ in range(self.barT):
            for line in dict['lines']:
                for t in range(self.barT):
                    lineValues = {}
                    lineValues[line] = [0] * self.barT
                    lineValues[line][self.barT - 1 - t] = 1
                    for index in dict['lines']:
                        if line != index:
                            lineValues[index] = [0] * self.barT
                            lineValues[index][t_] = 1
                    flag = evaluateChanceConstr(
                        generators, genValues, lines, lineValues)

                    if flag == 'infeasible':
                        infDay = t
                        genValuesPrev = genValues
                        lineValuesPrev = lineValues
                        continue
                    else:
                        if t == 0:
                            break
                        if infDay != None:
                            self.add_cover_inequality(
                                genValuesPrev, lineValuesPrev, generators, lines, type='stronger')
                            infDay = None
                        else:
                            break

        lineValues = {}
        for line in lines:
            if line.inSubset == 1:
                lineValues[line.listIndex] = [0] * self.barT
                if line.mostProbFail == 1:
                    lineValues[line.listIndex][0] = 1
                else:
                    lineValues[line.listIndex][0] = 1

        for t_ in range(self.barT):
            for gen in dict['generators']:
                for t in range(self.barT):
                    genValues = {}
                    genValues[gen] = [0] * self.barT
                    genValues[gen][self.barT - 1 - t] = 1
                    for index in dict['generators']:
                        if gen != index:
                            genValues[index] = [0] * self.barT
                            genValues[index][t_] = 1
                    flag = evaluateChanceConstr(
                        generators, genValues, lines, lineValues)

                    if flag == 'infeasible':
                        infDay = t
                        genValuesPrev = genValues
                        lineValuesPrev = lineValues
                        continue
                    else:
                        if t == 0:
                            break
                        if infDay != None:
                            self.add_cover_inequality(
                                genValuesPrev, lineValuesPrev, generators, lines, type='stronger')
                            infDay = None
                        else:
                            break


def SubProblemParallel(instance, genValues, lineValues, genScenarios, lineScenarios, solveAs, solveBy, discardBy,
                       scenNo, dayNo, minFlows=None, maxFlows=None, fixedMaintenance=None):
    rangeK, rangeT, rangeS = method(solveBy, scenNo, dayNo)

    buses, lines, generators, costs = Input(instance)

    # assign scenarios
    assignScenarios(generators, genScenarios)
    assignScenarios(lines, lineScenarios)

    if minFlows != None or maxFlows != None:
        discardSomeLines(discardBy, minFlows, maxFlows, lines)

    model = gp.Model('Sub-Problem')

    findCompUpperBoundsOnScenario(generators, genValues, scenNo)
    findCompUpperBoundsOnScenario(lines, lineValues, scenNo)
    model.setParam('OutputFlag', 0)

    for bus in buses:
        bus.add_operation_variables(model, solveBy, scenNo, dayNo)

    yStatus = {}
    for line in lines:
        line.add_operation_variables(model, solveAs, solveBy, scenNo, dayNo)
        yStatus[line.listIndex] = line.y

    xStatus = {}
    for gen in generators:
        gen.add_operation_variables(model, solveAs, solveBy, scenNo, dayNo)
        xStatus[gen.listIndex] = gen.x

    for bus in buses:
        bus.add_operation_constraints(model)

    for line in lines:
        line.add_operation_constraints(model)

    for gen in generators:
        gen.add_operation_constraints(model)

    bus_operation_cost = gp.quicksum(
        bus.add_operation_costs(scenNo) for bus in buses)

    gen_operation_cost = gp.quicksum(
        gen.add_operation_costs(scenNo) for gen in generators)

    totalCost = bus_operation_cost + gen_operation_cost

    model.setObjective(totalCost, gp.GRB.MINIMIZE)

    # assign upper bounds
    for gen in generators:
        if gen.inSubset == 1:
            for k in rangeK:
                for t in rangeT:
                    for s in rangeS:
                        gen.x[k, t, s].UB = gen.upperBounds[k][t]

    for line in lines:
        if line.inSubset == 1:
            for k in rangeK:
                for t in rangeT:
                    for s in rangeS:
                        if line.upperBounds[k][t] == 0:
                            line.y[k, t, s].UB = line.upperBounds[k][t]
                        if line.upperBounds[k][t] == 1:
                            line.y[k, t, s].LB = line.upperBounds[k][t]

    model.setParam("Threads", 1)
    model.optimize()

    if fixedMaintenance == 0:
        opt = model.objVal
        return opt

    else:
        curtailCost = sum([bus.operation_cost.getValue() for bus in buses])
        prodCost = sum([gen.C_prod.getValue() for gen in generators])
        commitCost = sum([gen.C_x.getValue() for gen in generators])
        startCost = sum([gen.C_u.getValue() for gen in generators])
        opt = model.objVal

        countBusCurtail = []
        for bus in buses:
            count = 0
            for key in bus.q.keys():
                if bus.q[key].x > 10 ** -6:
                    count += 1
            countBusCurtail.append(count)


        return (opt, curtailCost, prodCost, commitCost, startCost, sum(countBusCurtail))


class DiscardLinesModel():

    def __init__(self, buses, lines, generators, solveAs, solveBy, day, hour):

        self.model = gp.Model('DISCARD-LINES')

        self.model.setParam('OutputFlag', 0)

        self.solveAs = solveAs
        self.solveBy = solveBy

        for bus in buses:
            bus.delta = self.model.addVar(
                lb=bus.deltaMin, ub=bus.deltaMax, vtype=gp.GRB.CONTINUOUS, name='delta_(' + str(bus.listIndex) + ').')

            bus.q = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name='q_(' + str(bus.listIndex) + ').')

            bus.d = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name='d_(' + str(bus.listIndex) + ').')

        for line in lines:
            line.f = self.model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                       vtype=gp.GRB.CONTINUOUS, name='f_(' + str(line.listIndex) + ').')

            if line.inSubset == 1:
                line.y = self.model.addVar(
                    lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='y_(' + str(line.listIndex) + ').')

        for gen in generators:
            gen.x = self.model.addVar(
                lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='x_(' + str(gen.listIndex) + ').')
            gen.p = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name='p_(' + str(gen.listIndex) + ').')

        for bus in buses:
            self.model.addConstr(bus.q <= bus.d)
            self.model.addConstr(bus.d <= bus.dMax[day][hour])
            # Flow balance
            self.model.addConstr(gp.quicksum(gen.p for gen in bus.generators) + bus.q - bus.d == gp.quicksum(
                outgoing.f for outgoing in bus.outgoing_arcs)
                                 - gp.quicksum(incoming.f for incoming in bus.incoming_arcs))

        for line in lines:
            if line.inSubset == 1:
                # Upper bound - balance
                self.model.addConstr(
                    line.B * (line.from_bus.delta - line.to_bus.delta) + line.M * (1 - line.y) >= line.f)

                # Lower bound -  balance
                self.model.addConstr(
                    line.B * (line.from_bus.delta - line.to_bus.delta) - line.M * (1 - line.y) <= line.f)

                # Upper bound on power flow
                self.model.addConstr(line.f_bar * line.y >= line.f)

                # Lower bound on power flow
                self.model.addConstr(-line.f_bar * line.y <= line.f)

            if line.inSubset == 0:
                # flow definition
                self.model.addConstr(
                    line.B * (line.from_bus.delta - line.to_bus.delta) == line.f)

        for gen in generators:
            # Upper bound on power generation
            self.model.addConstr(gen.Pmax * gen.x >= gen.p)
            # Lower bound on power generation
            self.model.addConstr(gen.Pmin * gen.x <= gen.p)


def solveDiscardLinesModel(buses, lines, generators, solveAs, solveBy, discardBy):
    K, T, S, barT = set_horizon(opts)
    rangeT, rangeS = range(0, T), range(0, S)
    if discardBy == 'DISCARD-M0':
        rangeT, rangeS = range(0, 1), range(0, 1)
        for bus in buses:
            bus.dMax[0].append(bus.file_demand)

    elif discardBy == 'DISCARD-M1':
        # solve for each day
        rangeT, rangeS = range(0, T), range(0, 1)
        for bus in buses:
            for t in rangeT:
                bus.dMax[t].append(max(bus.demand[t]))

    elif discardBy == 'DISCARD-M2':
        # solve for each day and each hour
        rangeT, rangeS = range(0, T), range(0, S)
        for bus in buses:
            for t in rangeT:
                for s in rangeS:
                    bus.dMax[t].append(bus.demand[t][s])

    minFlows = {line.listIndex: {j: [0 for i in rangeS]
                                 for j in rangeT} for line in lines if line.inSubset == 0}
    maxFlows = {line.listIndex: {j: [0 for i in rangeS]
                                 for j in rangeT} for line in lines if line.inSubset == 0}

    # MINIMIZE FLOW
    for t in rangeT:
        for s in rangeS:
            l = DiscardLinesModel(buses, lines, generators,
                                  solveAs, solveBy, t, s)

            for line in lines:
                if line.inSubset == 0 and line.f_bar != 9900 / 100:

                    l.model.setObjective(line.f, gp.GRB.MINIMIZE)
                    l.model.optimize()

                    minFlows[line.listIndex][t][s] = l.model.objVal

                    if l.model.objVal < (-line.f_bar + 1e-4):
                        l.model.addConstr(-line.f_bar <= line.f)

                    l.model.setObjective(line.f, gp.GRB.MAXIMIZE)
                    l.model.optimize()

                    maxFlows[line.listIndex][t][s] = l.model.objVal

                    if l.model.objVal > (line.f_bar - 1e-4):
                        l.model.addConstr(line.f_bar >= line.f)


    return minFlows, maxFlows


def discardSomeLines(discardBy, minFlows, maxFlows, lines):
    K, T, S, barT = set_horizon(opts)

    UBList, LBList = [], []

    #### w vs x ####
    with open('line_eliminiation_' + str(discardBy) + '.csv', 'w', newline='') as file:

        fieldnames = ['discardBy', 'lineIndex',
                      'LB', 'fbar', 'UB', 't', 's', 'countUB', 'countLB']

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {'lineIndex': sum(1 for line in lines if line.inSubset == 0)})
        writer.writerow({'discardBy': discardBy})
        countUB = 0
        countLB = 0
        for line in lines:
            if line.inSubset == 0 and line.f_bar != 9900 / 100:
                if discardBy == 'DISCARD-M0':
                    if maxFlows[line.listIndex][0][0] > (line.f_bar - 1e-4):
                        writer.writerow({'lineIndex': line.listIndex, 'LB': '-',
                                         'UB': maxFlows[line.listIndex][0][0], 'fbar': line.f_bar, 't': 0, 's': 0})
                        line.isRemovedUB = 0
                    else:
                        for t in range(T):
                            for s in range(S):
                                line.f_barUB[t][s] = 9900 / 100
                                line.isRemovedUB = 1
                                countUB += 1
                                UBList.append(line.listIndex)

                    if minFlows[line.listIndex][0][0] < (- line.f_bar + 1e-4):
                        writer.writerow(
                            {'lineIndex': line.listIndex, 'LB': minFlows[line.listIndex][0][0], 'UB': '-',
                             'fbar': -line.f_bar, 't': 0, 's': 0})
                        line.isRemovedLB = 0
                        # print('lower bound for line' + str(line.listIndex) + ' ' + str(minFlows[line.listIndex][0][0]) + '<' + str((- line.f_bar)))
                    else:
                        for t in range(T):
                            for s in range(S):
                                line.f_barLB[t][s] = - 9900 / 100
                                line.isRemovedLB = 1
                                countLB += 1
                                LBList.append(line.listIndex)


                if discardBy == 'DISCARD-M1':
                    for t in range(T):
                        if maxFlows[line.listIndex][t][0] > (line.f_bar - 1e-4):
                            writer.writerow(
                                {'lineIndex': line.listIndex, 'LB': '-', 'UB': maxFlows[line.listIndex][t][0],
                                 'fbar': line.f_bar, 't': t, 's': 0})
                            line.isRemovedUB = 0
                        else:
                            for s in range(S):
                                line.f_barUB[t][s] = 9900 / 100
                                line.isRemovedUB = 1
                                countUB += 1
                                UBList.append(line.listIndex)


                        if minFlows[line.listIndex][t][0] < (- line.f_bar + 1e-4):
                            writer.writerow(
                                {'lineIndex': line.listIndex, 'LB': minFlows[line.listIndex][t][0], 'UB': '-',
                                 'fbar': -line.f_bar, 't': t, 's': 0})
                            line.isRemovedLB = 0
                        else:
                            for s in range(S):
                                line.f_barLB[t][s] = - 9900 / 100
                                line.isRemovedLB = 1
                                countLB += 1
                                LBList.append(line.listIndex)

                if discardBy == 'DISCARD-M2':
                    for t in range(T):
                        for s in range(S):
                            if maxFlows[line.listIndex][t][s] > (line.f_bar - 1e-4):
                                writer.writerow(
                                    {'lineIndex': line.listIndex, 'LB': '-', 'UB': maxFlows[line.listIndex][t][s],
                                     'fbar': line.f_bar, 't': t, 's': s})
                                line.isRemovedUB = 0
                            else:
                                line.f_barUB[t][s] = 9900 / 100
                                line.isRemovedUB = 1
                                countUB += 1
                                UBList.append(line.listIndex)


                            if minFlows[line.listIndex][t][s] < (- line.f_bar + 1e-4):
                                writer.writerow(
                                    {'lineIndex': line.listIndex, 'LB': minFlows[line.listIndex][t][s], 'UB': '-',
                                     'fbar': -line.f_bar, 't': t, 's': s})
                                line.isRemovedLB = 0
                            else:
                                line.f_barLB[t][s] = - 9900 / 100
                                line.isRemovedLB = 1
                                countLB += 1
                                LBList.append(line.listIndex)



        writer.writerow({'discardBy': 'UB is removed for',
                         'lineIndex': set(UBList)})
        writer.writerow({'discardBy': 'LB is removed for',
                         'lineIndex': set(LBList)})
        writer.writerow({'countUB': countUB, 'countLB': countLB})


def getRun(instanceList, seed, scenarios, solveAs, discardBy, solver=None):
    date = datetime.datetime.now()
    day = date.strftime("%d")
    month = date.strftime("%b")
    time = date.strftime("%X").replace(':', '-')

    if discardBy == None:
        discardBy = ['dummy']

    #### w vs x ####
    with open('run-time-' + str(day) + '-' + str(month) + '-' + str(time) + '.csv', 'x', newline='') as file:

        fieldnames = ['instance', 'Threads', 'withFixed', 'Safe', 'Exact', 'withPosterior', 'withNoFailure',
                      'cutGenerated', 'case', 'seed', 'scenNo',
                      'solveBy',
                      'discardTime', 'IterNo', 'lineValues', 'genValues', 'LB', 'UB', 'GAP',
                      'TIME',
                      'timeInLowerbouding', 'TimeInMPs', 'TimeInSPs', 'maintenanceCost', 'operationCost',
                      'genPredCost', 'genCorrCost', 'linePredCost', 'lineCorrCost',
                      'curtailCost', 'prodCost', 'commitCost', 'startCost', 'sumMaintenance', 'sumOperation']

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        SAAData = {}
        count = 0
        for instance in instanceList:
            for key in solver.keys():
                for case in solver[key]:
                    for discard in discardBy:
                        for scen in scenarios:
                            opts['K'] = scen

                            # when trying to compare results for different cases,
                            # comment this parts so that seed is fixed
                            if type(seed) == list:
                                seed_ = seed[count]
                            else:
                                seed_ = seed

                            if key == 'single':
                                buses, lines, generators, lineSubset, genSubset, MP, SP, Q, L, data, signalDataLine, signalDataGen, bigLineSubset, \
                                    bigGenSubset = SolveOptModel(
                                    instance,
                                    seed_,
                                    solveAs,
                                    'single', discardBy=discard)
                                writer.writerow({'instance': instance, 'Threads': num_cores, 'withFixed': withFixed,
                                                 'Safe': withSafe, 'Exact': str(withExact),
                                                 'case': 'single', 'seed': data[-2], 'scenNo': scen, 'solveBy': '-',
                                                 'IterNo': '-',
                                                 'lineValues': data[0], 'cutGenerated': data[-1],
                                                 'genValues': data[1], 'LB': data[2], 'UB': data[3], 'GAP': data[4],
                                                 'TIME': data[5]})
                            else:
                                buses, lines, generators, lineSubset, genSubset, MP, SP, Q, L, data, signalDataLine, signalDataGen, bigLineSubset, \
                                    bigGenSubset = SolveOptModel(
                                    instance,
                                    seed_,
                                    solveAs, key, discard,
                                    case)
                                SAAData[count] = data
                                writer.writerow(
                                    {'instance': instance, 'Threads': num_cores,
                                     'withFixed': withFixed, 'Safe': withSafe, 'Exact': str(withExact),
                                     'withPosterior': withPosterior,
                                     'withNoFailure': withNoFailure,
                                     'cutGenerated': data['cutGenerated'], 'case': case, 'seed': data['seed'],
                                     'scenNo': scen,
                                     'solveBy': '-', 'discardTime': data['discardTime'],
                                     'IterNo': data['IterNo'],
                                     'lineValues': data['lineValues'], 'genValues': data['genValues'], 'LB': data['LB'],
                                     'UB': data['UB'],
                                     'GAP': data['opt_gap'], 'timeInLowerbouding': data['timeInLowerbouding'],
                                     'TimeInMPs': data['TimeInMPs'], 'TimeInSPs': data['TimeInSPs'],
                                     'TIME': data['TimeInTotal'],
                                     'maintenanceCost': data['maintenanceCost'], 'operationCost': data['operationCost'],
                                     'genPredCost': data['genPredCost'], 'genCorrCost': data['genCorrCost'],
                                     'linePredCost': data['linePredCost'], 'lineCorrCost': data['lineCorrCost'],
                                     'curtailCost': data['curtailCost'], 'prodCost': data['prodCost'],
                                     'commitCost': data['commitCost'], 'startCost': data['startCost'],
                                     # BAHAR ADDITION
                                     'sumMaintenance': data['sumMaintenance'], 'sumOperation': data['sumOperation']
                                     })
                            count += 1
                            file.flush()

    return buses, lines, generators, lineSubset, genSubset, MP, SP, Q, L, data, SAAData, signalDataLine, signalDataGen, bigLineSubset, bigGenSubset


def set_horizon(opts):
    K, T, S = opts['K'], opts['T'], opts['S']
    barT = T + 1
    return K, T, S, barT


def scenario_prob():
    K, T, S, barT = set_horizon(opts)

    prob = []
    for k in range(K):
        prob.append(1 / K)

    return prob


def Demand(name):
    K, T, S, barT = set_horizon(opts)

    demand = {}
    for t in range(T):
        demand[t] = [0] * S

    # 'weekly-demand.csv'
    with open(name) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter='\t')

        day = 0
        hour = 0

        next(csv_reader)  # skips header

        for row in csv_reader:

            hour = hour % S

            demand[day][hour] = int(str(row).split(';')[2])

            hour += 1

            if hour == S:
                day += 1

    return demand


def Rescale(data, ub, lb):
    days = len(data)
    hours = len(data[0])

    rescaled = {}

    # Find max & min within a week
    max_ = max([max(data[key]) for key in data.keys()])
    min_ = min([min(data[key]) for key in data.keys()])

    for t in range(days):
        rescaled[t] = [0] * hours

        for s in range(hours):
            rescaled[t][s] = (data[t][s] - min_) / \
                             (max_ - min_) * (ub - lb) + lb

    new_keys = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

    rescaled = dict((new_keys[key], value) for (key, value) in rescaled.items())

    return rescaled


def method(solveBy, k=-1, t=-1):
    K, T, S, barT = set_horizon(opts)

    method = {'single': [range(0, K), range(0, T), range(0, S)],
              'K': [range(k, k + 1), range(0, T), range(0, S)],
              'KT': [range(k, k + 1), range(t, t + 1), range(0, S)],
              'DISCARD-M0': [range(0, 1), range(0, 1), range(0, 1)],
              'DISCARD-M1': [range(0, 1), range(0, 1), range(0, 1)],
              'DISCARD-M2': [range(0, 1), range(0, 1), range(0, 1)]}

    # bu method rangeK,T,S return eder.
    return method[solveBy][0], method[solveBy][1], method[solveBy][2]


def program(solveAs):
    program = {'LP': gp.GRB.CONTINUOUS, 'MIP': gp.GRB.BINARY}

    return program[solveAs]


def Input(name):
    buses, generators, lines, costs = [], [], [], []

    # read [bus] from m.file
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        isStart = 0
        for row in csv_reader:
            if not (row):
                continue

            if row[0] == '];' and isStart == 1:
                isStart = 0

            if isStart == 1:
                bus = Bus(row[1:])
                buses.append(bus)

            if row[0] == 'mpc.bus = [':
                isStart = 1

    # read [gencost] from m.file
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        isStart = 0
        for row in csv_reader:
            if not (row):
                continue

            if row[0] == '];' and isStart == 1:
                isStart = 0

            if isStart == 1:
                cost = Costs(row[1:])
                costs.append(cost)

            if row[0] == 'mpc.gencost = [':
                isStart = 1

    # read [gen] from m.file
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        isStart = 0
        costIndex = 0
        for row in csv_reader:
            if not (row):
                continue

            if row[0] == '];' and isStart == 1:
                isStart = 0

            if isStart == 1:
                generator = Generator(row[1:], buses, costs[costIndex])
                generators.append(generator)
                costIndex = costIndex + 1

            if row[0] == 'mpc.gen = [':
                isStart = 1

    # read [branch] from m.file
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        isStart = 0
        for row in csv_reader:
            if not (row):
                continue

            if row[0] == '];' and isStart == 1:
                isStart = 0

            if isStart == 1:
                if int(row[11]) != 0:
                    line = Line(row[1:], buses, generators)
                    lines.append(line)

            if row[0] == 'mpc.branch = [':
                isStart = 1

    # Indexing, since some can differ in m.file
    lists = {bus: buses, line: lines, generator: generators}
    for key in lists:
        index = 0
        for key_ in lists[key]:
            key_.listIndex = index
            index += 1

    return buses, lines, generators, costs


def chooseSubset(components, compScen, ratio, type='gen'):
    # create signal data

    T = opts['T']
    signalData = create(sampleSize, 100, compScen['threshold'], compScen['theta_mu'], compScen['theta_sigma'],
                        compScen['nu_mu'], compScen['nu_sigma'], compScen['e_sigma'])

    PriorTheta_mu, PriorNu_mu = Priors(signalData[:history])

    findProb = []
    figures = []
    for comp in components:
        comp.RLDIndex = np.random.randint(100, len(signalData))

        # find parameters
        signalData[comp.RLDIndex]['mean'], signalData[comp.RLDIndex]['shape'], signalData[comp.RLDIndex][
            'observedTime'], figure = RemainingCDF(
            signalData[comp.RLDIndex], compScen['threshold'], PriorTheta_mu, compScen['theta_sigma'] ** 2, PriorNu_mu,
                                                                             compScen['nu_sigma'] ** 2,
            compScen['e_sigma'], 1)

        comp.posteriorMean = signalData[comp.RLDIndex]['mean']
        comp.posteriorShape = signalData[comp.RLDIndex]['shape']

        figures.append(figure)

        findProb.append((stats.invgauss.cdf(T, comp.posteriorMean / comp.posteriorShape, 0, comp.posteriorShape),
                         comp.RLDIndex, comp.listIndex))

    # findProb[ ( P(t<8), distribution index, component index) ]
    findProb.sort(reverse=True)

    subset1 = []  # all components
    subset2 = []  # chosen components
    size = int(len(components) * ratio)
    if size == 0:
        size = 1

    if type == 'gen':
        p = 0.1
    elif type == 'line':
        p = (1 - includeLine) + 0.2

    for s in range(size):
        subset1.append((findProb[s][-1], findProb[s][1]))
        # append if probability is greater than 0.1
        if findProb[s][0] >= p:
            subset2.append([findProb[s][-1], findProb[s][1]])

    for comp in components:
        if comp.listIndex in [findProb[0][-1], findProb[1][-1], findProb[2][-1]]:
            comp.mostProbFail = 1
        else:
            comp.mostProbFail = 0

    for comp in components:
        if comp.listIndex in [findProb[0][-1], findProb[1][-1], findProb[2][-1]]:
            comp.mostProbFail = 1
        else:
            comp.mostProbFail = 0

    return subset1, subset2, signalData


def generateScenarios(compSubset, components, noOfScen, SAA=None):
    K, T, S, barT = set_horizon(opts)

    # a scenario sample of size K
    K = noOfScen

    scen = {item[0]: [] for item in compSubset}

    # item[0] listIndex
    # item[1] RLDIndex

    for item in compSubset:

        # if SAA is used, fix seed to each components' RLDIndex
        if SAA == 1:
            np.random.seed(item[-1])

        # s: comp list index , RLD index
        m, n = item[0], item[-1]

        # get distribution parameters (mu, lambda)
        mean, shape = components[m].posteriorMean, components[m].posteriorShape

        for k in range(K):

            if withPosterior == True:
                scen[m].append(min(int(mean + 0.5), barT))
            elif withNoFailure == True:
                scen[m].append(8)

            else:
                CDFs = np.array(
                    [stats.invgauss.cdf(t, mean / shape, 0, shape) for t in range(8)])

                # obtain a probability
                u = stats.uniform.rvs()

                # find an index j: F[j-1] <= u < F[j]
                j = sum(CDFs < u)

                if j <= T:
                    scen[m].append(j)
                else:
                    scen[m].append(barT)

    return scen


def assignScenarios(components, compScenarios):
    T = opts['T']
    for key in compScenarios.keys():
        components[key].inSubset = 1
        components[key].zeta = [min(x - 1, T) for x in compScenarios[key]]


def calculateLowerBound(instance, solveAs, solveBy, scenNo=-1, dayNo=-1):
    buses, lines, generators, costs = Input(instance)

    model = gp.Model('LB')
    model.setParam('OutputFlag', 0)
    model.setParam("Threads", 1)

    for bus in buses:
        bus.add_operation_variables(model, solveBy, scenNo, dayNo)

    for line in lines:
        line.add_maintenance_variables(model, solveAs)
        line.add_operation_variables(model, solveAs, solveBy, scenNo, dayNo)

    for gen in generators:
        gen.add_maintenance_variables(model, solveAs)
        gen.add_operation_variables(model, solveAs, solveBy, scenNo, dayNo)

    for bus in buses:
        bus.add_operation_constraints(model)

    for line in lines:
        line.add_maintenance_constraints(model)
        line.add_operation_constraints(model)
        if solveBy == 'K':
            if line.inSubset == 1:
                line.add_type_maintenance_constraints(model)

    for gen in generators:
        gen.add_maintenance_constraints(model)
        gen.add_operation_constraints(model)
        if solveBy == 'K':
            if gen.inSubset == 1:
                gen.add_type_maintenance_constraints(model)

    bus_operation_cost = gp.quicksum(
        bus.add_operation_costs(scenNo) for bus in buses)

    gen_operation_cost = gp.quicksum(
        gen.add_operation_costs(scenNo) for gen in generators)

    totalCost = bus_operation_cost + gen_operation_cost

    model.setObjective(totalCost, gp.GRB.MINIMIZE)

    model.update()

    model.setParam('MIPGap', mipgap)

    model.optimize()

    return model.objVal


def findCompUpperBoundsOnScenario(components, compValues, scenNo):
    """
    Given a first-stage decision vector for ALL generators in G',
    updates upperbounds in scenario (scenNo) from 1 to 0 if necessary
    """

    K, T, S, barT = set_horizon(opts)

    for comp in components:
        if comp.inSubset == 1:
            comp.resetScenarioUpperBounds(scenNo)
            firstStage = compValues[comp.listIndex]

            for t in range(barT):
                # under PREVENTIVE maintenance
                if t < comp.zeta[scenNo] and firstStage[t] == 1:
                    for j in range(t, min(t + comp.predTime, barT)):
                        comp.upperBounds[scenNo][j] = 0
                    break

                # under CORRECTIVE maintenance
                elif t == comp.zeta[scenNo]:
                    for j in range(comp.zeta[scenNo], min(comp.zeta[scenNo] + comp.corrTime, barT)):
                        comp.upperBounds[scenNo][j] = 0
                    break


def findCompUpperForAll(components):
    K, T, S, barT = set_horizon(opts)

    A = []
    for t in range(barT):
        temp = [0] * barT
        temp[t] = 1
        A.append(temp)

    compVal = {}
    for t in range(barT):
        for comp in components:
            if comp.inSubset == 1:
                compVal[comp.listIndex] = A[t]

        for k in range(K):
            findCompUpperBoundsOnScenario(components, compVal, k)
            for comp in components:
                if comp.inSubset == 1:
                    comp.statusList[k].append(comp.upperBounds[k])


def findCandStatus(generators, lines, decomp='KT'):
    K, T, S, barT = set_horizon(opts)

    cand_status = {k: [] for k in range(K)}

    if decomp == 'KT':
        for k in range(K):
            for t in range(T):
                hold = []

                for gen in generators:
                    if gen.inSubset == 1:
                        hold = hold + [gen.upperBounds[k][t]]

                for line in lines:
                    if line.inSubset == 1:
                        hold = hold + [line.upperBounds[k][t]]

                cand_status[k].append(hold)

    if decomp == 'K':
        for k in range(K):

            for gen in generators:
                if gen.inSubset == 1:
                    cand_status[k].extend(gen.upperBounds[k])

            for line in lines:
                if line.inSubset == 1:
                    cand_status[k].extend(line.upperBounds[k])

    return cand_status


def SingleStageModel(seed, buses, lines, generators, lineSubset, genSubset, solveAs, solveBy, type=''):
    K, T, S, barT = set_horizon(opts)
    prob = scenario_prob()
    status_code = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE',
                   4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 9: 'TIME_LIMIT'}

    np.random.seed(seed)
    # generate scenarios
    genScenarios = generateScenarios(genSubset, generators, K)
    lineScenarios = generateScenarios(lineSubset, lines, K)

    if withSafe == True:
        type = 'socp'

    NoCuts = 0
    # assign scenarios
    assignScenarios(generators, genScenarios)
    assignScenarios(lines, lineScenarios)

    model = gp.Model('single-stage')

    model.setParam('Threads', num_cores)

    model.setParam('OutputFlag', 1)

    for bus in buses:
        bus.add_operation_variables(model, solveBy)

    for line in lines:
        line.add_operation_variables(model, solveAs, solveBy)

        if line.inSubset == 1:
            line.add_maintenance_variables(model, solveAs)

    for gen in generators:
        gen.add_operation_variables(model, solveAs, solveBy)
        if gen.inSubset == 1:
            gen.add_maintenance_variables(model, solveAs)

    for bus in buses:
        bus.add_operation_constraints(model)

    for gen in generators:
        gen.add_operation_constraints(model)
        if gen.inSubset == 1:
            gen.add_maintenance_constraints(model)
            gen.add_type_maintenance_constraints(model)

    for line in lines:
        line.add_operation_constraints(model)
        if line.inSubset == 1:
            line.add_maintenance_constraints(model)
            line.add_type_maintenance_constraints(model)

    if withDynamic == True:
        type_ = 'Dynamic'
    if withFixed == True:
        type_ = 'Fixed'

    gen_maintenance_cost = gp.quicksum(prob[k] * gen.add_maintenance_costs(
        k, type_) for k in range(K) for gen in generators if gen.inSubset == 1)

    line_maintenance_cost = gp.quicksum(prob[k] * line.add_maintenance_costs(
        k, type_) for k in range(K) for line in lines if line.inSubset == 1)

    bus_operation_cost = gp.quicksum(
        prob[k] * bus.add_operation_costs(k) for k in range(K) for bus in buses)

    gen_operation_cost = gp.quicksum(
        prob[k] * gen.add_operation_costs(k) for k in range(K) for gen in generators)

    totalCost = (gen_maintenance_cost + line_maintenance_cost) + \
                (bus_operation_cost + gen_operation_cost)

    model.setObjective(totalCost, gp.GRB.MINIMIZE)

    # model.write("single-stage-formulation.lp")

    if type == 'socp':
        epsi = model.addVars(2, lb=0, ub=1, name='eps')

        model.addConstr(epsi[0] * epsi[1] >= 1 - eps)

        for gen in generators:
            if gen.inSubset == 1:
                gen.chanceCoef = [stats.invgauss.cdf(
                    t, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape) for t in range(1, T + 1)]
                gen.chanceCoef.append(stats.invgauss.cdf(
                    T, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape))
            else:
                gen.chanceCoef = stats.invgauss.cdf(
                    T, gen.posteriorMean / gen.posteriorShape, 0, gen.posteriorShape)

        for line in lines:
            if line.inSubset == 1:
                line.chanceCoef = [stats.invgauss.cdf(
                    t, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape) for t in range(1, T + 1)]
                line.chanceCoef.append(stats.invgauss.cdf(
                    T, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape))
            else:
                line.chanceCoef = stats.invgauss.cdf(
                    T, line.posteriorMean / line.posteriorShape, 0, line.posteriorShape)

        # sum([1 for gen in self.generators if gen.inSubset == 1])
        ro = max(1, int(len(generators) / divGen))

        q = ro * (1 - epsi[0])

        model.addConstr(
            (gp.quicksum(gen.chanceCoef[t] * gen.w[t] for t in range(barT) for gen in generators if gen.inSubset == 1)
             + gp.quicksum(gen.chanceCoef for gen in generators if gen.inSubset == 0))
            <= q)

        # sum([1 for line in self.lines if line.inSubset == 1 ])
        ro = max(1, int(len(lines) / divLine))

        q = ro * (1 - epsi[1])

        model.addConstr(
            (gp.quicksum(line.chanceCoef[t] * line.z[t] for t in range(barT) for line in lines if line.inSubset == 1)
             + gp.quicksum(line.chanceCoef for line in lines if line.inSubset == 0))
            <= q)

        # self.model.write('socp-chance.lp')
    model.setParam('MIPGap', mipgap)

    model.setParam('TimeLimit', 6 * 60 * 60)

    t0 = time.perf_counter()
    model.optimize()

    status = model.status

    if status == 2 or status == 9:

        data = [[], [], model.objBound,
                model.objVal, model.MIPGap * 100, 0, seed, 0]

        if str(model.MIPGap * 100) != 'inf':

            lineValues = {line.listIndex: [
                                              0] * barT for line in lines if line.inSubset == 1}
            for line in lines:
                if line.inSubset == 1:
                    t = 0
                    for v in line.z.values():
                        if v.X == 1:
                            data[0].append(v.varName)
                            lineValues[line.listIndex][t] = 1
                        t += 1

            genValues = {gen.listIndex: [
                                            0] * barT for gen in generators if gen.inSubset == 1}

            for gen in generators:
                if gen.inSubset == 1:
                    t = 0
                    for v in gen.w.values():
                        if v.X == 1:
                            data[1].append(v.varName)
                            genValues[gen.listIndex][t] = 1
                        t += 1

        if type == 'exact':
            exact = False

            while exact == False:

                flag = evaluateChanceConstr(
                    generators, genValues, lines, lineValues)

                if flag == 'feasible':
                    exact = True

                if flag == 'infeasible':

                    if status != 9:
                        NoCuts += 1

                        expr = gp.quicksum(generators[key].w[t] for key in genValues.keys() for t in
                                           range(genValues[key].index(1), barT)) \
                               + gp.quicksum(lines[key].z[t] for key in lineValues.keys()
                                             for t in range(lineValues[key].index(1), barT))

                        model.addConstr(expr <= len(genValues) +
                                        len(lineValues) - 1)

                        model.optimize()

                        lineValues = {line.listIndex: [
                                                          0] * barT for line in lines if line.inSubset == 1}
                        for line in lines:
                            if line.inSubset == 1:
                                t = 0
                                for v in line.z.values():
                                    if v.X > 0.5:
                                        lineValues[line.listIndex][t] = 1
                                    t += 1

                        genValues = {gen.listIndex: [
                                                        0] * barT for gen in generators if gen.inSubset == 1}
                        for gen in generators:
                            if gen.inSubset == 1:
                                t = 0
                                for v in gen.w.values():
                                    if v.X > 0.5:
                                        genValues[gen.listIndex][t] = 1
                                    t += 1
                    else:
                        NoCuts = 'infeasible_time_limit'

        t1 = time.perf_counter()
        data[-3] = t1 - t0
        data[2] = model.objBound
        data[3] = model.objVal

        if str(model.MIPGap * 100) != 'inf':
            data[0] = lineValues
            data[1] = genValues

        data[-1] = NoCuts

    if status == 3:
        model.computeIIS()
        model.write("single-stage-formulation.ilp")

    return buses, lines, generators, data


def runParallel(instance, genValues, lineValues, genScenarios, lineScenarios, solveAs, solveBy, discardBy, scen, day,
                minFlows, maxFlows, fixMaint):
    opt = SubProblemParallel(instance, genValues, lineValues, genScenarios,
                             lineScenarios, solveAs, solveBy, discardBy, scen, day, minFlows, maxFlows,
                             fixedMaintenance=fixMaint)


    return (scen, opt)


def runParallelBound(instance, solveBy, scen, day):
    opt = calculateLowerBound(instance, 'LP', solveBy, scen, day)

    return opt


def DecompModel(instance, seed, buses, lines, generators, lineSubset, genSubset, solveAs, solveBy, discardBy, case):
    data = {}
    data['seed'] = seed
    data['instance'] = instance

    if case.split(';')[0] == 'withoutStatus':
        withoutStatus = True
    else:
        withoutStatus = False

    case = case.split(';')[-1]

    K, T, S, barT = set_horizon(opts)
    rangeK = range(0, K)
    rangeT = range(0, 1)
    if solveBy == 'KT':
        rangeT = range(0, T)

    UB = float("inf")
    LB = -UB

    np.random.seed(seed)
    # generate scenarios
    genScenarios = generateScenarios(genSubset, generators, K)
    lineScenarios = generateScenarios(lineSubset, lines, K)

    # assign scenarios
    assignScenarios(generators, genScenarios)
    assignScenarios(lines, lineScenarios)

    minFlows, maxFlows = None, None
    time0 = time.perf_counter()
    if discardBy != 'dummy':
        minFlows, maxFlows = solveDiscardLinesModel(
            buses, lines, generators, solveAs, solveBy, discardBy)
        discardSomeLines(discardBy, minFlows, maxFlows, lines)
        time1 = time.perf_counter()
        data['discardBy'] = time1 - time0
    else:
        data['discardBy'] = '-'
        time1 = time0

    data['discardTime'] = time1 - time0

    data['minFlows'] = minFlows
    data['maxFlows'] = maxFlows

    IterNo = 0

    # LOWER BOUNDS ON THE SECOND STAGE OBJ.
    time0 = time.perf_counter()
    L_ = [[0] * K] * (rangeT.stop - rangeT.start)
    for t in rangeT:
        L_[t] = Parallel(n_jobs=num_cores)(
            delayed(runParallelBound)(instance, solveBy, k, t) for k in rangeK)
    L = [[0] * (rangeT.stop - rangeT.start)] * K
    for k in rangeK:
        for t in rangeT:
            L[k][t] = L_[t][k]
    time1 = time.perf_counter()
    data['timeInLowerbouding'] = time1 - time0

    # CREATE MASTER PROBLEM

    MP = MasterProblem(lines, generators, L, solveAs, solveBy, case)

    if withPosterior == False and withNoFailure == False:
        if withSafe == True:
            # MP.add_chance_constr()
            if includeChanceCons == True:
                MP.add_chance_constr(type='socp')

    SP = None

    # create all upper bounds for all scenarios given all decision vectors
    findCompUpperForAll(generators)
    findCompUpperForAll(lines)

    TimeMP = []
    TimeSP = []
    holdStatus = {t: {'status': [], 'value': []} for t in rangeT}
    NoCuts = 0
    start_time = time.perf_counter()
    flag = 'feasible'
    while (UB - LB) / UB > mipgap or math.isnan((UB - LB) / LB):

        IterNo += 1
        print('-' * 50 + '\n' + 'IterNo:' + str(IterNo) + '\n')
        data['IterNo'] = IterNo
        print('..........SOLVING MASTER...........')
        # SOLVE MASTER
        time0 = time.perf_counter()
        MP.optimize()
        time1 = time.perf_counter()

        TimeMP.append(time1 - time0)

        # STORE FIRST-STAGE VALUES
        lineValues, genValues = MP.store_first_stage()

        print('lineValues:' + str(lineValues))
        print('genValues' + str(genValues) + '\n')
        print('maint costs: ' + str(MP.gen_cost_value + MP.line_cost_value))
        print('theta: ' + str(MP.theta_cost_value))
        print('MP opt: ' + str(MP.opt) + '\n')

        if withExact != None:
            if withPosterior == False and withNoFailure == False:
                flag = evaluateChanceConstr(
                    generators, genValues, lines, lineValues)
                if flag == 'infeasible':
                    NoCuts += 1
                    if withExact == False:
                        MP.add_cover_inequality(
                            genValues, lineValues, generators, lines)

                    if withExact == True:
                        MP.add_cover_inequality(
                            genValues, lineValues, generators, lines, type='stronger')

        # UPPER BOUNDS ON STATUS VARS
        for k in rangeK:
            findCompUpperBoundsOnScenario(generators, genValues, k)
            findCompUpperBoundsOnScenario(lines, lineValues, k)

        cand_status = findCandStatus(generators, lines, decomp=solveBy)

        time0 = time.perf_counter()
        print('..........SOLVING SUBPROBLEMS...........')
        if withoutStatus == False and flag == 'feasible':
            # find subproblems to be solved
            solveThisSubproblems = {k: [] for k in rangeK}

            NotSolveThisSubproblems = {k: [] for k in rangeK}

            mapping = {k: {t: [] for t in rangeT} for k in rangeK}
            Q = [[0] * (rangeT.stop - rangeT.start) for k in rangeK]
            for t in rangeT:
                for k in rangeK:
                    temp = cand_status[k][t]
                    if temp not in holdStatus[t]['status']:
                        holdStatus[t]['status'].append(temp)
                        solveThisSubproblems[k].append(t)
                        mapping[k][t].append(temp)
                    else:
                        NotSolveThisSubproblems[k].append(t)
                        mapping[k][t].append(temp)

            # change keys in solveThisSubproblems
            SP_ = {t: [] for t in rangeT}
            for scen in solveThisSubproblems.keys():
                if solveThisSubproblems[scen] == []:
                    pass
                else:
                    for day in solveThisSubproblems[scen]:
                        SP_[day].append(scen)
            # SP is a dictionary with days (keys) holds lists of scenarios (items) to be solved for that day

            results = {}

            for day in SP_.keys():
                results[day] = Parallel(n_jobs=num_cores)(
                    delayed(runParallel)(instance, genValues, lineValues, genScenarios, lineScenarios, solveAs, solveBy,
                                         discardBy, scen, day, minFlows,
                                         maxFlows, fixMaint=0)
                    for scen in SP_[day])
            time1 = time.perf_counter()
            TimeSP.append(time1 - time0)

            for day in SP_.keys():
                if SP_[day] != []:
                    for j in range(len(SP_[day])):
                        value = results[day][j][1]
                        holdStatus[day]['value'].append(value)

            # MAP ALL OPTIMAL VALUES W.R.T. SCENARIO AND DAY

            for scen in mapping.keys():
                for day in mapping[scen].keys():
                    index = holdStatus[day]['status'].index(
                        mapping[scen][day][0])
                    Q[scen][day] = holdStatus[day]['value'][index]

        elif withoutStatus == True and flag == 'feasible':
            Q = [0] * K
            for k in rangeK:
                Q[k] = [0] * (rangeT.stop - rangeT.start)

            t0 = time.perf_counter()

            results = Parallel(n_jobs=num_cores)(
                delayed(runParallel)(instance, genValues, lineValues, genScenarios, lineScenarios, solveAs, solveBy,
                                     discardBy, k, 0, minFlows,
                                     maxFlows)
                for k in rangeK)
            for k in rangeK:
                Q[k][0] = results[k][-1]

            t1 = time.perf_counter()
            TimeSP.append(t1 - t0)

        if flag == 'feasible':
            LB = max(LB, MP.opt)
            print('LB:' + str(LB))
            second_stage_value = sum(sum(val) for val in Q)
            expected_second_stage = second_stage_value / K

            print('expected second stage: ' + str(expected_second_stage))

            # UPPER BOUND
            UB_cand = (MP.gen_cost_value + MP.line_cost_value) + \
                      expected_second_stage

            if UB_cand <= UB:
                data['genValues'] = genValues
                data['lineValues'] = lineValues
                UB = UB_cand

            # UB = min(UB, UB_cand)
            print('UB:' + str(UB) + '\n')
            data['UB'] = UB
            data['LB'] = LB

            # OPTIMALITY GAP
            opt_gap = ((UB - LB) * 100) / UB

            print('optimality gap: ' + str(opt_gap))
            data['opt_gap'] = opt_gap

            MP.add_cut(lineValues, genValues, Q, L)

            data['operationCost'] = expected_second_stage

        end_time = time.perf_counter()
        time__ = end_time - start_time
        print('TIME: ' + str(time__) + '\n')

        data['TimeInMPs'] = sum(TimeMP)
        data['TimeInSPs'] = sum(TimeSP)
        data['TimeInTotal'] = sum(TimeMP) + sum(TimeSP) + \
                              data['timeInLowerbouding']
        data['maintenanceCost'] = MP.gen_cost_value + MP.line_cost_value

        data['cutGenerated'] = NoCuts

        if time__ > 6 * 60 * 60:
            break

    data = solveModelFixedMaintenance(
        data, generators, lines, buses, lineSubset, genSubset)

    return MP, SP, Q, L, data


def solveModelFixedMaintenance(data, generators, lines, buses, lineSubset, genSubset, SAA=None):
    K, T, S, barT = set_horizon(opts)

    if SAA is not None:
        K = SAA

    status_code = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE',
                   4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 9: 'TIME_LIMIT'}

    solveBy = "single"

    genValues, lineValues = data['genValues'], data['lineValues']

    genPredK, genCorrK, linePredK, lineCorrK = [], [], [], []

    curtailCostK, prodCostK, commitCostK, startCostK = [], [], [], []

    if SAA is None:
        seed = data['seed']
        np.random.seed(seed)

    # generate scenarios
    if SAA is None:
        genScenarios = generateScenarios(genSubset, generators, K)
        lineScenarios = generateScenarios(lineSubset, lines, K)
    else:
        genScenarios = generateScenarios(genSubset, generators, K, 1)
        lineScenarios = generateScenarios(lineSubset, lines, K, 1)

    # assign scenarios
    assignScenarios(generators, genScenarios)
    assignScenarios(lines, lineScenarios)

    # FIX MAINTENANCE DECISIONS
    for key in genValues.keys():
        generators[key].w = genValues[key]

    for key in lineValues.keys():
        lines[key].z = lineValues[key]

    if SAA is not None:  # this solves w.r.t status
        N_ = K  # delete this later
        for k in range(N_):
            findCompUpperBoundsOnScenario(generators, genValues, k)
            findCompUpperBoundsOnScenario(lines, lineValues, k)

        cand_status = findCandStatus(generators, lines)

        # find subproblems to be solved
        solveThisSubproblems = {k: [] for k in range(N_)}
        storeOptForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        storeCurrForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        storeProdForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        storeCommitForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        storeStartForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        storeCurrNumberForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}

        NotSolveThisSubproblems = {k: [] for k in range(N_)}
        holdStatus = {t: [] for t in range(T)}
        holdStatusIndex = {t: [] for t in range(T)}
        mapping = {k: {t: [] for t in range(T)} for k in range(N_)}
        Q = [[0] * T for k in range(N_)]

        QCurr = [[0] * T for k in range(N_)]
        QProd = [[0] * T for k in range(N_)]
        QCommit = [[0] * T for k in range(N_)]
        QStart = [[0] * T for k in range(N_)]
        QCurrNumber = [[0] * T for k in range(N_)]

        for t in range(T):
            for k in range(N_):
                temp = cand_status[k][t]
                if temp not in holdStatus[t]:
                    holdStatus[t].append(temp)
                    holdStatusIndex[t].append(k)
                    solveThisSubproblems[k].append(t)
                    mapping[k][t].append([k, t])
                else:
                    NotSolveThisSubproblems[k].append(t)
                    mapFromScen = holdStatus[t].index(temp)
                    mapping[k][t].append([holdStatusIndex[t][mapFromScen], t])

        # change keys in solveThisSubproblems
        SP_ = {t: [] for t in range(T)}
        for scen in solveThisSubproblems.keys():
            if solveThisSubproblems[scen] == []:
                pass
            else:
                for day in solveThisSubproblems[scen]:
                    SP_[day].append(scen)

        # SP is a dictionary with days (keys) holds lists of scenarios (items) to be solved for that day
        results = {}
        for day in SP_.keys():
            results[day] = Parallel(n_jobs=num_cores)(
                delayed(runParallel)(data['instance'], genValues, lineValues, genScenarios, lineScenarios,
                                     'MIP', 'KT', None, scen, day, data['minFlows'], data['maxFlows'], fixMaint=1)
                for scen in SP_[day])

        for day in range(len(results)):
            for tuple in results[day]:
                scen_ = tuple[0]
                value = tuple[1]
                storeOptForThisSubProblems[scen_][day] = value[0]
                storeCurrForThisSubProblems[scen_][day] = value[1]
                storeProdForThisSubProblems[scen_][day] = value[2]
                storeCommitForThisSubProblems[scen_][day] = value[3]
                storeStartForThisSubProblems[scen_][day] = value[4]
                storeCurrNumberForThisSubProblems[scen_][day] = value[5]

        # MAP ALL OPTIMAL VALUES W.R.T. SCENARIO AND DAY
        for scen in mapping.keys():
            for day in mapping[scen].keys():
                for tuples in mapping[scen][day]:
                    Q[scen][day] = storeOptForThisSubProblems[tuples[0]][tuples[1]]
                    QCurr[scen][day] = storeCurrForThisSubProblems[tuples[0]][tuples[1]]
                    QProd[scen][day] = storeProdForThisSubProblems[tuples[0]][tuples[1]]
                    QCommit[scen][day] = storeCommitForThisSubProblems[tuples[0]][tuples[1]]
                    QStart[scen][day] = storeStartForThisSubProblems[tuples[0]][tuples[1]]
                    QCurrNumber[scen][day] = storeCurrNumberForThisSubProblems[tuples[0]][tuples[1]]

        curtailCostK = [sum(QCurr[k]) for k in range(K)]
        prodCostK = [sum(QProd[k]) for k in range(K)]
        commitCostK = [sum(QCommit[k]) for k in range(K)]
        startCostK = [sum(QStart[k]) for k in range(K)]
        currNumberK = [sum(QCurrNumber[k]) for k in range(K)]

        for k in range(K):
            genPredK.append(gp.quicksum(gen.fix_pred_cost * gen.w[t] for gen in generators if gen.inSubset == 1 for t in
                                        range(gen.zeta[k])).getValue())

            genCorrK.append(gp.quicksum(
                gen.corr_cost * gen.w[t] for gen in generators if gen.inSubset == 1 for t in
                range(gen.zeta[k], gen.T + 1) if gen.zeta[k] != gen.T).getValue())

            linePredK.append(gp.quicksum(
                line.fix_pred_cost * line.z[t] for line in lines if line.inSubset == 1 for t in
                range(line.zeta[k])).getValue())

            lineCorrK.append(gp.quicksum(
                line.corr_cost * line.z[t] for line in lines if line.inSubset == 1 for t in
                range(line.zeta[k], line.T + 1) if line.zeta[k] != line.T).getValue())

        data['genPredCost'] = sum(genPredK) / K

        data['genCorrCost'] = sum(genCorrK) / K

        data['linePredCost'] = sum(linePredK) / K

        data['lineCorrCost'] = sum(lineCorrK) / K

        data['curtailCost'] = sum(curtailCostK) / K

        data['prodCost'] = sum(prodCostK) / K

        data['commitCost'] = sum(commitCostK) / K

        data['startCost'] = sum(startCostK) / K

        # OVERWRITE
        data['startCost'] = sum(currNumberK) / (K * T * S)

        data['sumMaintenance'] = data['genPredCost'] + \
                                 data['genCorrCost'] + data['linePredCost'] + data['lineCorrCost']

        data['sumOperation'] = data['curtailCost'] + \
                               data['prodCost'] + data['commitCost'] + data['startCost']

    else:  # this solves all scenarios together
        model = gp.Model()
        for bus in buses:
            bus.add_operation_variables(model, solveBy)

        for line in lines:
            line.add_operation_variables(model, solveAs, solveBy)

        for gen in generators:
            gen.add_operation_variables(model, solveAs, solveBy)

        for bus in buses:
            bus.add_operation_constraints(model)

        for gen in generators:
            gen.add_operation_constraints(model)
            if gen.inSubset == 1:
                gen.add_type_maintenance_constraints(model)

        for line in lines:
            line.add_operation_constraints(model)
            if line.inSubset == 1:
                line.add_type_maintenance_constraints(model)

        curtailCost = gp.quicksum(
            bus.curtailmentCost * 100 * bus.q[k, t, s] for bus in buses for k in range(K) for t in range(T) for s in
            range(S)) / K

        prodCost = gp.quicksum(gen.Cost.quad * (100 ** 2) * gen.p[k, t, s] * gen.p[k, t, s]
                               + gen.Cost.lin * 100 * gen.p[k, t, s] for gen in generators for k in range(K) for t
                               in range(T) for s in range(S)) / K

        commitCost = gp.quicksum(
            gen.commit_cost * gen.x[k, t, s] for gen in generators for k in range(K) for t in range(T) for s in
            range(S)) / K

        startCost = gp.quicksum(
            gen.Cost.startup * gen.u[k, t, s] for gen in generators for k in range(K) for t in range(T) for s in
            range(S)) / K

        totalCost = curtailCost + prodCost + commitCost + startCost

        model.setObjective(totalCost, gp.GRB.MINIMIZE)

        model.setParam('MIPGap', mipgap)

        model.setParam('TimeLimit', 6 * 60 * 60)

        model.optimize()

        status = model.status
        print('The optimization status is ' + status_code[status])

        data['genPredCost'] = gp.quicksum(
            gen.fix_pred_cost * gen.w[t] for gen in generators if gen.inSubset == 1 for k in range(K) for t in
            range(gen.zeta[k])).getValue() / K

        data['genCorrCost'] = gp.quicksum(
            gen.corr_cost * gen.w[t] for gen in generators if gen.inSubset == 1 for k in range(K) for t in
            range(gen.zeta[k], gen.T + 1) if gen.zeta[k] != gen.T).getValue() / K

        data['linePredCost'] = gp.quicksum(
            line.fix_pred_cost * line.z[t] for line in lines if line.inSubset == 1 for k in range(K) for t in
            range(line.zeta[k])).getValue() / K

        data['lineCorrCost'] = gp.quicksum(
            line.corr_cost * line.z[t] for line in lines if line.inSubset == 1 for k in range(K) for t in
            range(line.zeta[k], line.T + 1) if line.zeta[k] != line.T).getValue() / K

        data['curtailCost'] = curtailCost.getValue()

        data['prodCost'] = prodCost.getValue()

        data['commitCost'] = commitCost.getValue()

        data['startCost'] = startCost.getValue()

        data['sumMaintenance'] = data['genPredCost'] + \
                                 data['genCorrCost'] + data['linePredCost'] + data['lineCorrCost']

        data['sumOperation'] = data['curtailCost'] + \
                               data['prodCost'] + data['commitCost'] + data['startCost']

    return data


def SampleAverageApprox(type, instance, SAAData, genSubset, lineSubset, bigGenSubset, bigLineSubset, lines,
                        generators, M, N_, alpha):
    data = {}

    subGenIndex = [gen.listIndex for gen in generators if gen.inSubset == 1]
    subLineIndex = [line.listIndex for line in lines if line.inSubset == 1]
    subGenIndex_ = [gen.listIndex for gen in generators if gen.inSubset == 0]
    subLineIndex_ = [line.listIndex for line in lines if line.inSubset == 0]

    if type in ['bigSAA', 'Posterior', 'noFailure']:

        genSubset = bigGenSubset
        lineSubset = bigLineSubset

        for j in range(M):
            for line in lines:
                if line.inSubset == 0:
                    SAAData[j]['lineValues'][line.listIndex] = [
                        0, 0, 0, 0, 0, 0, 0, 1]
            for gen in generators:
                if gen.inSubset == 0:
                    SAAData[j]['genValues'][gen.listIndex] = [
                        0, 0, 0, 0, 0, 0, 0, 1]

        for line in lines:
            line.inSubset = 1
        for gen in generators:
            gen.inSubset = 1

    # generate scenarios
    bigSampleGenScenarios = generateScenarios(genSubset, generators, N_, 1)
    bigSampleLineScenarios = generateScenarios(lineSubset, lines, N_, 1)

    # assign scenarios
    assignScenarios(generators, bigSampleGenScenarios)
    assignScenarios(lines, bigSampleLineScenarios)

    for j in range(M):
        data[j] = {}

        for gen in generators:
            if gen.inSubset == 1:
                gen.w = SAAData[j]['genValues'][gen.listIndex]

        for line in lines:
            if line.inSubset == 1:
                line.z = SAAData[j]['lineValues'][line.listIndex]

        data[j]['lineValues'] = SAAData[j]['lineValues']
        data[j]['genValues'] = SAAData[j]['genValues']

        # count failures
        data[j]['Failures_Gen'] = []
        data[j]['Failures_Line'] = []

        data[j]['Failures_Gen_'] = []
        data[j]['Failures_Line_'] = []

        T = opts['T']
        for i in subGenIndex:
            count = 0
            for k in range(N_):
                if generators[i].zeta[k] != T:
                    if generators[i].zeta[k] <= data[j]['genValues'][i].index(1):
                        count += 1
            data[j]['Failures_Gen'].append(count)

        data[j]['avg_Gen'] = sum(data[j]['Failures_Gen']) / N_

        for i in subLineIndex:
            count = 0
            for k in range(N_):
                if lines[i].zeta[k] != T:
                    if lines[i].zeta[k] <= data[j]['lineValues'][i].index(1):
                        count += 1
            data[j]['Failures_Line'].append(count)

        data[j]['avg_Line'] = sum(data[j]['Failures_Line']) / N_

        if type in ['bigSAA', 'Posterior', 'noFailure']:
            for i in subGenIndex_:
                count = 0
                for k in range(N_):
                    if generators[i].zeta[k] != T:
                        if generators[i].zeta[k] <= data[j]['genValues'][i].index(1):
                            count += 1
                data[j]['Failures_Gen_'].append(count)

            data[j]['avg_Gen_'] = sum(data[j]['Failures_Gen_']) / N_

            for i in subLineIndex_:
                count = 0
                for k in range(N_):
                    if lines[i].zeta[k] != T:
                        if lines[i].zeta[k] <= data[j]['lineValues'][i].index(1):
                            count += 1
                data[j]['Failures_Line_'].append(count)

            data[j]['avg_Line_'] = sum(data[j]['Failures_Line_']) / N_

        data[j]['maint_costs_gen'] = []
        data[j]['maint_costs_line'] = []
        data[j]['maint_costs'] = []

        if withDynamic == True:
            type_ = 'Dynamic'
        if withFixed == True:
            type_ = 'Fixed'

        for k in range(N_):
            data[j]['maint_costs_gen'].append(gp.quicksum(gen.add_maintenance_costs(
                k, type_) for gen in generators if gen.inSubset == 1))
            data[j]['maint_costs_line'].append(gp.quicksum(
                line.add_maintenance_costs(k, type_) for line in lines if line.inSubset == 1))
            maint_cost = data[j]['maint_costs_gen'][k].getValue(
            ) + data[j]['maint_costs_line'][k].getValue()
            data[j]['maint_costs'].append(maint_cost)

        data[j]['expected_gen_cost'] = sum(
            data[j]['maint_costs_gen']).getValue() / 1000
        data[j]['expected_line_cost'] = sum(
            data[j]['maint_costs_line']).getValue() / 1000
        data[j]['expected_maint_cost'] = (
                data[j]['expected_gen_cost'] + data[j]['expected_line_cost'])

        data[j]['fixed_maint_costs_gen'] = []
        data[j]['fixed_maint_costs_line'] = []
        data[j]['fixed_maint_costs'] = []

        type_ = 'Fixed'
        for k in range(N_):
            data[j]['fixed_maint_costs_gen'].append(gp.quicksum(
                gen.add_maintenance_costs(k, type_) for gen in generators if gen.inSubset == 1))
            data[j]['fixed_maint_costs_line'].append(gp.quicksum(
                line.add_maintenance_costs(k, type_) for line in lines if line.inSubset == 1))
            maint_cost = data[j]['fixed_maint_costs_gen'][k].getValue(
            ) + data[j]['fixed_maint_costs_line'][k].getValue()
            data[j]['fixed_maint_costs'].append(maint_cost)

        data[j]['fixed_expected_gen_cost'] = sum(
            data[j]['fixed_maint_costs_gen']).getValue() / 1000
        data[j]['fixed_expected_line_cost'] = sum(
            data[j]['fixed_maint_costs_line']).getValue() / 1000
        data[j]['fixed_expected_maint_cost'] = (
                data[j]['fixed_expected_gen_cost'] + data[j]['fixed_expected_line_cost'])

        opts['K'] = N_

        for k in range(N_):
            findCompUpperBoundsOnScenario(
                generators, SAAData[j]['genValues'], k)
            findCompUpperBoundsOnScenario(lines, SAAData[j]['lineValues'], k)

        cand_status = findCandStatus(generators, lines)

        # find subproblems to be solved
        solveThisSubproblems = {k: [] for k in range(N_)}
        storeOptForThisSubProblems = {
            k: {t: 0 for t in range(T)} for k in range(N_)}
        NotSolveThisSubproblems = {k: [] for k in range(N_)}
        holdStatus = {t: [] for t in range(T)}
        holdStatusIndex = {t: [] for t in range(T)}
        mapping = {k: {t: [] for t in range(T)} for k in range(N_)}
        Q = [[0] * T for k in range(N_)]
        for t in range(T):
            for k in range(N_):
                temp = cand_status[k][t]
                if temp not in holdStatus[t]:
                    holdStatus[t].append(temp)
                    holdStatusIndex[t].append(k)
                    solveThisSubproblems[k].append(t)
                    mapping[k][t].append([k, t])
                else:
                    NotSolveThisSubproblems[k].append(t)
                    mapFromScen = holdStatus[t].index(temp)
                    mapping[k][t].append([holdStatusIndex[t][mapFromScen], t])

        # change keys in solveThisSubproblems
        SP_ = {t: [] for t in range(T)}
        for scen in solveThisSubproblems.keys():
            if solveThisSubproblems[scen] == []:
                pass
            else:
                for day in solveThisSubproblems[scen]:
                    SP_[day].append(scen)

        # SP is a dictionary with days (keys) holds lists of scenarios (items) to be solved for that day
        results = {}
        for day in SP_.keys():
            results[day] = Parallel(n_jobs=num_cores)(
                delayed(runParallel)(instance, SAAData[j]['genValues'], SAAData[j]['lineValues'], bigSampleGenScenarios,
                                     bigSampleLineScenarios,
                                     'MIP', 'KT', discardBy[0], scen, day, SAAData[j]['minFlows'],
                                     SAAData[j]['maxFlows'], fixMaint=0)
                for scen in SP_[day])

        for day in range(len(results)):
            for tuple in results[day]:
                scen_ = tuple[0]
                value = tuple[1]
                storeOptForThisSubProblems[scen_][day] = value

        # MAP ALL OPTIMAL VALUES W.R.T. SCENARIO AND DAY
        for scen in mapping.keys():
            for day in mapping[scen].keys():
                for tuples in mapping[scen][day]:
                    Q[scen][day] = storeOptForThisSubProblems[tuples[0]][tuples[1]]

        data[j]['operation_costs'] = []
        for k in range(N_):
            data[j]['operation_costs'].append(sum(Q[k]))

        data[j]['expected_operation_cost'] = sum(
            data[j]['operation_costs']) / N_
        # total cost for big sample
        data[j]['expected_total_cost'] = data[j]['expected_gen_cost'] + \
                                         data[j]['expected_line_cost'] + data[j]['expected_operation_cost']

        # total cost for big sample
        data[j]['fixed_expected_total_cost'] = data[j]['fixed_expected_gen_cost'] + \
                                               data[j]['fixed_expected_line_cost'] + \
                                               data[j]['expected_operation_cost']

    # obtain the best (minimum) upper bound estimate
    mean_estimate_UB, min_key = math.inf, None
    for j in range(M):
        if data[j]['expected_total_cost'] < mean_estimate_UB:
            min_key = j
            mean_estimate_UB = data[j]['expected_total_cost']

    data['best'] = data[min_key]
    data['best']['minKey'] = min_key
    data['best']['instance'] = instance

    minFlows, maxFlows = solveDiscardLinesModel(
        buses, lines, generators, 'MIP', 'KT', 'DISCARD-M2')
    discardSomeLines(discardBy, minFlows, maxFlows, lines)
    data['best']['minFlows'], data['best']['maxFlows'] = minFlows, maxFlows

    data['best'] = solveModelFixedMaintenance(
        data['best'], generators, lines, buses, lineSubset, genSubset, SAA=N_)

    sumSquares = 0
    for k in range(N_):
        sumSquares += (data[min_key]['maint_costs'][k] + data[min_key]
        ['operation_costs'][k] - mean_estimate_UB) ** 2

    # variance of the true upper bound estimate
    var_estimate_UB = sumSquares / (N_ - 1)

    # test-statistics (a/2)
    z = stats.norm.ppf(1 - alpha / 2)
    t = stats.t.ppf(1 - alpha / 2, M - 1)

    # (1-a)% CI for the upper bound estimate
    UBlowerCI = mean_estimate_UB - (z) * (var_estimate_UB / N_) ** 0.5
    UBupperCI = mean_estimate_UB + (z) * (var_estimate_UB / N_) ** 0.5

    UB, LB = None, None
    LB_LB = None
    LB_UB = None
    if type in ['Normal', 'bigSAA']:
        # mean and variance of the true lower bound estimate
        mean_estimate_LB_LB = sum(SAAData[j]['LB'] for j in range(M)) / M
        var_estimate_LB_LB = sum(
            (SAAData[j]['LB'] - mean_estimate_LB_LB) ** 2 for j in range(M)) / (M - 1)
        # (1-a)% CI for the lower bound estimate
        LBlowerCI_LB = mean_estimate_LB_LB - \
                       (t) * (var_estimate_LB_LB / M) ** 0.5
        LBupperCI_LB = mean_estimate_LB_LB + \
                       (t) * (var_estimate_LB_LB / M) ** 0.5
        LB_LB = (LBlowerCI_LB, LBupperCI_LB)

        # mean and variance of the true lower bound estimate
        mean_estimate_LB_UB = sum(SAAData[j]['UB'] for j in range(M)) / M
        var_estimate_LB_UB = sum(
            (SAAData[j]['UB'] - mean_estimate_LB_UB) ** 2 for j in range(M)) / (M - 1)
        # (1-a)% CI for the lower bound estimate
        LBlowerCI_UB = mean_estimate_LB_UB - \
                       (t) * (var_estimate_LB_UB / M) ** 0.5
        LBupperCI_UB = mean_estimate_LB_UB + \
                       (t) * (var_estimate_LB_UB / M) ** 0.5
        LB_UB = (LBlowerCI_UB, LBupperCI_UB)

    UB = (UBlowerCI, UBupperCI)

    gap_LB = None
    gap_UB = None
    gap = None

    if type in ['Normal', 'bigSAA']:
        gap_LB = ((UBupperCI - LBlowerCI_LB) / UBlowerCI) * 100
        gap_UB = ((UBupperCI - LBlowerCI_UB) / UBlowerCI) * 100

    data['best']['UB'] = UB
    data['best']['LB_LB'] = LB_LB
    data['best']['LB_UB'] = LB_UB

    data['best']['gap'] = gap

    data['best']['gap_LB'] = gap_LB
    data['best']['gap_UB'] = gap_UB

    data['best']['gen_subset'] = subGenIndex
    data['best']['line_subset'] = subLineIndex

    if type in ['bigSAA', 'Posterior', 'noFailure']:
        data['best']['gen_subset__'] = subGenIndex_
        data['best']['line_subset__'] = subLineIndex_

    return data['best']


def SolveOptModel(instance, seed, solveAs, solveBy, discardBy=None, case=None):
    buses, lines, generators, costs = Input(instance)
    MP, SP, Q, L, data = None, None, None, None, None

    np.random.seed(seedLine)

    bigLineSubset, lineSubset, signalDataLine = chooseSubset(
        lines, lineScen, ratio=1, type='line')

    np.random.seed(seedGen)

    bigGenSubset, genSubset, signalDataGen = chooseSubset(
        generators, genScen, ratio=1, type='gen')

    print('scenGen: ' + str(genSubset))
    print('scenLine: ' + str(lineSubset))

    if solveBy == 'single':
        if withSafe == True:
            type_ = 'socp'
        if withExact == True:
            type_ = 'exact'
        else:
            type_ = None

        buses, lines, generators, data = SingleStageModel(seed, buses, lines, generators, lineSubset, genSubset,
                                                          solveAs,
                                                          solveBy, type=type_)

    else:
        MP, SP, Q, L, data = DecompModel(instance, seed, buses, lines, generators, lineSubset, genSubset,
                                         solveAs, solveBy, discardBy, case)

    return buses, lines, generators, lineSubset, genSubset, MP, SP, Q, L, data, signalDataLine, signalDataGen, bigLineSubset, bigGenSubset


if __name__ == '__main__':

    withPosterior = False
    withNoFailure = False

    num_cores = 32  # multiprocessing.cpu_count()

    mipgap = 10 ** -4

    # # #
    seedGen = 104
    seedLine = 272
    instanceList = ['case9.m']

    # seedGen = 104
    # seedLine = 104
    # instanceList = ['case39.m']

    # seedGen = 86
    # seedLine = 53
    # instanceList = ['case57.m']

    # seedGen = 423
    # seedLine = 276
    # instanceList = ['case118Blumsack.m']

    for i in range(2, 8):  # in range(3, 8):

        # if i == 0:
        #     withDynamic = True
        #     withFixed = False

        #     withSafe = True
        #     withExact = None

        # if i == 1:
        #     withDynamic = True
        #     withFixed = False

        #     withSafe = False
        #     withExact = True

        if i == 2 or i == 5:
            withDynamic = False
            withFixed = True

            withSafe = True
            withExact = None

            includeChanceCons = True

        if i == 3 or i == 6:
            withDynamic = False
            withFixed = True

            withSafe = False
            withExact = True

            includeChanceCons = True

        if i == 4 or i == 7:
            withDynamic = False
            withFixed = True

            withSafe = True
            withExact = None

            includeChanceCons = False

        if i <= 4:
            includeLine = 1

        if i >= 5:
            includeLine = 0

        solveAs = 'MIP'

        scenarios = [50] * 5

        seed = [16, 24, 40, 56, 88]

        discardBy = ['DISCARD-M2']

        solver = {'KT': ['KT+++']}
        buses, lines, generators, lineSubset, genSubset, MP, SP, Q, L, data, SAAData, signalDataLine, signalDataGen, bigLineSubset, bigGenSubset = getRun(
            instanceList, seed, scenarios, solveAs, discardBy, solver)

        case = instanceList[0]
        M = 5
        N_ = 1000
        date = datetime.datetime.now()
        time_ = date.strftime("%X").replace(':', '-')

        with open('SAA_Results_' + str(i) + '-' + str(case) + '_' + str(time_) + '.csv', 'x', newline='') as file:
            fieldnames = ['withFixed', 'withSafe', 'case', 'type', 'M', 'N_',

                          'gen_subset', 'line_subset',
                          'Failures_Gen', 'Failures_Line',
                          'avg_Gen', 'avg_Line',

                          'gen_subset__', 'line_subset__',
                          'Failures_Gen_', 'Failures_Line_',
                          'avg_Gen_', 'avg_Line_',

                          'expected_gen_cost', 'expected_line_cost',
                          'expected_maint_cost', 'expected_operation_cost', 'expected_total_cost',

                          'fixed_expected_gen_cost', 'fixed_expected_line_cost',
                          'fixed_expected_maint_cost', 'fixed_expected_total_cost',

                          'LB_lower_LB', 'LB_upper_LB', 'LB_lower_UB', 'LB_upper_UB',
                          'UB_lower', 'UB_upper', 'gap', 'gap_LB', 'gap_UB', 'minReplicate',
                          'genPredCost', 'genCorrCost', 'linePredCost', 'lineCorrCost',
                          'curtailCost', 'prodCost', 'commitCost', 'startCost',
                          'sumMaintenance', 'sumOperation']

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            type_ = 'bigSAA'
            data = SampleAverageApprox(type_, case, SAAData, genSubset, lineSubset,
                                       bigGenSubset, bigLineSubset, lines, generators, M, N_, alpha=0.05)

            writer.writerow({'withFixed': str(withFixed), 'withSafe': str(withSafe),
                             'case': case, 'type': type_, 'M': M, 'N_': N_,

                             'gen_subset': data['gen_subset'], 'line_subset': data['line_subset'],
                             'gen_subset__': data['gen_subset__'], 'line_subset__': data['line_subset__'],

                             'Failures_Gen': data['Failures_Gen'], 'Failures_Line': data['Failures_Line'],
                             'avg_Gen': data['avg_Gen'], 'avg_Line': data['avg_Line'],

                             'Failures_Gen_': data['Failures_Gen_'], 'Failures_Line_': data['Failures_Line_'],
                             'avg_Gen_': data['avg_Gen_'], 'avg_Line_': data['avg_Line_'],

                             'expected_gen_cost': data['expected_gen_cost'],
                             'expected_line_cost': data['expected_line_cost'],
                             'expected_maint_cost': data['expected_maint_cost'],
                             'expected_operation_cost': data['expected_operation_cost'],
                             'expected_total_cost': data['expected_total_cost'],

                             'fixed_expected_gen_cost': data['fixed_expected_gen_cost'],
                             'fixed_expected_line_cost': data['fixed_expected_line_cost'],
                             'fixed_expected_maint_cost': data['fixed_expected_maint_cost'],
                             'fixed_expected_total_cost': data['fixed_expected_total_cost'],

                             'LB_lower_LB': data['LB_LB'][0], 'LB_upper_LB': data['LB_LB'][1],
                             'LB_lower_UB': data['LB_UB'][0], 'LB_upper_UB': data['LB_UB'][1],
                             'UB_lower': data['UB'][0], 'UB_upper': data['UB'][1],
                             'gap_LB': data['gap_LB'], 'gap_UB': data['gap_UB'],
                             'minReplicate': data['minKey'],
                             'genPredCost': data['genPredCost'], 'genCorrCost': data['genCorrCost'],
                             'linePredCost': data['linePredCost'],
                             'lineCorrCost': data['lineCorrCost'],
                             'curtailCost': data['curtailCost'], 'prodCost': data['prodCost'],
                             'commitCost': data['commitCost'], 'startCost': data['startCost'],
                             'sumMaintenance': data['sumMaintenance'], 'sumOperation': data['sumOperation']
                             })

            file.flush()

# # single: single stage

# # 'withoutStatus; for scenario decomp.

# # std_L   : classical L-shaped (one cut)
# # std-LK  : K  - L-shaped (multi cut)
# # std-LKT : KT - L-shaped (multi cuts)

# # K+     :  K - improved cut
# # K++    :  K - much(?) improved cut

# # std-LKsumT   : KT - standard L-shaped (Q sum over T, L_k)

# # KT++   : KT - much improved cut
# # KT++T  : KT - much improved cut (Q sum over T, L_k)
# # KT+++  : KT - very much improved cut

# # combine-KT, 'parallel' : KT+++ & KT++T
