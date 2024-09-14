# %%
import os, sys


current_dir = os.getcwd()
sys.path.append(f'{current_dir}/flowsheet_models')
sys.path.append(f'{current_dir}/stress_creep_submodels')


import time
from enum import Enum
import pandas as pd
import numpy as np

import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.calc_var_value import calculate_variable_from_constraint

import idaes
import idaes.core.util.scaling as iscale
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes.core.solvers import petsc
import idaes.logger as idaeslog
import idaes.core.util.model_serializer as ms
from idaes.core.util.model_statistics import degrees_of_freedom as dof
from soec_dynamic_flowsheet_mk2 import SoecStandaloneFlowsheet
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from idaes.models.control.controller import ControllerType, ControllerMVBoundType, ControllerAntiwindupType
from save_results import (save_results, save_results_full_discretization)
from idaes.models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom as dof
# from idaes.core.util.initialization import initialize_by_time_element
from functions import generate_corrected_ramp_setpoints
from idaes.core.util.math import smooth_abs, smooth_min
from stress_functions_Zhao_thin import (_make_thermal_stress,
                                        _save_temperature_stress_flowsheet)

from pyomo.core.expr import Expr_if

# import logging and remove warning messages
import logging
# logging.disable(logging.WARNING)
# copy from Nishant's code
logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)
logging.getLogger('idaes.core.util.scaling').setLevel(logging.ERROR)
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
logging.getLogger('idaes.models.unit_models.heat_exchanger_1D').setLevel(logging.ERROR)

class OperatingScenario(Enum):
    maximum_production = 1
    minimum_production = 2
    power_mode = 3
    neutral = 4
    soec_fifty_fifty = 5
    sofc_fifty_fifty = 6
    soec_no_current = 7
    sofc_no_current = 8
    multi_step = 9

keep_controllers = False

def scale_indexed_constraint(con, sf):
    for idx, c in con.items():
        iscale.constraint_scaling_transform(c, sf)


def create_ramping_eqns(fs, vars, time_scaling_factor=1):
    def rule_ramp(b, t, dvdt, v_ramp):
        return dvdt[t] == v_ramp[t]

    t0 = fs.time.first()

    for var in vars:
        var.unfix()
        shortname = var.name.split(".")[-1]
        blk = var.parent_block()
        dvdt = DerivativeVar(var, wrt=fs.time, initialize=0)
        setattr(blk, "d" + shortname + "dt", dvdt)
        v_ramp = pyo.Var(fs.time, initialize=0)
        setattr(blk, shortname + "_ramp_rate", v_ramp)
        v_ramp_eqn = pyo.Constraint(fs.time, rule=lambda b, t: rule_ramp(b, t, dvdt, v_ramp))
        setattr(blk, shortname + "_ramp_rate_eqn", v_ramp_eqn)
        for t in fs.time:
            sv = iscale.get_scaling_factor(var[t], default=1)
            iscale.set_scaling_factor(dvdt[t], sv / time_scaling_factor)
            iscale.set_scaling_factor(v_ramp[t], sv / time_scaling_factor)
            iscale.constraint_scaling_transform(v_ramp_eqn[t], sv / time_scaling_factor)

        v_ramp_eqn[t0].deactivate()
        v_ramp[t0].fix(0)


if __name__ == "__main__":
    def set_indexed_variable_bounds(var, bounds):
        for idx, subvar in var.items():
            subvar.bounds = bounds


    from idaes.core.solvers import use_idaes_solver_configuration_defaults

    use_idaes_solver_configuration_defaults()
    idaes.cfg.ipopt.options.tol = 1e-8
    idaes.cfg.ipopt.options.nlp_scaling_method = "user-scaling"
    idaes.cfg.ipopt["options"]["linear_solver"] = "ma57"
    idaes.cfg.ipopt.options.OF_ma57_automatic_scaling = "yes"
    idaes.cfg.ipopt["options"]["max_iter"] = 400
    idaes.cfg.ipopt["options"]["halt_on_ampl_error"] = "no"


    dynamic_simulation = True
    optimize_steady_state = True
    operating_scenario = OperatingScenario.maximum_production
    adding_new_time_elements = True
    # petsc_solver = True

    m = pyo.ConcreteModel()

    if dynamic_simulation:
        # Define the initial process times
        t_start = 1 * 60 * 60  # Start time in seconds
        t_ramp = 5 * 60  # Ramp time in seconds
        # t_ramp = 50 * 60  # Ramp time in seconds increase ten times
        t_settle = 5 * 60 * 60  # Settle time in seconds
        t_end = 3 * 60 * 60  # End time in seconds

        dt_set = [t_start, t_ramp, t_settle, t_ramp, t_end]

        # dt_set = [t_start, t_ramp, t_multi_step, t_ramp, t_settle, t_ramp, t_multi_step, t_ramp, t_end]

        # dt_set = [t_start, t_ramp, t_ramp, t_ramp, t_settle, t_ramp, t_ramp, t_ramp, t_end]

        # dt_set = [t_start, t_ramp, t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp,
        #            t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp,
        #            t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp,
        #            t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp, t_settle, t_ramp,
        #            t_end]

        # original time set before adding new time elements manually
        if adding_new_time_elements:
            time_set_org = [sum(dt_set[:j]) for j in range(len(dt_set) + 1)]

            time_elements = 20
            number_elements_first_interval = int(3 / 30 * time_elements)
            number_elements_second_interval_ramp_down = int(3 / 30 * time_elements)
            number_elements_third_interval = int(12 / 30 * time_elements)
            number_elements_fourth_interval = int(9 / 30 * time_elements)
            number_elements_second_interval_ramp_up = int(3 / 30 * time_elements)

            # create a new list of number_elements_ramping times between each setpoint change
            time_set_add1 = [time_set_org[0] + j * (time_set_org[1] - time_set_org[0]) / number_elements_first_interval
                             for j in range(number_elements_first_interval)]
            time_set_add2 = [time_set_org[1] + j * (time_set_org[2] - time_set_org[1]) / number_elements_second_interval_ramp_down
                             for j in range(number_elements_second_interval_ramp_down)]
            time_set_add3 = [time_set_org[2] + j * (time_set_org[3] - time_set_org[2]) / number_elements_third_interval
                             for j in range(number_elements_third_interval)]
            time_set_add4 = [time_set_org[3] + j * (time_set_org[4] - time_set_org[3]) / number_elements_second_interval_ramp_up
                             for j in range(number_elements_second_interval_ramp_up)]
            time_set_add5 = [time_set_org[4] + j * (time_set_org[5] - time_set_org[4]) / number_elements_fourth_interval
                             for j in range(number_elements_fourth_interval)]

            time_set = (time_set_org + time_set_add1[1:] + time_set_add2[1:] + time_set_add3[1:] + time_set_add4[1:] +
                        time_set_add5[1:])

            time_set.sort()
        else:
            time_set = [sum(dt_set[:j]) for j in range(len(dt_set) + 1)]

        if adding_new_time_elements:
            # Generate corrected ramp setpoints based on the time set
            ramp_setpoints = generate_corrected_ramp_setpoints(time_set)
        else:

            ramp_setpoints = [
                "maximum_H2", "maximum_H2",
                "minimum_H2", "minimum_H2",
                "maximum_H2", "maximum_H2",
            ]

        # ramp_setpoints = [
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        # ]
        # ramp_setpoints = [
        #     "maximum_H2", "maximum_H2",
        #     "soec_no_current", "sofc_no_current",
        #     "power", "power",
        #     "sofc_no_current", "soec_no_current",
        #     "maximum_H2", "maximum_H2"
        # ]
        # ramp_setpoints = [
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        #     "power", "power",
        #     "maximum_H2", "maximum_H2",
        # ]
        step_setpoints = ramp_setpoints

        m.fs = SoecStandaloneFlowsheet(
            dynamic=True,
            time_set=time_set,
            time_units=pyo.units.s,
            thin_electrolyte_and_oxygen_electrode=True,
            has_gas_holdup=False,
            include_interconnect=True,
        )
    else:
        m.fs = SoecStandaloneFlowsheet(
            dynamic=False,
            thin_electrolyte_and_oxygen_electrode=True,
            has_gas_holdup=False,
            include_interconnect=True,
        )

    iscale.calculate_scaling_factors(m)
    solver = pyo.SolverFactory("ipopt")
    for t in m.fs.time:
        m.fs.condenser_flash.control_volume.properties_in[t].flow_mol_phase["Liq"].domain = pyo.Reals
        m.fs.condenser_flash.control_volume.properties_in[t].flow_mol_phase["Liq"].bounds = (None, None)
        m.fs.condenser_flash.control_volume.properties_in[t].phase_frac["Liq"].domain = pyo.Reals
        m.fs.condenser_flash.control_volume.properties_in[t].phase_frac["Liq"].bounds = (None, None)
        for var in [
            m.fs.condenser_flash.control_volume.properties_in[t].log_mole_frac_tdew,
            m.fs.condenser_flash.control_volume.properties_in[t]._mole_frac_tdew,
        ]:
            for idx in var.index_set():
                var[idx].domain = pyo.Reals
                var[idx].bounds = (None, None)

    if dynamic_simulation:
        # m.fs.deactivate_shortcut()

        antiwindup = ControllerAntiwindupType.BACK_CALCULATION
        inner_controller_pairs = ComponentMap()
        inner_controller_pairs[m.fs.feed_heater.electric_heat_duty] = (
            "feed_heater_inner_controller",
            m.fs.soc_module.fuel_inlet.temperature,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        inner_controller_pairs[m.fs.sweep_heater.electric_heat_duty] = (
            "sweep_heater_inner_controller",
            m.fs.soc_module.oxygen_inlet.temperature,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        m.fs.add_controllers(inner_controller_pairs)

        variable_pairs = ComponentMap()
        variable_pairs[m.fs.feed_heater_inner_controller.setpoint] = (
            "feed_heater_outer_controller",
            m.fs.soc_module.fuel_outlet.temperature,
            ControllerType.P,
            ControllerMVBoundType.NONE,
            ControllerAntiwindupType.NONE,
        )
        variable_pairs[m.fs.sweep_heater_inner_controller.setpoint] = (
            "sweep_heater_outer_controller",
            m.fs.soc_module.oxygen_outlet.temperature,
            ControllerType.P,
            ControllerMVBoundType.NONE,
            ControllerAntiwindupType.NONE,
        )
        variable_pairs[m.fs.soc_module.potential_cell] = (
            "voltage_controller",
            m.fs.soc_module.fuel_outlet_mole_frac_comp_H2,
            ControllerType.PI,
            ControllerMVBoundType.SMOOTH_BOUND,
            antiwindup,
        )
        # variable_pairs[m.fs.soc_module.total_current] = (
        #     "current_controller",
        #     m.fs.soc_module.fuel_outlet_mole_frac_comp_H2,
        #     ControllerType.PI,
        #     ControllerMVBoundType.NONE,
        #     # antiwindup,
        #     ControllerAntiwindupType.NONE,
        # )
        variable_pairs[m.fs.sweep_blower.inlet.flow_mol] = (
            "sweep_blower_controller",
            m.fs.stack_core_temperature,
            ControllerType.P,
            ControllerMVBoundType.SMOOTH_BOUND,
            # antiwindup,
            ControllerAntiwindupType.NONE,
        )
        variable_pairs[m.fs.makeup_mix.makeup.flow_mol] = (
            "h2_production_rate_controller",
            m.fs.h2_mass_production,
            ControllerType.P,
            ControllerMVBoundType.SMOOTH_BOUND,
            ControllerAntiwindupType.NONE,
            # antiwindup,
        )
        # variable_pairs[m.fs.sweep_recycle_split.recycle_ratio] = (
        #     "sweep_recycle_controller",
        #     m.fs.soc_module.oxygen_outlet.temperature,
        #     ControllerType.P,
        #     ControllerMVBoundType.SMOOTH_BOUND,
        #     ControllerAntiwindupType.NONE,
        # )
        # variable_pairs[m.fs.condenser_hx.cold_side_inlet.flow_mol] = (
        #     "condenser_controller",
        #     m.fs.condenser_hx.hot_side_outlet.temperature,
        #     ControllerType.P,
        #     ControllerMVBoundType.SMOOTH_BOUND,
        # )
        m.fs.add_controllers(variable_pairs)

        # K = 0
        K = 10e4
        tau_I = 15 * 60
        tau_D = 5 * 60
        m.fs.feed_heater_inner_controller.gain_p.fix(K)
        m.fs.feed_heater_inner_controller.gain_i.fix(K / tau_I)
        # m.fs.feed_heater_inner_controller.gain_d.fix(K * tau_D)
        m.fs.feed_heater_inner_controller.mv_lb = 0
        m.fs.feed_heater_inner_controller.mv_ub = 10e6
        # # change the heater duty upper bound to 20 MW
        # m.fs.feed_heater_inner_controller.mv_ub = 20e6
        m.fs.feed_heater_inner_controller.smooth_eps = 1000
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.feed_heater_inner_controller.gain_b.fix(0.5 / tau_I)

        # K = 0
        K = 20e4
        tau_I = 15 * 60
        tau_D = 5 * 60
        m.fs.sweep_heater_inner_controller.gain_p.fix(K)
        m.fs.sweep_heater_inner_controller.gain_i.fix(K / tau_I)
        # m.fs.sweep_heater_inner_controller.gain_d.fix(K * tau_D)
        m.fs.sweep_heater_inner_controller.mv_lb = 0
        m.fs.sweep_heater_inner_controller.mv_ub = 10e6
        # # change the heater duty upper bound to 20 MW
        # m.fs.sweep_heater_inner_controller.mv_ub = 20e6
        m.fs.sweep_heater_inner_controller.smooth_eps = 1000
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.sweep_heater_inner_controller.gain_b.fix(0.5 / tau_I)

        K = 0.75
        # K = 0
        tau_I = 60 * 60
        m.fs.feed_heater_outer_controller.gain_p.fix(K)
        # m.fs.feed_heater_outer_controller.gain_i.fix(K/tau_I)
        # m.fs.feed_heater_outer_controller.mv_lb = 0
        # m.fs.feed_heater_outer_controller.mv_ub = 4e6
        # m.fs.feed_heater_outer_controller.smooth_eps = 0.1

        K = 0.75
        # K = 0
        tau_I = 60 * 60
        m.fs.sweep_heater_outer_controller.gain_p.fix(K)
        # m.fs.sweep_heater_outer_controller.gain_i.fix(K/tau_I)
        # m.fs.sweep_heater_outer_controller.mv_lb = 0
        # m.fs.sweep_heater_outer_controller.mv_ub = 12e6
        # m.fs.sweep_heater_outer_controller.smooth_eps = 0.01

        K = -2
        tau_I = 240
        m.fs.voltage_controller.gain_p.fix(K)
        m.fs.voltage_controller.gain_i.fix(K / tau_I)
        m.fs.voltage_controller.mv_lb = 0.7
        m.fs.voltage_controller.mv_ub = 1.6
        m.fs.voltage_controller.smooth_eps = 0.01
        if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
            m.fs.voltage_controller.gain_b.fix(0.5 / tau_I)
        # K = -2e7
        # tau_I = 240
        # m.fs.current_controller.gain_p.fix(K)
        # m.fs.current_controller.gain_i.fix(K/tau_I)
        # # m.fs.current_controller.mv_lb = 0.7
        # # m.fs.current_controller.mv_ub = 1.6
        # # m.fs.current_controller.smooth_eps = 0.01
        # # if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
        # #     m.fs.current_controller.gain_b.fix(0.5/tau_I)

        K = -50
        tau_I = 40 * 60
        m.fs.sweep_blower_controller.gain_p.fix(K)
        # m.fs.sweep_blower_controller.gain_i.fix(K / tau_I)
        m.fs.sweep_blower_controller.mv_lb = 1500
        m.fs.sweep_blower_controller.mv_ub = 10000
        m.fs.sweep_blower_controller.smooth_eps = 10
        # if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
        #     m.fs.sweep_blower_controller.gain_b.fix(0.5/tau_I)

        # K = 0
        # # K = -0.5 * 0.025
        # # tau_I = 1200
        # m.fs.sweep_recycle_controller.gain_p.fix(K)
        # #m.fs.sweep_recycle_controller.gain_i.fix(K / tau_I)
        # m.fs.sweep_recycle_controller.mv_lb = 0.01
        # m.fs.sweep_recycle_controller.mv_ub = 2
        # m.fs.sweep_recycle_controller.smooth_eps = 1e-3

        K = 200
        tau_I = 20 * 60
        m.fs.h2_production_rate_controller.gain_p.fix(K)
        # m.fs.h2_production_rate_controller.gain_i.fix(K / tau_I)
        m.fs.h2_production_rate_controller.mv_lb = 1
        m.fs.h2_production_rate_controller.mv_ub = 1500
        m.fs.h2_production_rate_controller.smooth_eps = 1
        # if antiwindup == ControllerAntiwindupType.BACK_CALCULATION:
        #     m.fs.h2_production_rate_controller.gain_b.fix(0.5/tau_I)

        step_mvs = ComponentSet([])
        step_mvs.add(m.fs.voltage_controller.mv_ref)
        step_mvs.add(m.fs.voltage_controller.setpoint)
        # step_mvs.add(m.fs.current_controller.mv_ref)
        # step_mvs.add(m.fs.current_controller.setpoint)
        step_mvs.add(m.fs.makeup_mix.makeup_mole_frac_comp_H2)
        step_mvs.add(m.fs.makeup_mix.makeup_mole_frac_comp_H2O)
        # step_mvs.add(m.fs.h2_production_rate_controller.mv_ref)
        step_mvs.add(m.fs.h2_production_rate_controller.setpoint)
        step_mvs.add(m.fs.h2_production_rate_controller.gain_p)
        # step_mvs.add(m.fs.h2_production_rate_controller.gain_i)

        ramp_mvs = m.fs.manipulated_variables - step_mvs

        create_ramping_eqns(m.fs, ramp_mvs, 1)

        for ctrl in m.fs.controller_set:
            iscale.calculate_scaling_factors(ctrl)
        for ctrl in m.fs.controller_set:
            iscale.calculate_scaling_factors(ctrl)

        time_nfe = len(m.fs.time) - 1
        pyo.TransformationFactory("dae.finite_difference").apply_to(
            m.fs, nfe=time_nfe, wrt=m.fs.time, scheme="BACKWARD"
        )
        # pyo.TransformationFactory("dae.collocation").apply_to(
        #     m.fs, nfe=time_nfe, ncp=3, wrt=m.fs.time, scheme="LAGRANGE-RADAU"
        # )
        iscale.scale_time_discretization_equations(m, m.fs.time, 1 / (60 * 5))

        operating_scenario_name = operating_scenario.name
        if operating_scenario == OperatingScenario.minimum_production:
            ms.from_json(m, fname="json_initial_file/min_production.json.gz", wts=ms.StoreSpec.value())
            # ms.from_json(m, fname=f"json_initial_file/{operating_scenario_name}.json.gz")
        elif operating_scenario == OperatingScenario.maximum_production:
            ms.from_json(m, fname="json_initial_file/max_production.json.gz", wts=ms.StoreSpec.value())
            # ms.from_json(m, fname=f"json_initial_file/{operating_scenario_name}.json.gz")
        elif operating_scenario == OperatingScenario.multi_step:
            ms.from_json(m, fname="json_initial_file/multi_step.json.gz", wts=ms.StoreSpec.value())
        elif operating_scenario == OperatingScenario.power_mode:
            ms.from_json(m, fname="json_initial_file/power_mode.json.gz", wts=ms.StoreSpec.value())
        elif operating_scenario == OperatingScenario.neutral:
            ms.from_json(m, fname="json_initial_file/neutral.json.gz", wts=ms.StoreSpec.value())

        m.fs.feed_heater.electric_heat_duty.unfix()
        m.fs.sweep_heater.electric_heat_duty.unfix()

        # Copy initial conditions to rest of model for initialization
        m.fs.fix_initial_conditions()

        alias_dict = ComponentMap()
        alias_dict[m.fs.voltage_controller.mv_ref] = "potential"
        alias_dict[m.fs.voltage_controller.setpoint] = "soc_fuel_outlet_mole_frac_comp_H2"
        # alias_dict[m.fs.current_controller.mv_ref] = "current"
        # alias_dict[m.fs.current_controller.setpoint] = "soc_fuel_outlet_mole_frac_comp_H2"
        alias_dict[m.fs.soc_module.fuel_outlet_mole_frac_comp_H2] = "soc_fuel_outlet_mole_frac_comp_H2"
        # alias_dict[m.fs.makeup_mix.makeup.flow_mol] = "makeup_feed_rate"
        alias_dict[m.fs.h2_production_rate_controller.mv_ref] = "makeup_feed_rate"
        alias_dict[m.fs.h2_production_rate_controller.setpoint] = "h2_production_rate"
        alias_dict[m.fs.h2_production_rate_controller.gain_p] = "h2_production_rate_controller_gain_p"
        # alias_dict[m.fs.h2_production_rate_controller.gain_i] = "h2_production_rate_controller_gain_i"
        # alias_dict[m.fs.sweep_blower.inlet.flow_mol] = "sweep_feed_rate"
        alias_dict[m.fs.sweep_blower_controller.mv_ref] = "sweep_feed_rate"
        alias_dict[m.fs.sweep_blower_controller.setpoint] = "stack_core_temperature"
        alias_dict[m.fs.feed_heater_inner_controller.mv_ref] = "feed_heater_duty"
        alias_dict[m.fs.feed_heater_outer_controller.mv_ref] = "feed_heater_outlet_temperature"
        alias_dict[m.fs.feed_heater_outer_controller.setpoint] = "fuel_outlet_temperature"
        alias_dict[m.fs.sweep_heater_inner_controller.mv_ref] = "sweep_heater_duty"
        alias_dict[m.fs.sweep_heater_outer_controller.mv_ref] = "sweep_heater_outlet_temperature"
        alias_dict[m.fs.sweep_heater_outer_controller.setpoint] = "sweep_outlet_temperature"
        alias_dict[m.fs.makeup_mix.makeup_mole_frac_comp_H2] = "makeup_mole_frac_comp_H2"
        alias_dict[m.fs.makeup_mix.makeup_mole_frac_comp_H2O] = "makeup_mole_frac_comp_H2O"
        # alias_dict[m.fs.condenser_flash.heat_duty] = "condenser_heat_duty"
        alias_dict[m.fs.condenser_flash.vap_outlet.temperature] = "condenser_hot_outlet_temperature"

        alias_dict[m.fs.sweep_recycle_split.recycle_ratio] = "sweep_recycle_ratio"
        # alias_dict[m.fs.sweep_recycle_controller.mv_ref] = "sweep_recycle_ratio"
        # alias_dict[m.fs.sweep_recycle_controller.setpoint] = "sweep_outlet_temperature"

        alias_dict[m.fs.feed_recycle_split.recycle_ratio] = "fuel_recycle_ratio"
        alias_dict[m.fs.condenser_split.recycle_ratio] = "vgr_recycle_ratio"

        df = pd.read_csv("flowsheet_models/soec_flowsheet_operating_conditions.csv", index_col=0)
        t0 = m.fs.time.first()
        for var in ramp_mvs:
            shortname = var.name.split(".")[-1]
            alias = alias_dict[var]
            blk = var.parent_block()
            v_ramp = getattr(blk, shortname + "_ramp_rate")
            var[t0].fix(float(df[alias][ramp_setpoints[0]]))
            for i, t in enumerate(time_set):
                v_ramp[t].fix(float(
                    (df[alias][ramp_setpoints[i]] - df[alias][ramp_setpoints[i - 1]])
                    / (time_set[i] - time_set[i - 1])
                ))

        for var in step_mvs:
            shortname = var.name.split(".")[-1]
            alias = alias_dict[var]
            for i, t in enumerate(time_set):
                var[t].fix(float(df[alias][step_setpoints[i]]))

        # Need to initialize the setpoint for the inner controller or else it starts with the default value 0.5.
        m.fs.feed_heater_inner_controller.setpoint[0].value = m.fs.feed_heater_outer_controller.mv_ref[0].value
        m.fs.sweep_heater_inner_controller.setpoint[0].value = m.fs.sweep_heater_outer_controller.mv_ref[0].value
        # m.fs.feed_heater_inner_controller.negative_pv[0.0].value = - m.fs.soc_module.fuel_properties_in[0.0].temperature
        # m.fs.sweep_heater_inner_controller.negative_pv[0.0].value = - m.fs.soc_module.oxygen_properties_in[0.0].temperature

        for ctrl in m.fs.controller_set:
            if hasattr(ctrl, "mv_eqn"):
                calculate_variable_from_constraint(ctrl.manipulated_var[0], ctrl.mv_eqn[0])

        # add an extra expression to estimate efficiency
        m.fs.efficiency_hhv = pyo.Var(m.fs.time, initialize=1, units=pyo.units.dimensionless)


        @m.fs.Constraint(m.fs.time)
        def efficiency_hhv_eqn(b, t):
            return 141.80e6 * b.h2_mass_production[t] / b.total_electric_power[t] == b.efficiency_hhv[t]

        # define expression for estimating efficiency
        m.fs.integral_efficiency = pyo.Expression(expr=sum(m.fs.efficiency_hhv[t] for t in m.fs.time) / len(m.fs.time))

        # # check degrees of freedom
        # print('Degrees of freedom: ', dof(m))
        # assert dof(m) == 0

        ts_options = {
            "--ts_type": "beuler",
            "--ts_dt": 0.1,
            "--ts_rtol": 1e-3,
            # "--ts_adapt_clip":"0.001,300",
            # "--ksp_monitor":"",
            "--ts_adapt_dt_min": 1e-9,
            # "--ts_adapt_dt_max": 300,
            "--ksp_rtol": 1e-12,
            "--snes_type": "newtontr",
            # "--ts_max_reject": 200,
            # "--snes_monitor":"",
            "--ts_monitor": "",
            "--ts_save_trajectory": 1,
            "--ts_trajectory_type": "visualization",
            "--ts_max_snes_failures": 25,
            # "--show_cl":"",
            "-snes_max_it": 50,
            "-snes_rtol": 0,
            "-snes_stol": 0,
            "-snes_atol": 1e-6,
        }

        # initialize_model_with_propagation(m)
        folder = "petsc_cache"
        if not os.path.exists(folder + "/" + "petsc_results_nfe20.json.gz"):

            idaeslog.solver_log.tee = True
            results = petsc.petsc_dae_by_time_element(
                m,
                time=m.fs.time,
                # between=time_set[k:k+2],
                # timevar=m.fs.timevar,
                keepfiles=True,
                symbolic_solver_labels=True,
                ts_options=ts_options,
                skip_initial=False,
                initial_solver="ipopt",
                initial_solver_options={
                    # 'bound_push' : 1e-22,
                    'nlp_scaling_method': 'user-scaling',
                    'linear_solver': 'ma57',
                    'OF_ma57_automatic_scaling': 'yes',
                    'max_iter': 300,
                    'tol': 1e-6,
                    'halt_on_ampl_error': 'yes',
                    # 'mu_strategy': 'monotone',
                },
                # vars_stub="soec_flowsheet_prototype",
                # trajectory_save_prefix="soec_flowsheet_prototype",
                # previous_trajectory=traj
            )

            traj = results.trajectory
            # save Petsc results
            ms.to_json(m, fname=folder + "/" + "petsc_results_nfe20.json.gz")

            # save results
            if adding_new_time_elements:
                # number of elements = 20
                save_results(m, np.array(m.fs.time)[1:], results.trajectory, folder + "/" +
                             "PI_ramping_adding_new_time_elements")
            else:
                save_results(m, np.array(m.fs.time)[1:], results.trajectory, folder + "/" + "PI_ramping_base")

        else:
            # load Petsc results
            # folder = "health_submodels/optimization"
            print()
            print("Loading Petsc results from existing file")
            ms.from_json(m, fname=folder + "/" + "petsc_results_nfe20.json.gz")

        # #############################################################################
        # add stress models
        _make_thermal_stress(m, creep_model=True, stress_relaxation=True)

        # if adding_new_time_elements:
        # check degrees of freedom
        print('Degrees of freedom: ', dof(m))
        assert dof(m) == 0


        if not os.path.exists(folder + "/" + "Dynamics_Full_Discretization_BaseCase_nfe20.json.gz"):
            # solve full discretization with Ipopt
            results = solver.solve(m, tee=True)
            pyo.assert_optimal_termination(results)

            # save results
            save_results_full_discretization(m, np.array(m.fs.time)[1:],
                                             folder + "/" + "PI_ramping_full_discretization")
            _save_temperature_stress_flowsheet(m, folder + "/" + "Dynamics_Full_Discretization_BaseCase")

            # save full discretization results
            ms.to_json(m, fname=folder + "/" + "Dynamics_Full_Discretization_BaseCase_nfe20.json.gz")
            print('Saved full discretization results')
            print()
        else:
            # load full discretization results
            # folder = "health_submodels/optimization"
            print()
            print("Loading full discretization results from existing file")
            ms.from_json(m, fname=folder + "/" + "Dynamics_Full_Discretization_BaseCase_nfe20.json.gz")
        
        print('Cloning simulation moodel')
        m_sim = m.clone()
        print('Simulation model cloned')
        # ############################################################################
        # setup data from base case
        # write expression to sum the total of feed and sweep heater duties
        # define fraction of feed heater duty
        feed_heater_duty_tnaught = pyo.value(m.fs.feed_heater.electric_heat_duty[0]) * 1e-6
        sweep_heater_duty_tnaught = pyo.value(m.fs.sweep_heater.electric_heat_duty[0]) * 1e-6
        total_heater_duty_tnaught = feed_heater_duty_tnaught + sweep_heater_duty_tnaught

        feed_heater_inner_controller_setpoint_tnaught = pyo.value(m.fs.feed_heater_inner_controller.setpoint[0])
        sweep_heater_inner_controller_setpoint_tnaught = pyo.value(m.fs.sweep_heater_inner_controller.setpoint[0])
        total_heater_inner_controller_setpoint_tnaught = (feed_heater_inner_controller_setpoint_tnaught +
                                                          sweep_heater_inner_controller_setpoint_tnaught)

        feed_heater_inner_controller_manipulated_variable_tnaught = pyo.value(
            m.fs.feed_heater_inner_controller.manipulated_var[0]) * 1e-6
        sweep_heater_inner_controller_manipulated_variable_tnaught = pyo.value(
            m.fs.sweep_heater_inner_controller.manipulated_var[0]) * 1e-6
        total_heater_inner_controller_manipulated_variable_tnaught = (
                feed_heater_inner_controller_manipulated_variable_tnaught +
                sweep_heater_inner_controller_manipulated_variable_tnaught)

        feed_heater_outer_controller_setpoint_tnaught = pyo.value(m.fs.feed_heater_outer_controller.setpoint[0])
        sweep_heater_outer_controller_setpoint_tnaught = pyo.value(m.fs.sweep_heater_outer_controller.setpoint[0])
        total_heater_outer_controller_setpoint_tnaught = (feed_heater_outer_controller_setpoint_tnaught +
                                                          sweep_heater_outer_controller_setpoint_tnaught)

        # record the hydrogen mass production rate at the beginning of the optimization model
        desired_h2_production_rate_petsc = [pyo.value(m.fs.h2_mass_production[t]) for t in m.fs.time]

        # Define expression for total heater duty.
        m.fs.total_heater_duty = pyo.Expression(m.fs.time, rule=lambda b, t: \
            b.feed_heater.electric_heat_duty[t] + b.sweep_heater.electric_heat_duty[t])

        # Record the total heater duty at the beginning of the optimization model.
        feed_heater_duty_desired = [pyo.value(m.fs.feed_heater.electric_heat_duty[t]) for t in m.fs.time]
        sweep_heater_duty_desired = [pyo.value(m.fs.sweep_heater.electric_heat_duty[t]) for t in m.fs.time]
        ratio_heater_duty = [feed_heater_duty_desired[i] / sweep_heater_duty_desired[i] for i in range(len(m.fs.time))]

        total_heater_duty_desired = [pyo.value(m.fs.total_heater_duty[t]) for t in m.fs.time]
        ratio_feed_heater_duty = [feed_heater_duty_desired[i] / total_heater_duty_desired[i] \
                                  for i in range(len(m.fs.time))]

        setpoint_feed_heater_outer_controller_ref = [pyo.value(m.fs.feed_heater_outer_controller.setpoint[t]) \
                                                    for t in m.fs.time]
        setpoint_sweep_heater_outer_controller_ref = [pyo.value(m.fs.sweep_heater_outer_controller.setpoint[t]) \
                                                        for t in m.fs.time]

        setpoint_ramprate_sweep_heater_outer_controller_ref = [pyo.value(
            m.fs.sweep_heater_outer_controller.setpoint_ramp_rate[t]) for t in m.fs.time]
        setpoint_ramprate_feed_heater_outer_controller_ref = [pyo.value(
            m.fs.feed_heater_outer_controller.setpoint_ramp_rate[t]) for t in m.fs.time]

        fe_xfaces = m.fs.soc_module.solid_oxide_cell.fe_xfaces
        e_xfaces = m.fs.soc_module.solid_oxide_cell.e_xfaces
        oe_xfaces = m.fs.soc_module.solid_oxide_cell.oe_xfaces

        # record stress for electrolyte along time and iznodes elements from the base case
        # Assuming desired_electrolyte_stress is now a dictionary
        desired_fuel_electrode_first_layer_stress_simulation = {
            (t, iz): pyo.value(
                m.fs.soc_module.solid_oxide_cell.sigma_fe_creep[t, fe_xfaces.first(), iz])
            for t in m.fs.time
            for iz in m.fs.soc_module.solid_oxide_cell.iznodes}

        desired_fuel_electrode_last_layer_stress_simulation = {
            (t, iz): pyo.value(
                m.fs.soc_module.solid_oxide_cell.sigma_fe_creep[t, fe_xfaces.last(), iz])
            for t in m.fs.time
            for iz in m.fs.soc_module.solid_oxide_cell.iznodes}

        desired_electrolyte_stress_simulation = {
            (t, iz): pyo.value(m.fs.soc_module.solid_oxide_cell.sigma_e_creep[t, 0, iz])
            for t in m.fs.time
            for iz in m.fs.soc_module.solid_oxide_cell.iznodes}

        desired_oxygen_electrode_stress_simulation = {
            (t, iz): pyo.value(m.fs.soc_module.solid_oxide_cell.sigma_oe_creep[t, 0, iz])
            for t in m.fs.time
            for iz in m.fs.soc_module.solid_oxide_cell.iznodes}

        # find the max values in dictionary
        max_stress_electrolyte_simulation = max(desired_electrolyte_stress_simulation.values())
        print('Max electrolyte stress value from dynamic simulation:', max_stress_electrolyte_simulation)

        max_stress_oxygen_electrode_simulation = max(desired_oxygen_electrode_stress_simulation.values())
        print('Max oxygen electrode stress value from dynamic simulation:', max_stress_oxygen_electrode_simulation)

        max_stress_fuel_electrode_first_layer_simulation = max(desired_fuel_electrode_first_layer_stress_simulation.values())
        print('Max fuel electrode first layer stress value from dynamic simulation:', max_stress_fuel_electrode_first_layer_simulation)

        max_stress_fuel_electrode_last_layer_simulation = max(desired_fuel_electrode_last_layer_stress_simulation.values())
        print('Max fuel electrode last layer stress value from dynamic simulation:', max_stress_fuel_electrode_last_layer_simulation)

        # save the max values of stress from dynamic simulation to a csv file
        stress_dict = {
            'max_stress_electrolyte_simulation': max_stress_electrolyte_simulation,
            'max_stress_oxygen_electrode_simulation': max_stress_oxygen_electrode_simulation,
            'max_stress_fuel_electrode_first_layer_simulation': max_stress_fuel_electrode_first_layer_simulation,
            'max_stress_fuel_electrode_last_layer_simulation': max_stress_fuel_electrode_last_layer_simulation
        }
        stress_df = pd.DataFrame(stress_dict, index=[0])
        stress_df.to_csv(folder + "/" + "max_stress_values_simulation.csv")

        # ############################################################################
        # optimization model
        # add constraint to limit temperature in cells
        T_cell_up = 1073  # K, reference: Mina Naeni et al., 2022

        soec = m.fs.soc_module.solid_oxide_cell
        iznodes = soec.iznodes
        @m.fs.Constraint(m.fs.time, iznodes)
        def temperature_upper_bound_eqn(b, t, iz):
            return b.soc_module.solid_oxide_cell.temperature_z[t, iz] - T_cell_up <= 0

        # add constraint to set limit ratio values between feed and sweep heater duty
        @m.fs.Constraint(m.fs.time)
        def feed_heater_duty_constraint_1(b, t):
            return b.feed_heater.electric_heat_duty[t] - 0.05 * b.sweep_heater.electric_heat_duty[t] >= 0

        @m.fs.Constraint(m.fs.time)
        def feed_heater_duty_constraint_2(b, t):
            return b.feed_heater.electric_heat_duty[t] - 1.25 * b.sweep_heater.electric_heat_duty[t] <= 0


        scale_indexed_constraint(m.fs.temperature_upper_bound_eqn, 1e-2)
        # scale_indexed_constraint(m.fs.electrolyte_stress_constraint, 1e-2)
        scale_indexed_constraint(m.fs.feed_heater_duty_constraint_1, 1e-12)
        scale_indexed_constraint(m.fs.feed_heater_duty_constraint_2, 1e-12)


        # define a penalty function to penalize the difference between
        # the desired temperature and actual temperature
        def temperature_penalty_function(model, t, iz, max_temperature):
            """
                Load data from the desired temperature and calculate the penalty function.

                Args:
                    model (Block): The model object.
                    t (float): The time index.
                    iz (float): The index of the iznodes.
                    penalty_function (float): The penalty function for temperature.

                Returns:
                    penalty_function (float): The penalty function for temperature.
                """
            max_desired_temperature = max_temperature
            actual_temperature = model.fs.soc_module.solid_oxide_cell.temperature_z[t, iz]
            return (actual_temperature - max_desired_temperature) ** 2
        
        # Define a penalty function that minimizes the difference between 
        # the initial stress profile and the stress profile at the final stress profile


        @m.fs.Expression()
        def stress_terminal_penalty_function(blk):

            layer_weights = {'fe':1e-4, 'oe':1e-4, 'e':1e-4}

            FE_stress = blk.soc_module.solid_oxide_cell.sigma_fe_creep
            OE_stress = blk.soc_module.solid_oxide_cell.sigma_oe_creep
            E_stress = blk.soc_module.solid_oxide_cell.sigma_e_creep

            t_initial = 0
            t_final = blk.time.last()

            FE_stress_difference = sum((FE_stress[t_final, ix, iz] - FE_stress[t_initial, ix, iz]) ** 2
                                        for ix in blk.soc_module.solid_oxide_cell.fe_xfaces
                                        for iz in blk.soc_module.solid_oxide_cell.iznodes)
            
            OE_stress_difference = sum((OE_stress[t_final, ix, iz] - OE_stress[t_initial, ix, iz]) ** 2
                                        for ix in blk.soc_module.solid_oxide_cell.oe_xfaces
                                        for iz in blk.soc_module.solid_oxide_cell.iznodes)
            
            E_stress_difference = sum((E_stress[t_final, ix, iz] - E_stress[t_initial, ix, iz]) ** 2
                                        for ix in blk.soc_module.solid_oxide_cell.e_xfaces
                                        for iz in blk.soc_module.solid_oxide_cell.iznodes)
            
            return (layer_weights['fe'] * FE_stress_difference + 
                    layer_weights['oe'] * OE_stress_difference + 
                    layer_weights['e'] * E_stress_difference)
                                       



        # ############################################################################
        power_conversion_factor = 1e-9
        omega_temperature = 0  # if we increase from 1e-10 to 1e-4, temperature dominates objective function.
        objective_expr = sum(power_conversion_factor * m.fs.total_electric_power[t] for t in m.fs.time)
        print('Objective function value of steady state case:', pyo.value(objective_expr))
        obj_val_base_case = pyo.value(objective_expr)

        # write objective function to minimize the total of electric power consumption for time horizon
        print('')
        print('solving optimization model without stress constraint')
        if omega_temperature != 0:
            m.fs.objective = pyo.Objective(expr=objective_expr + sum(
                omega_temperature * temperature_penalty_function(
                    m, t, iz, max_temperature=T_cell_up) for t in m.fs.time
                for iz in iznodes) / len(m.fs.time), sense=pyo.minimize)
        else:
            m.fs.objective = pyo.Objective(expr=objective_expr, sense=pyo.minimize)
        
        # Set up ramp rate for heed and sweep heater duties

        @m.fs.feed_heater.Expression(m.fs.time)
        def ramp_rate(b, t):
            if t == 0:
                return 0
            else:
                return (b.electric_heat_duty[t] - b.electric_heat_duty[m.fs.time.prev(t)]) / (t - m.fs.time.prev(t))
            
        @m.fs.sweep_heater.Expression(m.fs.time)
        def ramp_rate(b, t):
            if t == 0:
                return 0
            else:
                return (b.electric_heat_duty[t] - b.electric_heat_duty[m.fs.time.prev(t)]) / (t - m.fs.time.prev(t))
            
        
        if keep_controllers:
            # max_feed_heater_ramprate = np.abs(np.array(m.fs.feed_heater.ramp_rate[:]())).max()
            # max_sweep_heater_ramprate = np.abs(np.array(m.fs.sweep_heater.ramp_rate[:]())).max()
            # max_H2_ramprate = np.abs(np.array(m.fs.ramp_rate_h2_production[:]())).max()


            # alpha_feed = max_feed_heater_ramprate / max_H2_ramprate / 2 * 1e-3
            # alpha_sweep = max_sweep_heater_ramprate / max_H2_ramprate / 2 * 1e-3

            # beta_feed = alpha_feed 
            # beta_sweep = alpha_sweep 

            # Set up ramp rate constraint
            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_feed_heater_ub_constraint(b, t):
            #     alpha_feed = 5e4
            #     beta_feed = 1e3
            #     if b.ramp_rate_h2_production[t]() >= 0:
            #         return b.feed_heater.ramp_rate[t]  <= (alpha_feed * b.ramp_rate_h2_production[t] + beta_feed)
            #     else:
            #         return b.feed_heater.ramp_rate[t]  >= (alpha_feed * b.ramp_rate_h2_production[t] - beta_feed)

            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_sweep_heater_ub_constraint(b, t):
            #     alpha_sweep = 5e4
            #     beta_sweep = 5e4
            #     if b.ramp_rate_h2_production[t]() >= 0:
            #         return b.sweep_heater.ramp_rate[t]  <= (alpha_sweep * b.ramp_rate_h2_production[t] + beta_sweep)
            #     else:
            #         return b.sweep_heater.ramp_rate[t]  >= (alpha_sweep * b.ramp_rate_h2_production[t] - beta_sweep)
        
            
            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_feed_heater_ub_constraint(b, t):
            #     return b.feed_heater.ramp_rate[t]  <= 0.5 * np.array(m.fs.feed_heater.ramp_rate[:]()).max()
            
            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_feed_heater_lb_constraint(b, t):
            #     return b.feed_heater.ramp_rate[t]  >= 0.5 * np.array(m.fs.feed_heater.ramp_rate[:]()).min()
            

            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_sweep_heater_ub_constraint(b, t):
            #     return b.sweep_heater.ramp_rate[t]  <= 0.5 * np.array(m.fs.sweep_heater.ramp_rate[:]()).max()
            
            # @m.fs.Constraint(m.fs.time)
            # def ramp_rate_sweep_heater_lb_constraint(b, t):
            #     return b.sweep_heater.ramp_rate[t]  >= 0.5 * np.array(m.fs.sweep_heater.ramp_rate[:]()).min()

            ramp_time_points = [3750.0, 22050.0]
            for t in ramp_time_points:
                m.fs.feed_heater_outer_controller.setpoint_ramp_rate[t].unfix()
            # m.fs.feed_heater_outer_controller.setpoint[0].unfix()

            m.fs.feed_heater_outer_controller.setpoint_ramp_rate[3750.0].setlb(-2)
            m.fs.feed_heater_outer_controller.setpoint_ramp_rate[3750.0].setub(0)

            m.fs.feed_heater_outer_controller.setpoint_ramp_rate[22050.0].setlb(0)
            m.fs.feed_heater_outer_controller.setpoint_ramp_rate[22050.0].setub(2)

            m.fs.sweep_heater_outer_controller.setpoint_ramp_rate[3750.0].setlb(-2)
            m.fs.sweep_heater_outer_controller.setpoint_ramp_rate[3750.0].setub(0)

            m.fs.sweep_heater_outer_controller.setpoint_ramp_rate[22050.0].setlb(0)
            m.fs.sweep_heater_outer_controller.setpoint_ramp_rate[22050.0].setub(2)

            # set_indexed_variable_bounds(m.fs.feed_heater_outer_controller.setpoint_ramp_rate, (-1, 0.3))
            set_indexed_variable_bounds(m.fs.feed_heater_outer_controller.setpoint, (910, 1000))
            
            # set_indexed_variable_bounds(m.fs.sweep_heater_outer_controller.setpoint_ramp_rate, (-0.3, 0.3))
            set_indexed_variable_bounds(m.fs.sweep_heater_outer_controller.setpoint, (910, 1000))

            m.fs.objective = pyo.Objective(expr=objective_expr 
                                        , sense=pyo.minimize)
            

        else:
            # optimize case 2

            @m.fs.Expression(m.fs.time)
            def feed_heater_ramp_rate_penalty(b, t):
                return 1e-6*(b.feed_heater.ramp_rate[t] - b.feed_heater.ramp_rate[t]())**2 
            
            @m.fs.Expression(m.fs.time)
            def sweep_heater_ramp_rate_penalty(b, t):
                return 1e-6*(b.sweep_heater.ramp_rate[t] - b.sweep_heater.ramp_rate[t]())**2

            # unfix controllers of heater for inner and outer for both feed and sweep sides
            m.fs.feed_heater_inner_controller.deactivate()
            m.fs.feed_heater_outer_controller.deactivate()
            m.fs.sweep_heater_inner_controller.deactivate()
            m.fs.sweep_heater_outer_controller.deactivate()

            # fix heater duty for feed and sweep sides at t = 0
            m.fs.feed_heater.electric_heat_duty[0].fix(pyo.value(m.fs.feed_heater.electric_heat_duty[0]))
            m.fs.sweep_heater.electric_heat_duty[0].fix(pyo.value(m.fs.sweep_heater.electric_heat_duty[0]))

            # set lb and up for heater duty
            m.fs.feed_heater.electric_heat_duty.setlb(1e3)
            m.fs.feed_heater.electric_heat_duty.setub(20e6)
            m.fs.sweep_heater.electric_heat_duty.setlb(1e3)
            m.fs.sweep_heater.electric_heat_duty.setub(20e6)

            objective_expr = sum(power_conversion_factor * m.fs.total_electric_power[t] for t in m.fs.time)
            m.fs.objective.deactivate()
            alpha = 0.05
            time_points = set(m.fs.time) - set( [3750.0, 22050.0])
            m.fs.objective = pyo.Objective(expr=objective_expr 
                                        + alpha * sum(m.fs.feed_heater_ramp_rate_penalty[t] for t in time_points) 
                                        + alpha * sum(m.fs.sweep_heater_ramp_rate_penalty[t] for t in time_points)
                                        + m.fs.stress_terminal_penalty_function, sense=pyo.minimize)
            


        results_folder = 'reworked_results/results'
        
        if not os.path.exists(f'{results_folder}/base_no_stress_constraint'):
            os.makedirs(f'{results_folder}/base_no_stress_constraint')


        file = f"{results_folder}/base_no_stress_constraint/model_base.json.gz"
        if os.path.exists(file):
            print('loading base dynamic optimization from json')
            ms.from_json(m, fname=file)
        else:
            # solve model with Ipopt for base case
            print("Running dynamic optimization without considering stress constraint")
            results_base = solver.solve(m, tee=True)
            pyo.assert_optimal_termination(results_base)
    
            obj_val_base = pyo.value(objective_expr)

            ms.to_json(m, fname=f'{results_folder}/base_no_stress_constraint/model_base.json.gz')

        m_no_stress = m.clone()
        # save the max values of stress from dynamic simulation to a csv file
        dyn_opt_stress_dict = {
            'max_stress_electrolyte_simulation': max(desired_electrolyte_stress_simulation.values()),
            'max_stress_oxygen_electrode_simulation': max(desired_oxygen_electrode_stress_simulation.values()),
            'max_stress_fuel_electrode_first_layer_simulation': max(desired_fuel_electrode_first_layer_stress_simulation.values()),
            'max_stress_fuel_electrode_last_layer_simulation': max(desired_fuel_electrode_last_layer_stress_simulation.values()),
        }
        dyn_opt_stress_df = pd.DataFrame(dyn_opt_stress_dict, index=[0])
        dyn_opt_stress_df.to_csv(folder + "/" + "max_stress_values_dyn_opt.csv")

        ## Step-1: Find maximum stress value and layer


        stress_values = np.array([max((np.array(m.fs.soc_module.solid_oxide_cell.sigma_fe_creep[:, :,:]()))),
                       max((np.array(m.fs.soc_module.solid_oxide_cell.sigma_e_creep[:, :,:]()))),
                       max((np.array(m.fs.soc_module.solid_oxide_cell.sigma_oe_creep[:, :,:]()))),
                       ])

        ## Step-2: Define three stress constraint cases (Inactive, 0.99, max_stress, 0.95)

        # stress_reduction_factors = [1.2, 0.95, 0.9] #[1.2, 0.99, 0.95, 0.9]

        ## Step-3: Run optimization model with stress constraints

        def solve_stress_constraint(stress_reduction_factor=None, stress_penalty=None, init_from_file=False):
            if init_from_file:
                file = f"figures/new_optimization_plots/opt_stress_constraint-{stress_reduction_factor}/model_opt.json.gz"
                if os.path.exists(file):
                    ms.from_json(m, fname=file)
                else:
                    # file = f"figures/new_optimization_plots/opt_stress_constraint-{stress_reduction_factor}/model_opt.json.gz"
                    raise Exception('File does not exist')
            else:
                file = f"figures/new_optimization_plots/opt_stress_constraint-{stress_reduction_factor}/model_opt.json.gz"
            
            if stress_penalty is None:
                max_stress = np.max(stress_values)
                assert max_stress > 0, "Stress value is negative, put the abs back in"
                if np.argmax(stress_values) == 0:
                    stress_var = m.fs.soc_module.solid_oxide_cell.sigma_fe_creep
                    ixset = m.fs.soc_module.solid_oxide_cell.fe_xfaces
                elif np.argmax(stress_values) == 1:
                    stress_var = m.fs.soc_module.solid_oxide_cell.sigma_e_creep
                    ixset = m.fs.soc_module.solid_oxide_cell.e_xfaces
                else:
                    stress_var = m.fs.soc_module.solid_oxide_cell.sigma_oe_creep
                    ixset = m.fs.soc_module.solid_oxide_cell.oe_xfaces

                m.fs.stress_constraint = pyo.Constraint(
                    m.fs.time, ixset, iznodes, rule=lambda b, t, ix, iz: stress_var[t, ix, iz]**2 <= (stress_reduction_factor * max_stress)**2 )

                stress_penalty_term = 0

            elif stress_reduction_factor is None:
            
                m.fs.stress_penalty_weight = pyo.Var(initialize=stress_penalty)
                m.fs.stress_penalty_weight.fix()

                # Fuel electrode stress deviation from initial point
                @m.fs.Expression()
                def fe_stress_penalty_term(b):
                    ixset = b.soc_module.solid_oxide_cell.fe_xfaces
                    return sum((b.soc_module.solid_oxide_cell.sigma_fe_creep[t, ix, iz] - b.soc_module.solid_oxide_cell.sigma_fe_creep[0, ix, iz])**2 for
                     t in b.time for ix in ixset for iz in iznodes)


                # Oxygen electrode stress deviation from initial point
                @m.fs.Expression()
                def oe_stress_penalty_term(b):
                    ixset = b.soc_module.solid_oxide_cell.oe_xfaces
                    return sum((b.soc_module.solid_oxide_cell.sigma_oe_creep[t, ix, iz] - b.soc_module.solid_oxide_cell.sigma_oe_creep[0, ix, iz])**2 for
                     t in m.fs.time for ix in ixset for iz in iznodes)
                    
                # Electrolyte stress deviation from initial point

                @m.fs.Expression()
                def e_stress_penalty_term(b):
                    ixset = b.soc_module.solid_oxide_cell.e_xfaces
                    return sum((b.soc_module.solid_oxide_cell.sigma_e_creep[t, ix, iz] - b.soc_module.solid_oxide_cell.sigma_e_creep[0, ix, iz])**2 for
                     t in b.time for ix in ixset for iz in iznodes)

                
                stress_penalty_term =  m.fs.stress_penalty_weight * 1e-6 * (m.fs.fe_stress_penalty_term + m.fs.oe_stress_penalty_term + m.fs.e_stress_penalty_term)
            m.fs.objective.deactivate()
            alpha = 0.05
            m.fs.objective = pyo.Objective(expr=objective_expr
                                        + alpha * sum(m.fs.feed_heater_ramp_rate_penalty[t] for t in time_points) 
                                        + alpha * sum(m.fs.sweep_heater_ramp_rate_penalty[t] for t in time_points)
                                        +   stress_penalty_term
                                            , sense=pyo.minimize)
            folder = f"reworked_results/results/opt_stress_constraint-{stress_reduction_factor}-{stress_penalty}"
            file = f"{folder}/model_opt.json.gz"

            if not os.path.exists(folder):
                os.makedirs(folder)
                print('Created a new folder for results')

            if not os.path.exists(file):
                results = solver.solve(m, tee=True)
                pyo.assert_optimal_termination(results)

                ms.to_json(m, fname=file)

            # # Delete constraint
            # m.fs.del_component("stress_constraint")
            # m.fs.del_component("stress_constraint_index")

            return m



# %%
weights = [0.01, 0.05, 0.1, 0.2, 0.5]

for weight in weights:
    m = solve_stress_constraint(stress_reduction_factor=None, stress_penalty=weight, init_from_file=False)

# %% 




case_name = 'base_no_stress_constraint'
# case_name = 'opt_stress_constraint-0.75'
figure_save_location = f'figures/new_optimization_plots/{case_name}'

if not os.path.exists(figure_save_location):
    os.makedirs(figure_save_location)
    ms.to_json(m, fname=f'{figure_save_location}/model_opt.json.gz')
    ms.to_json(m_sim, fname=f'{figure_save_location}/model_sim.json.gz')
else:
    raise Exception('Folder already exists')




# %%


plt.plot(np.array(m.fs.time)/3600, m.fs.soc_module.solid_oxide_cell.potential[:](), color = "blue",label = 'optimization' )
plt.plot(np.array(m.fs.time)/3600, m_sim.fs.soc_module.solid_oxide_cell.potential[:](), color = "black",linestyle = '--', label = 'simulation' )
plt.xlabel('Time (h)')
plt.ylabel('Voltage (V)')
plt.legend(loc = 'best')
# increase number of xticks
plt.xticks(np.arange(0, 10, 1))
# plt.title(f'Weight: {alpha}')
# plt.savefig(f'{figure_save_location}/voltage.png')
plt.show()

# Plotting fs.total_electric_power

plt.plot(np.array(m.fs.time)/3600, np.array(m.fs.total_electric_power[:]()) * 1e-6, 
        color = "blue"
        ,label = f'optimization {np.array(m.fs.total_electric_power[:]()).sum() * 1e-6:.2f} MW' )
plt.plot(np.array(m.fs.time)/3600, np.array(m_sim.fs.total_electric_power[:]()) * 1e-6, 
        color = "black",linestyle = '--', 
        label = f'simulation {np.array(m_sim.fs.total_electric_power[:]()).sum() * 1e-6:.2f} MW' )
plt.xlabel('Time (h)')
plt.ylabel('Power (MW)')
plt.legend(loc = 'best')
plt.xticks(np.arange(0, 10, 1))
# plt.title(f'Weight: {alpha}')
# plt.savefig(f'{figure_save_location}/power.png')
plt.show()

# Plotting mean(fs.soc_module.solid_oxide_cell.temperature_z)

plt.plot(np.array(m.fs.time)/3600, [np.mean(np.array(m.fs.soc_module.solid_oxide_cell.temperature_z[t,:]())) for t in m.fs.time],
            color = "blue", label = f'optimization')
plt.plot(np.array(m.fs.time)/3600, [np.mean(np.array(m_sim.fs.soc_module.solid_oxide_cell.temperature_z[t,:]())) for t in m.fs.time],
            color = "black", linestyle = '--', label = f'simulation')
# plt.axhline(y = 1073, color = 'r', linestyle = '--', label = 'Temperature limit')
plt.xlabel('Time (h)')
plt.ylabel('Average Cell Temperature (K)')
# Move legend down a bit from the best loc
plt.legend(loc = 'best')
plt.xticks(np.arange(0, 10, 1))
# plt.title(f'Weight: {alpha}')
# plt.savefig(f'{figure_save_location}/temperature.png')
plt.show()


# Plot Heater duties

plt.plot(np.array(m.fs.time)/3600, np.array(m.fs.feed_heater.electric_heat_duty[:]()) * 1e-6,
            'b-', label = 'Feed heater duty opt')
plt.plot(np.array(m.fs.time)/3600, np.array(m.fs.sweep_heater.electric_heat_duty[:]()) * 1e-6,
            'r-', label = 'Sweep heater duty opt')
plt.plot(np.array(m.fs.time)/3600, np.array(m_sim.fs.feed_heater.electric_heat_duty[:]()) * 1e-6,
            'b-.', label = 'Feed heater duty base')
plt.plot(np.array(m.fs.time)/3600, np.array(m_sim.fs.sweep_heater.electric_heat_duty[:]()) * 1e-6,
            'r-.', label = 'Sweep heater duty base')
plt.xlabel('Time (h)')
plt.ylabel('Heater Duty (MW)')
plt.legend(loc = 'best') 
plt.xticks(np.arange(0, 10, 1))
# plt.title(f'Weight: {alpha}')
# plt.savefig(f'{figure_save_location}/heater_duty.png')
plt.show()



# Plot fuel electrode stress profiles
stresses_opt = np.array(m.fs.soc_module.solid_oxide_cell.sigma_fe_creep[:, :, :]())

stresses_opt = stresses_opt.reshape(len(m.fs.time.ordered_data()),
                    len(m.fs.soc_module.solid_oxide_cell.fe_xfaces), 
                     len(m.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_opt = np.unravel_index(np.abs(stresses_opt).argmax(), stresses_opt.shape)

stresses_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.sigma_fe_creep[:, :, :]())

stresses_sim = stresses_sim.reshape(len(m_sim.fs.time.ordered_data()),
                    len(m_sim.fs.soc_module.solid_oxide_cell.fe_xfaces), 
                     len(m_sim.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_sim = np.unravel_index(np.abs(stresses_sim).argmax(), stresses_sim.shape)


plt.plot(m.fs.soc_module.solid_oxide_cell.znodes, stresses_opt[max_stress_index_opt[0], max_stress_index_opt[1], :],
            color = "blue", label = f'optimization - location_ix = {max_stress_index_opt[1]} - time = {max_stress_index_opt[0]}')
plt.plot(m_sim.fs.soc_module.solid_oxide_cell.znodes, stresses_sim[max_stress_index_sim[0], max_stress_index_sim[1], :],
            color = "blue", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - time = {max_stress_index_sim[0]}')
plt.xlabel('Znodes')
plt.ylabel('Fuel Electrode Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/fuel_electrode_stress.png')
plt.show()


# Plot temporal stress profile at max stress location
plt.plot(np.array(m.fs.time)/3600, stresses_opt[:, max_stress_index_opt[1], max_stress_index_opt[2]],
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, stresses_sim[:, max_stress_index_sim[1], max_stress_index_sim[2]],
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Fuel Electrode Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/fuel_electrode_stress_temporal.png')
plt.show()


# Plot temporal temperature profile at max stress location
plt.plot(np.array(m.fs.time)/3600, m.fs.soc_module.solid_oxide_cell.fuel_electrode.temperature[:, 1, max_stress_index_opt[2]+1](),
            color = "blue", label = f'optimization - xnode = {max_stress_index_opt[1]}, znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, m_sim.fs.soc_module.solid_oxide_cell.fuel_electrode.temperature[:, 1, max_stress_index_sim[2]+1](),
            color = "blue", linestyle = '--', label = f'simulation - xnode = {max_stress_index_sim[1]}, znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Local Fuel Electrode Temperature (K)')
plt.legend(loc = 'best')
plt.show()


# Plot temporal creep_strain profile at max stress location for Fuel Electrode
creep_strain_opt = np.array(m.fs.soc_module.solid_oxide_cell.creep_strain_fe[:, max_stress_index_opt[1], max_stress_index_opt[2]+1]())
creep_strain_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.creep_strain_fe[:, max_stress_index_sim[1], max_stress_index_sim[2]+1]())

plt.plot(np.array(m.fs.time)/3600, creep_strain_opt,
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, creep_strain_sim,
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Creep Strain FE')
plt.legend(loc = 'best')
plt.show()
# %%
# Plot electrolyte stress profiles
stresses_opt = np.array(m.fs.soc_module.solid_oxide_cell.sigma_e_creep[:, :, :]())

stresses_opt = stresses_opt.reshape(len(m.fs.time.ordered_data()),
                    len(m.fs.soc_module.solid_oxide_cell.e_xfaces), 
                     len(m.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_opt = np.unravel_index(np.abs(stresses_opt).argmax(), stresses_opt.shape)

stresses_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.sigma_e_creep[:, :, :]())

stresses_sim = stresses_sim.reshape(len(m_sim.fs.time.ordered_data()),
                    len(m_sim.fs.soc_module.solid_oxide_cell.e_xfaces), 
                     len(m_sim.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_sim = np.unravel_index(np.abs(stresses_sim).argmax(), stresses_sim.shape)




plt.plot(m.fs.soc_module.solid_oxide_cell.znodes, stresses_opt[max_stress_index_opt[0], max_stress_index_opt[1], :],
            color = "blue", label = f'optimization - location_ix = {max_stress_index_opt[1]} - time = {max_stress_index_opt[0]}')
plt.plot(m_sim.fs.soc_module.solid_oxide_cell.znodes, stresses_sim[max_stress_index_sim[0], max_stress_index_sim[1], :],
            color = "blue", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - time = {max_stress_index_sim[0]}')
plt.xlabel('Znodes')
plt.ylabel('Electrolyte Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/electrolyte_stress.png')
plt.show()

# Plot temporal stress profile at max stress location
plt.plot(np.array(m.fs.time)/3600, stresses_opt[:, max_stress_index_opt[1], max_stress_index_opt[2]],
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, stresses_sim[:, max_stress_index_sim[1], max_stress_index_sim[2]],
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Electrolyte Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/electrolyte_stress_temporal.png')
plt.show()


# Plot temporal temperature profile at max stress location
plt.plot(np.array(m.fs.time)/3600, m.fs.soc_module.solid_oxide_cell.electrolyte.temperature[:, max_stress_index_opt[2]+1](),
            color = "blue", label = f'optimization - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, m_sim.fs.soc_module.solid_oxide_cell.electrolyte.temperature[:, max_stress_index_sim[2]+1](),
            color = "blue", linestyle = '--', label = f'simulation - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Local Electrolyte Temperature (K)')
plt.legend(loc = 'best')
plt.show()


# Plot temporal creep_strain profile at max stress location for Electrolyte
creep_strain_opt = np.array(m.fs.soc_module.solid_oxide_cell.creep_strain_e[:, max_stress_index_opt[1], max_stress_index_opt[2]+1]())
creep_strain_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.creep_strain_e[:, max_stress_index_sim[1], max_stress_index_sim[2]+1]())

plt.plot(np.array(m.fs.time)/3600, creep_strain_opt,
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]+1}')
plt.plot(np.array(m_sim.fs.time)/3600, creep_strain_sim,
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]+1}')
plt.xlabel('Time (h)')
plt.ylabel('Creep Strain E')
plt.legend(loc = 'best')
plt.show()

# %%
# Plot oxygen electrode stress profiles
stresses_opt = np.array(m.fs.soc_module.solid_oxide_cell.sigma_oe_creep[:, :, :]())

stresses_opt = stresses_opt.reshape(len(m.fs.time.ordered_data()),
                    len(m.fs.soc_module.solid_oxide_cell.oe_xfaces), 
                     len(m.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_opt = np.unravel_index(np.abs(stresses_opt).argmax(), stresses_opt.shape)

stresses_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.sigma_oe_creep[:, :, :]())

stresses_sim = stresses_sim.reshape(len(m_sim.fs.time.ordered_data()),
                    len(m_sim.fs.soc_module.solid_oxide_cell.oe_xfaces), 
                     len(m_sim.fs.soc_module.solid_oxide_cell.znodes))

max_stress_index_sim = np.unravel_index(np.abs(stresses_sim).argmax(), stresses_sim.shape)


plt.plot(m.fs.soc_module.solid_oxide_cell.znodes, stresses_opt[max_stress_index_opt[0], max_stress_index_opt[1], :],
            color = "blue", label = f'optimization - location_ix = {max_stress_index_opt[1]} - time = {max_stress_index_opt[0]}')
plt.plot(m_sim.fs.soc_module.solid_oxide_cell.znodes, stresses_sim[max_stress_index_sim[0], max_stress_index_sim[1], :],
            color = "blue", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - time = {max_stress_index_sim[0]}')
plt.xlabel('Znodes')
plt.ylabel('Oxygen Electrode Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/oxygen_electrode_stress.png')
plt.show()

# Plot temporal stress profile at max stress location
plt.plot(np.array(m.fs.time)/3600, stresses_opt[:, max_stress_index_opt[1], max_stress_index_opt[2]],
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, stresses_sim[:, max_stress_index_sim[1], max_stress_index_sim[2]],
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Oxygen Electrode Stress (MPa)')
plt.legend(loc = 'best')
# plt.title(f'Stress reduction factor = 0.95')
# plt.savefig(f'{figure_save_location}/oxygen_electrode_stress_temporal.png')
plt.show()


# Plot temporal temperature profile at max stress location
plt.plot(np.array(m.fs.time)/3600, m.fs.soc_module.solid_oxide_cell.oxygen_electrode.temperature[:, max_stress_index_opt[1]](),
            color = "blue", label = f'optimization - znode = {max_stress_index_opt[2]}')
plt.plot(np.array(m_sim.fs.time)/3600, m_sim.fs.soc_module.solid_oxide_cell.oxygen_electrode.temperature[:, max_stress_index_sim[2]](),
            color = "blue", linestyle = '--', label = f'simulation - znode = {max_stress_index_sim[2]}')
plt.xlabel('Time (h)')
plt.ylabel('Local Oxygen Electrode Temperature (K)')
plt.legend(loc = 'best')
plt.show()

# Plot temporal creep_strain profile at max stress location for Oxygen Electrode
creep_strain_opt = np.array(m.fs.soc_module.solid_oxide_cell.creep_strain_oe[:, max_stress_index_opt[1], max_stress_index_opt[2]+1]())
creep_strain_sim = np.array(m_sim.fs.soc_module.solid_oxide_cell.creep_strain_oe[:, max_stress_index_sim[1], max_stress_index_sim[2]+1]())

plt.plot(np.array(m.fs.time)/3600, creep_strain_opt,
            color = "red", label = f'optimization - location_ix = {max_stress_index_opt[1]} - znode = {max_stress_index_opt[2]+1}')
plt.plot(np.array(m_sim.fs.time)/3600, creep_strain_sim,
            color = "red", linestyle = '--', label = f'simulation - location_ix = {max_stress_index_sim[1]} - znode = {max_stress_index_sim[2]+1}')
plt.xlabel('Time (h)')
plt.ylabel('Creep Strain OE')
plt.legend(loc = 'best')
plt.show()




    

# %%

# alpha = np.linspace(0.3, 1, len(iznodes)+1)

# for i, iz in enumerate(list(iznodes)):
#     plt.plot(np.array(m.fs.time)/3600, m.fs.soc_module.solid_oxide_cell.temperature_z[:, iz](), label = f'znode = {iz}')
# plt.xlabel('Time (h)')
# plt.ylabel('Temperature (K)')
# plt.legend(loc = 'best')
# plt.show()


# for iz in iznodes:
#     plt.plot(np.array(m.fs.time)/3600, m_sim.fs.soc_module.solid_oxide_cell.temperature_z[:, iz](), label = f'znode = {iz}')
# plt.xlabel('Time (h)')
# plt.ylabel('Temperature (K)')
# plt.legend(loc = 'best')
# plt.show()

# %%
