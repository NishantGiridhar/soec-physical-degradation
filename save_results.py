import numpy as np
from scipy.io import savemat
import pyomo.environ as pyo

def save_results(m, ramp_list, traj, filename):
    """Extracts trajectories of variables of interest from a PETSc trajectory object and saves them in a .mat file.

    Args:
        m: ConcreteModel object (to get variable names from)
        ramp_list: A list of times, in seconds, when ramping starts and stops
        traj: PETSc trajectory object
        filename: Name .mat file to write
        PI_control: Whether to include reference trajectories from PI controllers

    Returns:
        None
    """

    time_set = m.fs.time.ordered_data()
    tf = time_set[-1]
    soec = m.fs.soc_module.solid_oxide_cell

    out_dict = {
        "ramp_list": np.array(ramp_list),
        "time": np.array(traj.time),
        "potential": np.array(traj.vecs[str(soec.potential[tf])]),
        "current": np.array(traj.vecs[str(m.fs.soc_module.total_current[tf])]),
        "soec_fuel_inlet_flow": np.array(traj.vecs[str(m.fs.soc_module.fuel_inlet.flow_mol[tf])]),
        "soec_oxygen_inlet_flow": np.array(traj.vecs[str(m.fs.soc_module.oxygen_inlet.flow_mol[tf])]),
        "fuel_heater_duty": np.array(traj.vecs[str(m.fs.feed_heater.electric_heat_duty[tf])]),
        "sweep_heater_duty": np.array(traj.vecs[str(m.fs.sweep_heater.electric_heat_duty[tf])]),
        "fuel_inlet_H2": np.array(traj.vecs[str(soec.fuel_inlet.mole_frac_comp[tf, "H2"])]),
        "fuel_inlet_H2O": np.array(traj.vecs[str(soec.fuel_inlet.mole_frac_comp[tf, "H2O"])]),
        "fuel_outlet_H2O": np.array(traj.vecs[str(soec.fuel_channel.mole_frac_comp[tf, soec.iznodes.last(), "H2O",])]),
        "sweep_inlet_O2": np.array(traj.vecs[str(soec.oxygen_inlet.mole_frac_comp[tf, "O2"])]),
        "sweep_outlet_O2": np.array(traj.vecs[str(soec.oxygen_channel.mole_frac_comp[tf, soec.iznodes.first(), "O2"])]),
        "H2_production": np.array(traj.vecs[str(m.fs.h2_mass_production[tf])]),
        "fuel_outlet_mole_frac_comp_H2": np.array(traj.vecs[str(m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[tf])]),
        # "steam_feed_rate": np.array(traj.vecs[str(m.fs.feed_medium_exchanger.cold_side_inlet.flow_mol[tf])]),
        "steam_feed_rate": np.array(traj.vecs[str(m.fs.makeup_mix.makeup.flow_mol[tf])]),
        "sweep_feed_rate": np.array(traj.vecs[str(m.fs.sweep_blower.inlet.flow_mol[tf])]),
        "total_electric_power": np.array(traj.vecs[str(m.fs.total_electric_power[tf])]),
        # "efficiency_hhv": np.array(traj.vecs[str(m.fs.efficiency_hhv[tf])]),
        # "efficiency_lhv": np.array(traj.vecs[str(m.fs.efficiency_lhv[tf])]),
        "fuel_inlet_temperature": np.array(traj.vecs[str(soec.fuel_channel.temperature_inlet[tf])]) ,
        "sweep_inlet_temperature": np.array(traj.vecs[str(soec.oxygen_channel.temperature_inlet[tf])]) ,
        "stack_core_temperature": np.array(traj.vecs[str(m.fs.stack_core_temperature[tf])]) ,
        "fuel_outlet_temperature": np.array(traj.vecs[str(soec.fuel_channel.temperature_outlet[tf])]) ,
        "sweep_outlet_temperature": np.array(traj.vecs[str(soec.oxygen_channel.temperature_outlet[tf])]) ,
        "fuel_heater_outer_setpoint":np.array(traj.vecs[str(m.fs.feed_heater_outer_controller.setpoint[tf])]),
        "sweep_heater_outer_setpoint":np.array(traj.vecs[str(m.fs.sweep_heater_outer_controller.setpoint[tf])]),
        "product_mole_frac_H2": np.array(
            traj.vecs[str(m.fs.condenser_split.inlet.mole_frac_comp[tf, "H2"])]
        ),
        "condenser_outlet_temperature": np.array(
            traj.vecs[str(m.fs.condenser_flash.control_volume.properties_out[tf].temperature)]
        ),
        "condenser_heat_duty": np.array(
            traj.vecs[str(m.fs.condenser_flash.heat_duty[tf])]
        ),
        "temperature_z": np.array([traj.vecs[str(soec.temperature_z[tf, iz])] for iz in soec.iznodes]) ,
        "fuel_electrode_temperature_deviation_x": np.array(
            [traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, iz])] for iz in soec.iznodes]
        ),
        "interconnect_temperature_deviation_x":np.array(
            [traj.vecs[str(soec.interconnect.temperature_deviation_x[tf, 1, iz])] for iz in soec.iznodes]
        ),
        "temperature_z_gradient": np.array(
            [traj.vecs[str(soec.dtemperature_z_dz[tf, iz])] for iz in soec.iznodes]
        ),
        "fuel_electrode_gradient": np.array(
            [traj.vecs[str(soec.fuel_electrode.dtemperature_dz[tf, 1, iz])] for iz in soec.iznodes]
        ),
        "fuel_electrode_mixed_partial": np.array(
            [traj.vecs[str(soec.fuel_electrode.d2temperature_dzdt_dummy[tf, 1, iz])] for iz in soec.iznodes]
        ),
        "interconnect_gradient": np.array(
            [traj.vecs[str(soec.interconnect.dtemperature_dz[tf, 1, iz])] for iz in soec.iznodes]
        ),
        "current_density": np.array(
            [traj.vecs[str(soec.current_density[tf, iz])] for iz in soec.iznodes]
        ),
        "feed_heater_temperature": np.array(
            [traj.vecs[str(m.fs.feed_heater.temp_wall_center[tf, z])]
                for z in m.fs.feed_heater.control_volume.length_domain]
        ),
        "sweep_heater_temperature": np.array(
            [traj.vecs[str(m.fs.sweep_heater.temp_wall_center[tf, z])]
                for z in m.fs.sweep_heater.control_volume.length_domain]
        ),
        # "feed_heater_temperature": np.array(
        #     [traj.vecs[str(m.fs.feed_heater.temp_wall[tf, z])]
        #      for z in m.fs.feed_heater.control_volume.length_domain]
        # ),
        # "sweep_heater_temperature": np.array(
        #     [traj.vecs[str(m.fs.sweep_heater.temp_wall[tf, z])]
        #      for z in m.fs.sweep_heater.control_volume.length_domain]
        # ),
        "feed_medium_exchanger_temperature": np.array(
            [traj.vecs[str(m.fs.feed_medium_exchanger.temp_wall_center[tf, z])]
                for z in m.fs.feed_medium_exchanger.cold_side.length_domain]
        ),
        "feed_hot_exchanger_temperature": np.array(
            [traj.vecs[str(m.fs.feed_hot_exchanger.temp_wall_center[tf, z])]
                for z in m.fs.feed_hot_exchanger.cold_side.length_domain]
        ),
        "sweep_exchanger_temperature": np.array(
            [traj.vecs[str(m.fs.sweep_exchanger.temp_wall_center[tf, z])]
             for z in m.fs.sweep_exchanger.cold_side.length_domain]
        ),
        # "feed_medium_exchanger_temperature": np.array(
        #     [traj.vecs[str(m.fs.feed_medium_exchanger.temp_wall_center[tf, z])]
        #      for z in m.fs.feed_medium_exchanger.tube.length_domain]
        # ),
        # "feed_hot_exchanger_temperature": np.array(
        #     [traj.vecs[str(m.fs.feed_hot_exchanger.temp_wall_center[tf, z])]
        #      for z in m.fs.feed_hot_exchanger.tube.length_domain]
        # ),
        # "sweep_exchanger_temperature": np.array(
        #     [traj.vecs[str(m.fs.sweep_exchanger.temp_wall_center[tf, z])]
        #      for z in m.fs.sweep_exchanger.tube.length_domain]
        # ),
    }

    #if PI_control:
    for controller in m.fs.controller_set:
        ctrl_name = controller.local_name
        out_dict[ctrl_name + "_mv_ref"] = np.array(traj.vecs[str(controller.mv_ref[tf])])
        out_dict[ctrl_name + "_setpoint"] = np.array(traj.vecs[str(controller.setpoint[tf])])

    savemat(filename, out_dict, )


def save_results_full_discretization(m, ramp_list, filename):
    """Extracts trajectories of variables of interest from a PETSc trajectory object and saves them in a .mat file.

    Args:
        m: ConcreteModel object (to get variable names from)
        ramp_list: A list of times, in seconds, when ramping starts and stops
        filename: Name .mat file to write
        PI_control: Whether to include reference trajectories from PI controllers

    Returns:
        None
    """

    time_set = m.fs.time.ordered_data()
    soec = m.fs.soc_module.solid_oxide_cell

    out_dict = {
        "ramp_list": np.array(ramp_list),
        "time": np.array(time_set),
        "potential": np.array([pyo.value(soec.potential[t]) for t in time_set]),
        "current": np.array([pyo.value(m.fs.soc_module.total_current[t]) for t in time_set]),
        "soec_fuel_inlet_flow": np.array([pyo.value(m.fs.soc_module.fuel_inlet.flow_mol[t]) for t in time_set]),
        "soec_oxygen_inlet_flow": np.array([pyo.value(m.fs.soc_module.oxygen_inlet.flow_mol[t]) for t in time_set]),
        "fuel_heater_duty": np.array([pyo.value(m.fs.feed_heater.electric_heat_duty[t]) for t in time_set]),
        "sweep_heater_duty": np.array([pyo.value(m.fs.sweep_heater.electric_heat_duty[t]) for t in time_set]),
        "fuel_inlet_H2": np.array([pyo.value(soec.fuel_inlet.mole_frac_comp[t, "H2"]) for t in time_set]),
        "fuel_inlet_H2O": np.array([pyo.value(soec.fuel_inlet.mole_frac_comp[t, "H2O"]) for t in time_set]),
        "fuel_outlet_H2O": np.array([pyo.value(soec.fuel_channel.mole_frac_comp[t, soec.iznodes.last(), "H2O",]
                                               ) for t in time_set]),
        "sweep_inlet_O2": np.array([pyo.value(soec.oxygen_inlet.mole_frac_comp[t, "O2"]) for t in time_set]),
        "sweep_outlet_O2": np.array([pyo.value(soec.oxygen_channel.mole_frac_comp[t, soec.iznodes.first(), "O2"]
                                                  ) for t in time_set]),
        "H2_production": np.array([pyo.value(m.fs.h2_mass_production[t]) for t in time_set]),
        "fuel_outlet_mole_frac_comp_H2": np.array([pyo.value(m.fs.soc_module.fuel_outlet_mole_frac_comp_H2[t]
                                                              ) for t in time_set]),
        "steam_feed_rate": np.array([pyo.value(m.fs.makeup_mix.makeup.flow_mol[t]) for t in time_set]),
        "sweep_feed_rate": np.array([pyo.value(m.fs.sweep_blower.inlet.flow_mol[t]) for t in time_set]),
        "total_electric_power": np.array([pyo.value(m.fs.total_electric_power[t]) for t in time_set]),
        # "efficiency_hhv": np.array(traj.vecs[str(m.fs.efficiency_hhv[tf])]),
        # "efficiency_lhv": np.array(traj.vecs[str(m.fs.efficiency_lhv[tf])]),
        "fuel_inlet_temperature": np.array([pyo.value(soec.fuel_channel.temperature_inlet[t]) for t in time_set]),
        "sweep_inlet_temperature": np.array([pyo.value(soec.oxygen_channel.temperature_inlet[t]) for t in time_set]),
        "stack_core_temperature": np.array([pyo.value(m.fs.stack_core_temperature[t]) for t in time_set]),
        "fuel_outlet_temperature": np.array([pyo.value(soec.fuel_channel.temperature_outlet[t]) for t in time_set]),
        "sweep_outlet_temperature": np.array([pyo.value(soec.oxygen_channel.temperature_outlet[t]) for t in time_set]),
        "fuel_heater_outer_setpoint": np.array([pyo.value(m.fs.feed_heater_outer_controller.setpoint[t]) for t in time_set]),
        "sweep_heater_outer_setpoint": np.array([pyo.value(m.fs.sweep_heater_outer_controller.setpoint[t]) for t in time_set]),
        "product_mole_frac_H2": np.array([pyo.value(m.fs.condenser_split.inlet.mole_frac_comp[t, "H2"]) for t in time_set]),
        "condenser_outlet_temperature": np.array([pyo.value(m.fs.condenser_flash.control_volume.properties_out[t].temperature
                                                              ) for t in time_set]),
        "condenser_heat_duty": np.array([pyo.value(m.fs.condenser_flash.heat_duty[t]) for t in time_set]),
        "temperature_z": np.array([pyo.value(soec.temperature_z[t, iz]) for t in time_set for iz in soec.iznodes]),
        "fuel_electrode_temperature_deviation_x": np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, iz]
                                                                      ) for t in time_set for iz in soec.iznodes]),
        "interconnect_temperature_deviation_x": np.array([pyo.value(soec.interconnect.temperature_deviation_x[t, 1, iz]
                                                                        ) for t in time_set for iz in soec.iznodes]),
        "temperature_z_gradient": np.array([pyo.value(soec.dtemperature_z_dz[t, iz]) for t in time_set for iz in soec.iznodes]
                                             ),
        "fuel_electrode_gradient": np.array([pyo.value(soec.fuel_electrode.dtemperature_dz[t, 1, iz]
                                                        ) for t in time_set for iz in soec.iznodes]),
        "fuel_electrode_mixed_partial": np.array([pyo.value(soec.fuel_electrode.d2temperature_dzdt_dummy[t, 1, iz]
                                                            ) for t in time_set for iz in soec.iznodes]),
        "interconnect_gradient": np.array([pyo.value(soec.interconnect.dtemperature_dz[t, 1, iz]
                                                        ) for t in time_set for iz in soec.iznodes]),
        "current_density": np.array([pyo.value(soec.current_density[t, iz]) for t in time_set for iz in soec.iznodes]),
        "feed_heater_temperature": np.array([pyo.value(m.fs.feed_heater.temp_wall_center[t, z]
                                                       ) for t in time_set for z in m.fs.feed_heater.control_volume.length_domain]),
        "sweep_heater_temperature": np.array([pyo.value(m.fs.sweep_heater.temp_wall_center[t, z]
                                                        ) for t in time_set for z in m.fs.sweep_heater.control_volume.length_domain]),
        "feed_medium_exchanger_temperature": np.array([pyo.value(m.fs.feed_medium_exchanger.temp_wall_center[t, z]
                                                                 ) for t in time_set for z in m.fs.feed_medium_exchanger.cold_side.length_domain]),
        "feed_hot_exchanger_temperature": np.array([pyo.value(m.fs.feed_hot_exchanger.temp_wall_center[t, z]
                                                                ) for t in time_set for z in m.fs.feed_hot_exchanger.cold_side.length_domain]),
        "sweep_exchanger_temperature": np.array([pyo.value(m.fs.sweep_exchanger.temp_wall_center[t, z]
                                                            ) for t in time_set for z in m.fs.sweep_exchanger.cold_side.length_domain]),
            }

    savemat(filename, out_dict, )



