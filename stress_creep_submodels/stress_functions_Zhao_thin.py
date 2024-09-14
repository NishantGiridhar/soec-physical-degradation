import pyomo.environ as pyo
# import pyomo.dae as pyodae
import numpy as np
from idaes.core.util.math import (smooth_abs, safe_sqrt)
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom as dof
import json
import pandas as pd
import pyomo.dae as pyodae
from pyomo.core.expr import Expr_if

# save temperature profiles
def _save_results_dynamic_flowsheet(blk, traj, filename):
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    soec = blk.fs.soc_module.solid_oxide_cell
    time_set = blk.fs.time.ordered_data()
    tf = time_set[-1]
    time_vec = np.array(traj.time)
    # first = soec.fuel_electrode.ixfaces.first()
    data = {
        "time_list": time_vec,
        "H2_production": np.array(traj.vecs[str(blk.fs.h2_mass_production[tf])]),
        "cell potential": np.array(traj.vecs[str(soec.potential[time_set[-1]])]),
        "cell_temperature_znode_average": np.mean(
            [traj.vecs[str(soec.temperature_z[tf, iz])] for iz in soec.iznodes], axis=0),
        "Cell_average_temperature": np.mean([traj.vecs[str(soec.temperature_z[tf, iz])] for iz in soec.iznodes],
                                            axis=0),
        "cell_inlet_temperature": np.array(traj.vecs[str(soec.temperature_z[tf, soec.iznodes.first()])]),
        "cell_outlet_temperature": np.array(traj.vecs[str(soec.temperature_z[tf, soec.iznodes.last()])]),
        "oxygen inlet temperature": np.array(traj.vecs[str(
            soec.oxygen_inlet.temperature[tf])]),
        "fuel inlet temperature": np.array(traj.vecs[str(
            soec.fuel_inlet.temperature[tf])]),
        "fuel_electrode_inlet_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first()])]),
        "fuel_electrode_outlet_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.last()])]),
        "oxygen_electrode_inlet_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first()])]),
        "oxygen_electrode_outlet_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.last()])]),
        "electrolyte_inlet_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first()])]),
        "electrolyte_outlet_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.last()])]),
        # ###############################################################################
        "electrolyte_first_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first()])]),
        "electrolyte_second_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 1])]),
        "electrolyte_third_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 2])]),
        "electrolyte_fourth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 3])]),
        "electrolyte_fifth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 4])]),
        "electrolyte_sixth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 5])]),
        "electrolyte_seventh_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 6])]),
        "electrolyte_eighth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 7])]),
        "electrolyte_ninth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 8])]),
        "electrolyte_tenth_node_temperature": np.array(
            traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 9])]),
        # "electrolyte_eleventh_node_temperature": np.array(
        #     traj.vecs[str(soec.electrolyte.temperature_z[tf, soec.iznodes.first() + 10])]),

        # ###############################################################################
        "oxygen_electrode_first_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first()])]),
        "oxygen_electrode_second_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 1])]),
        "oxygen_electrode_third_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 2])]),
        "oxygen_electrode_fourth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 3])]),
        "oxygen_electrode_fifth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 4])]),
        "oxygen_electrode_sixth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 5])]),
        "oxygen_electrode_seventh_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 6])]),
        "oxygen_electrode_eighth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 7])]),
        "oxygen_electrode_ninth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 8])]),
        "oxygen_electrode_tenth_node_temperature": np.array(
            traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 9])]),
        # "oxygen_electrode_eleventh_node_temperature": np.array(
        #     traj.vecs[str(soec.oxygen_electrode.temperature_z[tf, soec.iznodes.first() + 10])]),

        # ###############################################################################
        "fuel_electrode_first_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first()])]),
        "fuel_electrode_second_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 1])]),
        "fuel_electrode_third_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 2])]),
        "fuel_electrode_fourth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 3])]),
        "fuel_electrode_fifth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 4])]),
        "fuel_electrode_sixth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 5])]),
        "fuel_electrode_seventh_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 6])]),
        "fuel_electrode_eighth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 7])]),
        "fuel_electrode_ninth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 8])]),
        "fuel_electrode_tenth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 9])]),

        "fuel_electrode_temperature_deviation_x_first_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first()])]),
        "fuel_electrode_temperature_deviation_x_second_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 1])]),
        "fuel_electrode_temperature_deviation_x_third_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 2])]),
        "fuel_electrode_temperature_deviation_x_fourth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 3])]),
        "fuel_electrode_temperature_deviation_x_fifth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 4])]),
        "fuel_electrode_temperature_deviation_x_sixth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 5])]),
        "fuel_electrode_temperature_deviation_x_seventh_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 6])]),
        "fuel_electrode_temperature_deviation_x_eighth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 7])]),
        "fuel_electrode_temperature_deviation_x_ninth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 8])]),
        "fuel_electrode_temperature_deviation_x_tenth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 9])]),

        # ###############################################################################
        "fuel_electrode_inletfaces_first_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first()])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first()])]),
        "fuel_electrode_inletfaces_second_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 1])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 1])]),
        "fuel_electrode_inletfaces_third_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 2])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 2])]),
        "fuel_electrode_inletfaces_fourth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 3])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 3])]),
        "fuel_electrode_inletfaces_fifth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 4])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 4])]),
        "fuel_electrode_inletfaces_sixth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 5])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 5])]),
        "fuel_electrode_inletfaces_seventh_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 6])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 6])]),
        "fuel_electrode_inletfaces_eighth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 7])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 7])]),
        "fuel_electrode_inletfaces_ninth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 8])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 8])]),
        "fuel_electrode_inletfaces_tenth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_z[tf, soec.iznodes.first() + 9])]) + \
            np.array(traj.vecs[str(soec.fuel_electrode.temperature_deviation_x[tf, 1, soec.iznodes.first() + 9])]),

        # ###############################################################################

        "fuel_electrode_faces_first_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first()])]),
        "fuel_electrode_faces_second_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 1])]),
        "fuel_electrode_faces_third_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 2])]),
        "fuel_electrode_faces_fourth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 3])]),
        "fuel_electrode_faces_fifth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 4])]),
        "fuel_electrode_faces_sixth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 5])]),
        "fuel_electrode_faces_seventh_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 6])]),
        "fuel_electrode_faces_eighth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 7])]),
        "fuel_electrode_faces_ninth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 8])]),
        "fuel_electrode_faces_tenth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 1, soec.iznodes.first() + 9])]),

        "fuel_electrode_second_faces_first_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first()])]),
        "fuel_electrode_second_faces_second_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 1])]),
        "fuel_electrode_second_faces_third_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 2])]),
        "fuel_electrode_second_faces_fourth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 3])]),
        "fuel_electrode_second_faces_fifth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 4])]),
        "fuel_electrode_second_faces_sixth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 5])]),
        "fuel_electrode_second_faces_seventh_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 6])]),
        "fuel_electrode_second_faces_eighth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 7])]),
        "fuel_electrode_second_faces_ninth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 8])]),
        "fuel_electrode_second_faces_tenth_node_temperature": np.array(
            traj.vecs[str(soec.fuel_electrode.temperature_xfaces_var[tf, 2, soec.iznodes.first() + 9])]),
        # ###############################################################################
        "fuel_electrode_first_face_thermal_stress": np.array
                                                    ([traj.vecs[str(
                                                        soec.fuel_electrode_residual_thermal_stress_var[tf, 1, iz])] for
                                                      iz in soec.iznodes]) * 1e-6,
        "fuel_electrode_second_face_thermal_stress": np.array
                                                     ([traj.vecs[str(
                                                         soec.fuel_electrode_residual_thermal_stress_var[tf, 2, iz])]
                                                       for iz in soec.iznodes]) * 1e-6,
        "electrolyte_thermal_stress": np.array
                                      ([traj.vecs[str(soec.electrolyte_residual_thermal_stress_var[tf, iz])] for iz in
                                        soec.iznodes]) * 1e-6,
        "oxygen_electrode_thermal_stress": np.array
                                           ([traj.vecs[str(soec.oxygen_electrode_residual_thermal_stress_var[tf, iz])]
                                             for iz in soec.iznodes]) * 1e-6,

        # ###############################################################################

        "fuel_electrode_faces_first_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first()])]) * 1e-6,
        "fuel_electrode_faces_second_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 1])]) * 1e-6,
        "fuel_electrode_faces_third_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 2])]) * 1e-6,
        "fuel_electrode_faces_fourth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 3])]) * 1e-6,
        "fuel_electrode_faces_fifth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 4])]) * 1e-6,
        "fuel_electrode_faces_sixth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 5])]) * 1e-6,
        "fuel_electrode_faces_seventh_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 6])]) * 1e-6,
        "fuel_electrode_faces_eighth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 7])]) * 1e-6,
        "fuel_electrode_faces_ninth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 8])]) * 1e-6,
        "fuel_electrode_faces_tenth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 1, soec.iznodes.first() + 9])]) * 1e-6,

        "fuel_electrode_second_faces_first_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first()])]) * 1e-6,
        "fuel_electrode_second_faces_second_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 1])]) * 1e-6,
        "fuel_electrode_second_faces_third_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 2])]) * 1e-6,
        "fuel_electrode_second_faces_fourth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 3])]) * 1e-6,
        "fuel_electrode_second_faces_fifth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 4])]) * 1e-6,
        "fuel_electrode_second_faces_sixth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 5])]) * 1e-6,
        "fuel_electrode_second_faces_seventh_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 6])]) * 1e-6,
        "fuel_electrode_second_faces_eighth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 7])]) * 1e-6,
        "fuel_electrode_second_faces_ninth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 8])]) * 1e-6,
        "fuel_electrode_second_faces_tenth_node_thermal_stress": np.array(
            traj.vecs[str(soec.fuel_electrode_residual_thermal_stress_var[tf, 2, soec.iznodes.first() + 9])]) * 1e-6,

        # ###############################################################################
        "fuel_electrode_faces_first_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first()])]),
        "fuel_electrode_faces_second_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 1])]),
        "fuel_electrode_faces_third_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 2])]),
        "fuel_electrode_faces_fourth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 3])]),
        "fuel_electrode_faces_fifth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 4])]),
        "fuel_electrode_faces_sixth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 5])]),
        "fuel_electrode_faces_seventh_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 6])]),
        "fuel_electrode_faces_eighth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 7])]),
        "fuel_electrode_faces_ninth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 8])]),
        "fuel_electrode_faces_tenth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 1, soec.iznodes.first() + 9])]),

        "fuel_electrode_second_faces_first_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first()])]),
        "fuel_electrode_second_faces_second_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 1])]),
        "fuel_electrode_second_faces_third_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 2])]),
        "fuel_electrode_second_faces_fourth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 3])]),
        "fuel_electrode_second_faces_fifth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 4])]),
        "fuel_electrode_second_faces_sixth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 5])]),
        "fuel_electrode_second_faces_seventh_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 6])]),
        "fuel_electrode_second_faces_eighth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 7])]),
        "fuel_electrode_second_faces_ninth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 8])]),
        "fuel_electrode_second_faces_tenth_node_failure_probability": np.array(
            traj.vecs[str(soec.fuel_electrode_failure_probability_var[tf, 2, soec.iznodes.first() + 9])]),
        # ###############################################################################
        "effiency_hhv": np.array(traj.vecs[str(blk.fs.efficiency_hhv[tf])]),
        'cell_integral_efficiency': pyo.value(blk.fs.integral_efficiency),
        "electrolyte_failure_probability": np.array
                                      ([traj.vecs[str(soec.electrolyte_failure_probability_var[tf, iz])] for iz in
                                        soec.iznodes]),

    }

    with open(filename + '.json', 'w') as fp:
        json.dump(data, fp, cls=NumpyEncoder, indent=4)

    print('Finishing saving temperature & stress results')

def save_results_full_discretization(blk, file_name):
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    soec = blk.fs.soc_module.solid_oxide_cell
    time_list = []
    cell_inlet_temperature_list = []
    fe_first_node_temperature_full_discrete = []
    fe_second_node_temperature_full_discrete = []
    fe_third_node_temperature_full_discrete = []
    fe_fourth_node_temperature_full_discrete = []
    fe_fifth_node_temperature_full_discrete = []
    fe_sixth_node_temperature_full_discrete = []
    fe_seventh_node_temperature_full_discrete = []
    fe_eighth_node_temperature_full_discrete = []
    fe_ninth_node_temperature_full_discrete = []
    fe_tenth_node_temperature_full_discrete = []
    el_first_node_temperature_full_discrete = []
    el_second_node_temperature_full_discrete = []
    el_third_node_temperature_full_discrete = []
    el_fourth_node_temperature_full_discrete = []
    el_fifth_node_temperature_full_discrete = []
    el_sixth_node_temperature_full_discrete = []
    el_seventh_node_temperature_full_discrete = []
    el_eighth_node_temperature_full_discrete = []
    el_ninth_node_temperature_full_discrete = []
    el_tenth_node_temperature_full_discrete = []
    oe_first_node_temperature_full_discrete = []
    oe_second_node_temperature_full_discrete = []
    oe_third_node_temperature_full_discrete = []
    oe_fourth_node_temperature_full_discrete = []
    oe_fifth_node_temperature_full_discrete = []
    oe_sixth_node_temperature_full_discrete = []
    oe_seventh_node_temperature_full_discrete = []
    oe_eighth_node_temperature_full_discrete = []
    oe_ninth_node_temperature_full_discrete = []
    oe_tenth_node_temperature_full_discrete = []
    H2_production_list = []

    for t in blk.fs.time:
        time_list.append(t)
        H2_production_list.append(pyo.value(blk.fs.h2_mass_production[t]))
        cell_inlet_temperature_list.append(pyo.value(soec.temperature_z[t, soec.iznodes.first()]))
        fe_first_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first()]))
        fe_second_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 1]))
        fe_third_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 2]))
        fe_fourth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 3]))
        fe_fifth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 4]))
        fe_sixth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 5]))
        fe_seventh_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 6]))
        fe_eighth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 7]))
        fe_ninth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 8]))
        fe_tenth_node_temperature_full_discrete.append(
            pyo.value(soec.fuel_electrode.temperature_xfaces_var[t, 1, soec.iznodes.first() + 9]))
        el_first_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first()]))
        el_second_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 1]))
        el_third_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 2]))
        el_fourth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 3]))
        el_fifth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 4]))
        el_sixth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 5]))
        el_seventh_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 6]))
        el_eighth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 7]))
        el_ninth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 8]))
        el_tenth_node_temperature_full_discrete.append(
            pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 9]))
        oe_first_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first()]))
        oe_second_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 1]))
        oe_third_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 2]))
        oe_fourth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 3]))
        oe_fifth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 4]))
        oe_sixth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 5]))
        oe_seventh_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 6]))
        oe_eighth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 7]))
        oe_ninth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 8]))
        oe_tenth_node_temperature_full_discrete.append(
            pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 9]))

    data = {
        "time_list_full_discretization": np.array(time_list),
        "cell_inlet_temperature_full_discrete": np.array(cell_inlet_temperature_list),
        "fuel_electrode_first_node_temperature_full_discrete":
            np.array(fe_first_node_temperature_full_discrete),
        "fuel_electrode_second_node_temperature_full_discrete":
            np.array(fe_second_node_temperature_full_discrete),
        "fuel_electrode_third_node_temperature_full_discrete":
            np.array(fe_third_node_temperature_full_discrete),
        "fuel_electrode_fourth_node_temperature_full_discrete":
            np.array(fe_fourth_node_temperature_full_discrete),
        "fuel_electrode_fifth_node_temperature_full_discrete":
            np.array(fe_fifth_node_temperature_full_discrete),
        "fuel_electrode_sixth_node_temperature_full_discrete":
            np.array(fe_sixth_node_temperature_full_discrete),
        "fuel_electrode_seventh_node_temperature_full_discrete":
            np.array(fe_seventh_node_temperature_full_discrete),
        "fuel_electrode_eighth_node_temperature_full_discrete":
            np.array(fe_eighth_node_temperature_full_discrete),
        "fuel_electrode_ninth_node_temperature_full_discrete":
            np.array(fe_ninth_node_temperature_full_discrete),
        "fuel_electrode_tenth_node_temperature_full_discrete":
            np.array(fe_tenth_node_temperature_full_discrete),
        "electrolyte_first_node_temperature_full_discrete":
            np.array(el_first_node_temperature_full_discrete),
        "electrolyte_second_node_temperature_full_discrete":
            np.array(el_second_node_temperature_full_discrete),
        "electrolyte_third_node_temperature_full_discrete":
            np.array(el_third_node_temperature_full_discrete),
        "electrolyte_fourth_node_temperature_full_discrete":
            np.array(el_fourth_node_temperature_full_discrete),
        "electrolyte_fifth_node_temperature_full_discrete":
            np.array(el_fifth_node_temperature_full_discrete),
        "electrolyte_sixth_node_temperature_full_discrete":
            np.array(el_sixth_node_temperature_full_discrete),
        "electrolyte_seventh_node_temperature_full_discrete":
            np.array(el_seventh_node_temperature_full_discrete),
        "electrolyte_eighth_node_temperature_full_discrete":
            np.array(el_eighth_node_temperature_full_discrete),
        "electrolyte_ninth_node_temperature_full_discrete":
            np.array(el_ninth_node_temperature_full_discrete),
        "electrolyte_tenth_node_temperature_full_discrete":
            np.array(el_tenth_node_temperature_full_discrete),
        "oxygen_electrode_first_node_temperature_full_discrete":
            np.array(oe_first_node_temperature_full_discrete),
        "oxygen_electrode_second_node_temperature_full_discrete":
            np.array(oe_second_node_temperature_full_discrete),
        "oxygen_electrode_third_node_temperature_full_discrete":
            np.array(oe_third_node_temperature_full_discrete),
        "oxygen_electrode_fourth_node_temperature_full_discrete":
            np.array(oe_fourth_node_temperature_full_discrete),
        "oxygen_electrode_fifth_node_temperature_full_discrete":
            np.array(oe_fifth_node_temperature_full_discrete),
        "oxygen_electrode_sixth_node_temperature_full_discrete":
            np.array(oe_sixth_node_temperature_full_discrete),
        "oxygen_electrode_seventh_node_temperature_full_discrete":
            np.array(oe_seventh_node_temperature_full_discrete),
        "oxygen_electrode_eighth_node_temperature_full_discrete":
            np.array(oe_eighth_node_temperature_full_discrete),
        "oxygen_electrode_ninth_node_temperature_full_discrete":
            np.array(oe_ninth_node_temperature_full_discrete),
        "oxygen_electrode_tenth_node_temperature_full_discrete":
            np.array(oe_tenth_node_temperature_full_discrete),
        "H2_production": np.array(H2_production_list),

        }

    with open('health_submodels/_full_discretization_temperature_data_' + file_name + '.json', 'w') as fp:
        json.dump(data, fp, cls=NumpyEncoder, indent=4)

    print('Save results for full discretization temperature data complete')

# define the stress functions
def _make_thermal_stress(blk, creep_model = True, stress_relaxation = True):

    # physical degradation section
    # Reference: Zhao et al., 2019
    reference_temperature = 1200 + 273.15  # K
    reference_temperature_oe = 1200 + 273.15  # K

    fe_modulus = 53  # Pa
    fe_cte = 12.41  # K^-1
    fe_pc = 0.39
    e_modulus = 144  # Pa
    e_cte = 10.35  # K^-1
    e_pc = 0.29
    oe_modulus = 52  # Pa
    oe_cte = 12.19  # K^-1
    oe_pc = 0.28

    # save alias variables
    fs = blk.fs
    soec = fs.soc_module.solid_oxide_cell
    soec.fuel_electrode_young_modulus_sc = pyo.Var(initialize=fe_modulus)
    soec.electrolyte_young_modulus_sc = pyo.Var(initialize=e_modulus)
    soec.oxygen_electrode_young_modulus_sc = pyo.Var(initialize=oe_modulus)
    soec.fuel_electrode_cte_sc = pyo.Var(initialize=fe_cte)
    soec.electrolyte_cte_sc = pyo.Var(initialize=e_cte)
    soec.oxygen_electrode_cte_sc = pyo.Var(initialize=oe_cte)
    soec.fuel_electrode_young_modulus_sc.fix()
    soec.electrolyte_young_modulus_sc.fix()
    soec.oxygen_electrode_young_modulus_sc.fix()
    soec.fuel_electrode_cte_sc.fix()
    soec.electrolyte_cte_sc.fix()
    soec.oxygen_electrode_cte_sc.fix()

    # soec.reference_temperature = pyo.Var(initialize=reference_temperature)
    # soec.reference_temperature_oe = pyo.Var(initialize=reference_temperature_oe)
    # soec.reference_temperature.fix()
    # soec.reference_temperature_oe.fix()

    soec.fuel_electrode_poisson_coefficient = pyo.Var(initialize=fe_pc)
    soec.oxygen_electrode_poisson_coefficient = pyo.Var(initialize=oe_pc)
    soec.electrolyte_poisson_coefficient = pyo.Var(initialize=e_pc)
    soec.fuel_electrode_poisson_coefficient.fix()
    soec.electrolyte_poisson_coefficient.fix()
    soec.oxygen_electrode_poisson_coefficient.fix()

    # set alias with variables
    fe_ixfaces = soec.fuel_electrode.ixfaces
    # fe_ixnodes = soec.fuel_electrode.ixnodes
    fe_iznodes = soec.fuel_electrode.iznodes
    oe_iznodes = soec.oxygen_electrode.iznodes
    e_iznodes = soec.electrolyte.iznodes
    fe_length_x = soec.fuel_electrode.length_x
    # oe_length_x = soec.oxygen_electrode.length_x
    # e_length_x = soec.electrolyte.length_x

    scf_modulus = 1e9
    scf_cte = 1e-6

    # # # applying transformation factory to the model with respect to time
    # # time_nfe = len(fs.time) - 1
    # # pyo.TransformationFactory("dae.finite_difference").apply_to(
    # #     fs, nfe=time_nfe, wrt=fs.time, scheme="BACKWARD"
    # # )
    #
    # # if thermal_stress:
    # # property model
    # @soec.Expression(blk.fs.time, doc='Young modulus in fuel electrode')
    # def fuel_electrode_young_modulus_sc(b, t):
    #     if t == 0:
    #         return 53
    #     temperature_fe = sum(b.fuel_electrode.temperature_xfaces[t, 1, iz] for iz in fe_iznodes) / len(fe_iznodes)
    #     return 66.37 * (1 - 3.67e-4 * temperature_fe)  # reference: Mounir et al., Energy 66, 2014
    #
    # @soec.Expression(blk.fs.time, doc='Young modulus in electrolyte')
    # def electrolyte_young_modulus_sc(b, t):
    #     if t == 0:
    #         return 144
    #     temperature_e = sum(b.electrolyte.temperature[t, iz] for iz in e_iznodes) / len(e_iznodes)
    #     return 0.0547 * (temperature_e - 273.15) + 101.47
    #
    # @soec.Expression(blk.fs.time, doc='Young modulus in oxygen electrode')
    # def oxygen_electrode_young_modulus_sc(b, t):
    #     if t == 0:
    #         return 52
    #     temperature_oe = sum(b.oxygen_electrode.temperature[t, iz] for iz in oe_iznodes) / len(oe_iznodes)
    #     return 0.0046 * (temperature_oe - 273.15) + 44.412
    #
    # @soec.Expression(blk.fs.time, doc='CTE scaled in fuel electrode')
    # def fuel_electrode_cte_sc(b, t):
    #     if t == 0:
    #         return 12.41
    #     temperature_fe = sum(b.fuel_electrode.temperature_xfaces[t, 1, iz] for iz in fe_iznodes) / len(fe_iznodes)
    #     return 0.0043 * (temperature_fe - 273.15) + 10.05
    #
    # @soec.Expression(blk.fs.time, doc='CTE scaled in electrolyte')
    # def electrolyte_cte_sc(b, t):
    #     if t == 0:
    #         return 10.35
    #     temperature_e = sum(b.electrolyte.temperature[t, iz] for iz in e_iznodes) / len(e_iznodes)
    #     return 7.507 * pyo.exp(0.000345 * temperature_e) - 5.855 * pyo.exp(
    #         -0.00694 * temperature_e)  # ref: (Osman et al., 2021)

    # @soec.Expression(blk.fs.time, doc='CTE scaled in oxygen electrode')
    # def oxygen_electrode_cte_sc(b, t):
    #     if t == 0:
    #         return 12.19
    #     temperature_oe = sum(b.oxygen_electrode.temperature[t, iz] for iz in oe_iznodes) / len(oe_iznodes)
    #     return 10.5 * (1 + 1.47e-4 * temperature_oe)  # Ref: Mounir et al., Energy 66, 2014

    @soec.Expression()
    def fuel_electrode_young_modulus(b):
        return scf_modulus * b.fuel_electrode_young_modulus_sc

    @soec.Expression()
    def electrolyte_young_modulus(b):
        return scf_modulus * b.electrolyte_young_modulus_sc

    @soec.Expression()
    def oxygen_electrode_young_modulus(b):
        return scf_modulus * b.oxygen_electrode_young_modulus_sc

    @soec.Expression()
    def fuel_electrode_cte(b):
        return scf_cte * b.fuel_electrode_cte_sc

    @soec.Expression()
    def electrolyte_cte(b):
        return scf_cte * b.electrolyte_cte_sc

    @soec.Expression()
    def oxygen_electrode_cte(b):
        return scf_cte * b.oxygen_electrode_cte_sc

    # parameter for thermal stress calculations

    feE = soec.fuel_electrode_young_modulus
    fecte = soec.fuel_electrode_cte
    eE = soec.electrolyte_young_modulus
    ecte = soec.electrolyte_cte
    oeE = soec.oxygen_electrode_young_modulus
    oecte = soec.oxygen_electrode_cte
    fepc = soec.fuel_electrode_poisson_coefficient
    oepc = soec.oxygen_electrode_poisson_coefficient
    epc = soec.electrolyte_poisson_coefficient

    if soec.config.thin_electrolyte and soec.config.thin_oxygen_electrode:
        oe_thickness = 40e-6
        e_thickness = 10.5e-6
        # if thermal_stress:
        soec.oxygen_electrode.length_x = pyo.Var(initialize=oe_thickness)
        soec.electrolyte.length_x = pyo.Var(initialize=e_thickness)
        soec.oxygen_electrode.length_x.fix()
        soec.electrolyte.length_x.fix()

    oe_length_x = soec.oxygen_electrode.length_x
    e_length_x = soec.electrolyte.length_x

    # create set of length x
    number_of_element_spacing = 28  # changing from 30 to 28 to convert model at SOEC level
    nfe_length_x_fe = int(4 / 7 * number_of_element_spacing)
    nfe_length_x_e = int(1 / 7 * number_of_element_spacing)
    nfe_length_x_oe = int(2 / 7 * number_of_element_spacing)

    soec.fe_x_tolist = np.linspace(0, pyo.value(fe_length_x), nfe_length_x_fe + 1).tolist()
    soec.e_x_tolist = np.linspace(0, pyo.value(e_length_x), nfe_length_x_e + 1).tolist()
    soec.oe_x_tolist = np.linspace(0, pyo.value(oe_length_x), nfe_length_x_oe + 1).tolist()

    soec.fe_xfaces = pyo.Set(initialize=range(0, len(soec.fe_x_tolist)))
    soec.e_xfaces = pyo.Set(initialize=range(0, len(soec.e_x_tolist)))
    soec.oe_xfaces = pyo.Set(initialize=range(0, len(soec.oe_x_tolist)))

    fe_x_tolist_reversed = list(reversed(soec.fe_x_tolist))

    # set alias with variables
    fe_xfaces = soec.fe_xfaces
    e_xfaces = soec.e_xfaces
    oe_xfaces = soec.oe_xfaces

    ############################################################################################

    # if thermal_stress:
    # ### value obtained from Nakajo et al., Ceramic Internationals, 3907-3927 (2012) ####
    fesigmanot = soec.fuel_electrode_sigmanot = pyo.Var(initialize=79.1)
    fewebp = soec.fuel_electrode_weib_m = pyo.Var(initialize=7.0)
    reference_volume_fe = soec.fuel_electrode_reference_volume = pyo.Var(initialize=4.812e-9)
    oesigmanot = soec.oxygen_electrode_sigmanot = pyo.Var(initialize=75)
    oewebp = soec.oxygen_electrode_weib_m = pyo.Var(initialize=3.7)
    reference_volume_oe = soec.oxygen_electrode_reference_volume = pyo.Var(initialize=2.836e-9)
    elsigmanot = soec.electrolyte_sigmanot = pyo.Var(initialize=131)
    elwebp = soec.electrolyte_weib_m = pyo.Var(initialize=5.6)
    reference_volume_el = soec.electrolyte_reference_volume = pyo.Var(initialize=0.547e-9)
    fracture_strength_el = soec.electrolyte_fracture_strength = pyo.Var(initialize=300)

    # fixing failure probability parameters
    soec.fuel_electrode_sigmanot.fix()
    soec.fuel_electrode_weib_m.fix()
    soec.fuel_electrode_reference_volume.fix()
    soec.oxygen_electrode_sigmanot.fix()
    soec.oxygen_electrode_weib_m.fix()
    soec.oxygen_electrode_reference_volume.fix()
    soec.electrolyte_sigmanot.fix()
    soec.electrolyte_weib_m.fix()
    soec.electrolyte_reference_volume.fix()
    soec.electrolyte_fracture_strength.fix()  # fracture strength of electrolyte

    # parameters for thermal stress model
    # soec.bending_axis = pyo.Expression(rule=lambda b: (-feE[0] / (1 - fepc) * fe_length_x ** 2 + oeE / (1 - oepc) * (
    #         2 * oe_length_x * e_length_x + oe_length_x ** 2) + eE / (1 - epc) * e_length_x ** 2) / (
    #         2 * (feE / (1 - fepc) * fe_length_x + eE / (1 - epc) * e_length_x + oeE / (1 - oepc) * oe_length_x)))

    # if thermal_stress:
    @soec.Expression(doc='bending axis location')
    def bending_axis(b):
        return (-(feE) / (1 - fepc) * fe_length_x ** 2 + oeE / (1 - oepc) * (
                2 * oe_length_x * e_length_x + oe_length_x ** 2) + eE / (1 - epc) * e_length_x ** 2) / (
                       2 * (feE / (1 - fepc) * fe_length_x + eE / (1 - epc) * e_length_x + oeE / (
                       1 - oepc) * oe_length_x))

    # ############################################################################################
    # write expressions to define temperature profiles along thickness of fuel electrode, electrolyte, and oxygen electrode
    # temperature along thickness of fuel electrode, if ix == (first,last) return the value, or apply linear interpolation
    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='temperature profile in fuel electrode')
    def fuel_electrode_temperature_xfaces(b, t, ix, iz):
        temperature_first_face = b.fuel_electrode.temperature_xfaces[t, fe_ixfaces.first(), iz]
        temperature_last_face = b.fuel_electrode.temperature_xfaces[t, fe_ixfaces.last(), iz]
        if ix == fe_xfaces.first() or ix == fe_xfaces.last():
            return temperature_first_face if ix == fe_xfaces.first() else temperature_last_face
        else:
            return temperature_first_face + (temperature_last_face - temperature_first_face) * ix / fe_xfaces.last()

    """
    Since the temperature variation across the thickness is minimal, we might consider using a simple approach 
    where the temperature changes linearly from the base temperature at iz with a small gradient, 
    perhaps defined as a proportion of the base temperature. 
    """
    @soec.Expression(fs.time, e_xfaces, e_iznodes, doc='temperature profile in electrolyte')
    def electrolyte_temperature_xfaces(b, t, ix, iz):
        # temperature_first_face = b.fuel_electrode.temperature_xfaces[t, fe_ixfaces.last(), iz]
        temperature_z = b.electrolyte.temperature[t, iz]  # Base temperature at each z-node
        # Assume a small proportional change across the x-direction, like 0.01% of the base temperature
        temperature_gradient = 0.0001 * temperature_z
        # Linearly interpolate from the base temperature up by the gradient over the x-span
        fraction = ix / e_xfaces.last()  # Fractional position along x-direction
        return temperature_z + temperature_gradient * fraction

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes, doc='temperature profile in oxygen electrode')
    def oxygen_electrode_temperature_xfaces(b, t, ix, iz):
        # temperature_first_face = b.fuel_electrode.temperature_xfaces[t, fe_ixfaces.last(), iz]
        temperature_z = b.oxygen_electrode.temperature[t, iz]
        temperature_gradient = 0.0001 * temperature_z
        fraction = ix / oe_xfaces.last()
        return temperature_z + temperature_gradient * fraction

    soec.reference_temperature_fuel_electrode = pyo.Var(fe_xfaces, fe_iznodes, initialize=1200 + 273.15)
    soec.reference_temperature_electrolyte = pyo.Var(e_xfaces, e_iznodes, initialize=1200 + 273.15)
    soec.reference_temperature_oxygen_electrode = pyo.Var(oe_xfaces, oe_iznodes, initialize=1200 + 273.15)
    soec.reference_temperature_fuel_electrode.fix(1200 + 273.15)
    soec.reference_temperature_electrolyte.fix(1200 + 273.15)
    soec.reference_temperature_oxygen_electrode.fix(1200 + 273.15)

    # main equations for stress model without considering creep model
    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='temperature change in fuel electrode')
    def fuel_electrode_temperature_change(b, t, ix, iz):
        return b.fuel_electrode_temperature_xfaces[t, ix, iz] - b.reference_temperature_fuel_electrode[ix, iz]

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                     doc='temperature change in oxygen electrode')
    def oxygen_electrode_temperature_change(b, t, ix, iz):
        return b.oxygen_electrode_temperature_xfaces[t, ix, iz] - b.reference_temperature_oxygen_electrode[ix, iz]

    @soec.Expression(fs.time, e_xfaces, e_iznodes,
                     doc='temperature change in electrolyte')
    def electrolyte_temperature_change(b, t, ix, iz):
        return b.electrolyte_temperature_xfaces[t, ix, iz] - soec.reference_temperature_electrolyte[ix, iz]

    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='uniform strain component in fuel electrode')
    def fuel_electrode_uniform_strain_component(b, t, ix, iz):
        deltaT = b.fuel_electrode_temperature_change[t, ix, iz]
        return fecte * deltaT + (((ecte - fecte) * eE / (1 - epc) * e_length_x) + (
                (oecte - fecte) * oeE / (1 - oepc) * oe_length_x)) * deltaT / (
                       feE / (1 - fepc) * fe_length_x + eE / (1 - epc) * e_length_x + oeE / (1 - oepc) * oe_length_x)

    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='radius of curvature of fuel electrode')
    def fuel_electrode_reverse_curvature_radius(b, t, ix, iz):
        deltaT = b.fuel_electrode_temperature_change[t, ix, iz]
        return 6 * deltaT * (feE / (1 - fepc) * fe_length_x * eE / (1 - epc) * e_length_x * (fe_length_x + e_length_x
                                                                                             ) * (
                                         ecte - fecte) + feE / (
                                     1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (
                                     fe_length_x + 2 * e_length_x + oe_length_x) * (oecte - fecte) + eE / (
                                     1 - epc) * e_length_x * oeE / (1 - oepc) * oe_length_x * (e_length_x + oe_length_x
                                                                                               ) * (oecte - ecte)) / (
                       (feE / (1 - fepc)) ** 2 * (fe_length_x) ** 4 + (eE / (1 - epc)) ** 2 * (e_length_x) ** 4 + (
                       oeE / (1 - oepc)) ** 2 * (oe_length_x) ** 4 + 2 * feE / (1 - fepc) * fe_length_x * eE / (1 - epc
                                                                                                                ) * e_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * e_length_x + 2 * e_length_x ** 2
                       ) + 2 * eE / (1 - epc) * e_length_x * oeE / (1 - oepc) * oe_length_x * (
                               2 * e_length_x ** 2 + 3 * e_length_x * oe_length_x + 2 * oe_length_x ** 2
                       ) + 2 * feE / (1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * oe_length_x + 2 * oe_length_x ** 2
                       ) + 2 * feE / (1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (6 * e_length_x * (
                       fe_length_x + e_length_x + oe_length_x)))

    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='total strain in fuel electrode')
    def fuel_electrode_total_strain(b, t, ix, iz):
        ixr = fe_x_tolist_reversed[ix]
        return (b.fuel_electrode_uniform_strain_component[t, ix, iz] - (
                b.bending_axis + ixr) * b.fuel_electrode_reverse_curvature_radius[t, ix, iz])

    @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='residual thermal stress in fuel electrode')
    def fuel_electrode_residual_thermal_stress(b, t, ix, iz):
        deltaT = b.fuel_electrode_temperature_change[t, ix, iz]
        return (
                feE / (1 - fepc) * (b.fuel_electrode_total_strain[t, ix, iz] - fecte * deltaT))

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                        doc='uniform strain component in oxygen electrode')
    def oxygen_electrode_uniform_strain_component(b, t, ix, iz):
        deltaT = b.oxygen_electrode_temperature_change[t, ix, iz]
        return fecte * deltaT + (((ecte - fecte) * eE / (1 - epc) * e_length_x) + ((oecte - fecte) * oeE / (
                1 - oepc) * oe_length_x)) * deltaT / (
                           feE / (1 - fepc) * fe_length_x + eE / (1 - epc) * e_length_x + oeE / (
                           1 - oepc) * oe_length_x)

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                     doc='radius of curvature in oxygen electrode')
    def oxygen_electrode_reverse_curvature_radius(b, t, ix, iz):
        deltaT = b.oxygen_electrode_temperature_change[t, ix, iz]
        return 6 * deltaT * (feE / (1 - fepc) * fe_length_x * eE / (1 - epc) * e_length_x * (
                fe_length_x + e_length_x) * (ecte - fecte) + feE / (1 - fepc) * fe_length_x * oeE / (
                                     1 - oepc) * oe_length_x * (
                                     fe_length_x + 2 * e_length_x + oe_length_x) * (oecte - fecte) + eE / (
                                     1 - epc) * e_length_x * oeE / (1 - oepc) * oe_length_x * (
                                         e_length_x + oe_length_x) * (
                                     oecte - ecte)
                             ) / (
                       (feE / (1 - fepc)) ** 2 * (fe_length_x) ** 4 + (eE / (1 - epc)) ** 2 * (e_length_x) ** 4 + (
                       oeE / (1 - oepc)) ** 2 * (oe_length_x) ** 4 + 2 * feE / (1 - fepc) * fe_length_x * eE / (
                               1 - epc) * e_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * e_length_x + 2 * e_length_x ** 2) + 2 * eE / (
                               1 - epc) * e_length_x * oeE / (1 - oepc) * oe_length_x * (
                               2 * e_length_x ** 2 + 3 * e_length_x * oe_length_x + 2 * oe_length_x ** 2) + 2 * feE / (
                               1 - fepc) * fe_length_x * oeE / (
                               1 - oepc) * oe_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * oe_length_x + 2 * oe_length_x ** 2
                       ) + 2 * feE / (1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (
                               6 * e_length_x * (fe_length_x + e_length_x + oe_length_x)))

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                        doc='total strain in oxygen electrode')
    def oxygen_electrode_total_strain(b, t, ix, iz):
        y_oe = oe_length_x
        return (b.oxygen_electrode_uniform_strain_component[t, ix, iz]) + (
                    (y_oe + b.electrolyte.length_x) - b.bending_axis) * \
               b.oxygen_electrode_reverse_curvature_radius[t, ix, iz]

    @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                        doc='residual thermal stress in oxygen electrode')
    def oxygen_electrode_residual_thermal_stress(b, t, ix, iz):
        deltaT = b.oxygen_electrode_temperature_change[t, ix, iz]
        return (
                oeE / (1 - oepc) * (b.oxygen_electrode_total_strain[t, ix, iz] - oecte * deltaT))

    # parameters for electrolyte
    @soec.Expression(fs.time, e_xfaces, e_iznodes,
                        doc='uniform strain component in electrolyte')
    def electrolyte_uniform_strain_component(b, t, ix, iz):
        deltaT = b.electrolyte_temperature_change[t, ix, iz]
        return fecte * deltaT + (((ecte - fecte) * eE / (1 - epc) * e_length_x) + ((oecte - fecte) * oeE / (
                1 - oepc) * oe_length_x)) * deltaT / (
                           feE / (1 - fepc) * fe_length_x + eE / (1 - epc) * e_length_x + oeE / (
                           1 - oepc) * oe_length_x)

    @soec.Expression(fs.time, e_xfaces, e_iznodes,
                        doc='radius of curvature of electrolyte')
    def electrolyte_reverse_curvature_radius(b, t, ix, iz):
        deltaT = b.electrolyte_temperature_change[t, ix, iz]
        return 6 * deltaT * (
                    feE / (1 - fepc) * fe_length_x * eE / (1 - epc) * e_length_x * (fe_length_x + e_length_x) * (
                    ecte - fecte) + feE / (
                            1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (
                            fe_length_x + 2 * e_length_x + oe_length_x) * (oecte - fecte) + eE / (
                            1 - epc) * e_length_x * oeE / (1 - oepc) * oe_length_x * (e_length_x + oe_length_x) * (
                            oecte - ecte)) / (
                       (feE / (1 - fepc)) ** 2 * (fe_length_x) ** 4 + (eE / (1 - epc)) ** 2 * (e_length_x) ** 4 + (
                       oeE / (1 - oepc)) ** 2 * (oe_length_x) ** 4 + 2 * feE / (1 - fepc) * fe_length_x * eE / (
                               1 - epc) * e_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * e_length_x + 2 * e_length_x ** 2) + 2 * eE / (
                               1 - epc) * e_length_x * oeE / (
                               1 - oepc) * oe_length_x * (
                               2 * e_length_x ** 2 + 3 * e_length_x * oe_length_x + 2 * oe_length_x ** 2) + 2 * feE / (
                               1 - fepc) * fe_length_x * oeE / (1 - oepc) * oe_length_x * (
                               2 * fe_length_x ** 2 + 3 * fe_length_x * oe_length_x + 2 * oe_length_x ** 2
                       ) + 2 * feE / (1 - fepc) * fe_length_x * oeE / (
                               1 - oepc) * oe_length_x * (6 * e_length_x * (fe_length_x + e_length_x + oe_length_x)))

    @soec.Expression(fs.time, e_xfaces, e_iznodes,
                        doc='total strain in electrolyte')
    def electrolyte_total_strain(b, t, ix, iz):
        y_el = e_length_x
        return (b.electrolyte_uniform_strain_component[t, ix, iz]) + (y_el - b.bending_axis
                                                                ) * b.electrolyte_reverse_curvature_radius[t, ix, iz]

    @soec.Expression(fs.time, e_xfaces, e_iznodes,
                        doc='residual thermal stress in electrolyte')
    def electrolyte_residual_thermal_stress(b, t, ix, iz):
        deltaT = b.electrolyte_temperature_change[t, ix, iz]
        return (eE / (1 - epc) * (b.electrolyte_total_strain[t, ix, iz] - ecte * deltaT))

    # @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
    #                     doc='Weibull statistics for Failure probability in fuel electrode layer')
    # def fuel_electrode_failure_probability(b, t, ix, iz):
    #     sigma_fe = b.fuel_electrode_residual_thermal_stress[t, ix, iz] * 1e-6
    #     volume = fe_length_x * b.length_y * b.length_z
    #     if pyo.value(sigma_fe) <= 0:
    #         return 0
    #     return 1 - pyo.exp(-(volume/reference_volume_fe) * (sigma_fe / fesigmanot) ** fewebp)

    # @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
    #                     doc='Weibull statistics for Failure probability in oxygen electrode layer')
    # def oxygen_electrode_weib_distr_failure(b, t, ix, iz):
    #     sigma_oe = b.oxygen_electrode_residual_thermal_stress[t, ix, iz] * 1e-6
    #     volume = oe_length_x * b.length_y * b.length_z
    #     if pyo.value(sigma_oe) <= 0:
    #         return 0
    #     return 1 - pyo.exp(-(volume/reference_volume_oe) * (sigma_oe / oesigmanot) ** oewebp)

    # @soec.Expression(fs.time, e_xfaces, e_iznodes,
    #                  doc='Weibull statistics for Failure probability in electrolyte layer')
    # def electrolyte_failure_probability(b, t, ix, iz):
    #     sigma_el = b.electrolyte_residual_thermal_stress[t, ix, iz] * 1e-6
    #     volume = e_length_x * b.length_y * b.length_z
    #     if pyo.value(sigma_el) <= 0:
    #         return abs(sigma_el) / fracture_strength_el
    #     return 1 - pyo.exp(-(volume / reference_volume_el) * (sigma_el / elsigmanot) ** elwebp)

    print("after adding thermal stress model, degree_of_freedom = ", dof(blk))
    # creep model
    print()

    fs.time_index = pyo.Param(fs.time, initialize=1, mutable=True, doc="Integer Indexing for Time Domain")
    for index_t, value_t in enumerate(fs.time, 1):
        fs.time_index[value_t] = index_t

    pyo_dae = False

    # define the creep model
    if creep_model:
        soec.sigma_fe_creep = pyo.Var(fs.time, fe_xfaces, fe_iznodes, initialize=0)
        soec.sigma_oe_creep = pyo.Var(fs.time, oe_xfaces, oe_iznodes, initialize=0)
        soec.sigma_e_creep = pyo.Var(fs.time, e_xfaces, e_iznodes, initialize=0)
        soec.creep_strain_fe = pyo.Var(fs.time, fe_xfaces, fe_iznodes, initialize=0)
        soec.creep_strain_oe = pyo.Var(fs.time, oe_xfaces, oe_iznodes, initialize=0)
        soec.creep_strain_e = pyo.Var(fs.time, e_xfaces, e_iznodes, initialize=0)

        if pyo_dae:

            soec.sigma_rate_fe_creep = pyodae.DerivativeVar(soec.sigma_fe_creep, wrt=fs.time)
            soec.sigma_rate_e_creep = pyodae.DerivativeVar(soec.sigma_e_creep, wrt=fs.time)
            soec.sigma_rate_oe_creep = pyodae.DerivativeVar(soec.sigma_oe_creep, wrt=fs.time)
            soec.creep_rate_fe = pyodae.DerivativeVar(soec.creep_strain_fe, wrt=fs.time)
            soec.creep_rate_e = pyodae.DerivativeVar(soec.creep_strain_e, wrt=fs.time)
            soec.creep_rate_oe = pyodae.DerivativeVar(soec.creep_strain_oe, wrt=fs.time)

        else:
            @soec.Expression(fs.time, fe_xfaces, fe_iznodes)
            def sigma_rate_fe_creep(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.sigma_fe_creep[t, ix, iz] - b.sigma_fe_creep[fs.time.at(fs.time_index[t].value - 1), ix, iz]
                        ) / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, e_xfaces, e_iznodes)
            def sigma_rate_e_creep(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.sigma_e_creep[t, ix, iz] - b.sigma_e_creep[fs.time.at(fs.time_index[t].value - 1), ix, iz]
                        ) / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, oe_xfaces, oe_iznodes)
            def sigma_rate_oe_creep(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.sigma_oe_creep[t, ix, iz] - b.sigma_oe_creep[fs.time.at(fs.time_index[t].value - 1), ix, iz]
                        ) / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, fe_xfaces, fe_iznodes)
            def creep_rate_fe(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.creep_strain_fe[t, ix, iz] - b.creep_strain_fe[fs.time.at(fs.time_index[t].value - 1), ix, iz]) \
                    / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, e_xfaces, e_iznodes)
            def creep_rate_e(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.creep_strain_e[t, ix, iz] - b.creep_strain_e[fs.time.at(fs.time_index[t].value - 1), ix, iz]) \
                    / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, oe_xfaces, oe_iznodes)
            def creep_rate_oe(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.creep_strain_oe[t, ix, iz] - b.creep_strain_oe[fs.time.at(fs.time_index[t].value - 1), ix, iz]) \
                    / (t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, fe_xfaces, fe_iznodes)
            def temperature_fe_grad(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.fuel_electrode_temperature_xfaces[t, ix, iz] -
                        b.fuel_electrode_temperature_xfaces[fs.time.at(fs.time_index[t].value - 1), ix, iz]) / (
                        t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, e_xfaces, e_iznodes)
            def temperature_e_grad(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.electrolyte_temperature_xfaces[t, ix, iz] -
                        b.electrolyte_temperature_xfaces[fs.time.at(fs.time_index[t].value - 1), ix, iz]) / (
                        t - fs.time.at(fs.time_index[t].value - 1))

            @soec.Expression(fs.time, oe_xfaces, oe_iznodes)
            def temperature_oe_grad(b, t, ix, iz):
                if t == 0:
                    return pyo.Expression.Skip
                return (b.oxygen_electrode_temperature_xfaces[t, ix, iz] -
                        b.oxygen_electrode_temperature_xfaces[fs.time.at(fs.time_index[t].value - 1), ix, iz]) / (
                        t - fs.time.at(fs.time_index[t].value - 1))

        soec.n_fe = pyo.Var(initialize=1.7)
        soec.a_fe = pyo.Var(initialize=2.6e-11)

        soec.n_e = pyo.Var(initialize=1)
        soec.a_e = pyo.Var(initialize=1.18e-14)

        soec.n_oe = pyo.Var(initialize=1.7)
        soec.a_oe = pyo.Var(initialize=1.27e-12)
        creep_model_parameters = [soec.n_fe, soec.a_fe, soec.n_e, soec.a_e, soec.n_oe, soec.a_oe]
        for param in creep_model_parameters:
            param.fix()

        z1 = pyo.value(fe_length_x)
        z2 = pyo.value(fe_length_x + e_length_x)
        z3 = pyo.value(fe_length_x + e_length_x + oe_length_x)

        # fe_delta_temp = soec.fuel_electrode_temperature_change
        # e_delta_temp = soec.electrolyte_temperature_change
        # oe_delta_temp = soec.oxygen_electrode_temperature_change

        soec.creep_factor_c1 = pyo.Expression(rule=lambda b: 1e-6 * (feE/ (1-fepc) * fe_length_x + \
                                                                  eE / (1-epc) * e_length_x + oeE / (1-oepc) * oe_length_x))

        soec.creep_factor_c2 = pyo.Expression(
            rule=lambda b: 0.5 * 1e-6 * (feE/ (1-fepc) * z1 ** 2 + eE / (1-epc) * (z2 ** 2 - z1 ** 2) + \
                                         oeE / (1-oepc) * (z3 ** 2 - z2 ** 2)))

        soec.creep_factor_c3 = pyo.Expression(
            rule=lambda b: 1 / 3 * 1e-6 * (feE/ (1-fepc) * z1 ** 3 + eE / (1-epc) * (z2 ** 3 - z1 ** 3) + \
                                           oeE / (1-oepc) * (z3 ** 3 - z2 ** 3)))

        # soec.creep_factor_n = pyo.Expression(fe_iznodes, rule=lambda b, iz: 1e-6 * (
        #         feE/ (1-fepc) * fe_length_x * fecte * fe_delta_temp[0, 0, iz] +
        #         eE / (1-epc) * e_length_x * ecte * e_delta_temp[0, 0, iz] +
        #         oeE / (1-oepc) * oe_length_x * oecte * oe_delta_temp[0, 0, iz]))
        #
        # soec.creep_factor_m = pyo.Expression(fe_iznodes, rule=lambda b, iz:
        # 1 / 2 * 1e-6 * (feE/ (1-fepc) * fecte * fe_delta_temp[0, 0, iz] * z1 ** 3 +
        #                 eE / (1-epc) * ecte * e_delta_temp[0, 0, iz] * (z2 ** 3 - z1 ** 3) +
        #                 oeE / (1-oepc) * oecte * oe_delta_temp[0, 0, iz] * (z3 ** 3 - z2 ** 3)))

        @soec.Expression(fs.time, fe_xfaces, fe_iznodes)
        def pseudo_var_fe(b, t, ix, iz):
            return b.sigma_fe_creep[t, ix, iz] * b.sigma_fe_creep[t, ix, iz]

        @soec.Expression(fs.time, oe_xfaces, oe_iznodes)
        def pseudo_var_oe(b, t, ix, iz):
            return b.sigma_oe_creep[t, ix, iz] * b.sigma_oe_creep[t, ix, iz]

        @soec.Expression(fs.time, e_xfaces, e_iznodes)
        def pseudo_var_e(b, t, ix, iz):
            return b.sigma_e_creep[t, ix, iz] * b.sigma_e_creep[t, ix, iz]

        # Equations for estimation of creep rate
        epsilon = 1e-12

        @soec.Constraint(fs.time, fe_xfaces, fe_iznodes, doc='creep rate only caused by creep in fuel electrode')
        def creep_rate_fe_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.creep_rate_fe[t, ix, iz] == (b.sigma_fe_creep[t, ix, iz] / smooth_abs(
                b.sigma_fe_creep[t, ix, iz], eps=epsilon)) * b.a_fe * safe_sqrt(b.pseudo_var_fe[t, ix, iz],
                                                                            eps=epsilon) ** b.n_fe

        @soec.Constraint(fs.time, oe_xfaces, oe_iznodes, doc='creep rate only caused by creep in oxygen electrode')
        def creep_rate_oe_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.creep_rate_oe[t, ix, iz] == (b.sigma_oe_creep[t, ix, iz] / smooth_abs(
                b.sigma_oe_creep[t, ix, iz], eps=epsilon)) * b.a_oe * safe_sqrt(
                b.pseudo_var_oe[t, ix, iz], eps=epsilon) ** b.n_oe

        @soec.Constraint(fs.time, e_xfaces, e_iznodes, doc='creep rate only caused by creep in electrolyte')
        def creep_rate_e_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.creep_rate_e[t, ix, iz] == (b.sigma_e_creep[t, ix, iz] / smooth_abs(
                b.sigma_e_creep[t, ix, iz], eps=epsilon)) * b.a_e * safe_sqrt(
                b.pseudo_var_e[t, ix, iz], eps=epsilon) ** b.n_e

        delta_y_fe = pyo.value(fe_length_x) / (len(fe_xfaces) - 1)
        delta_y_e = pyo.value(e_length_x) / (len(e_xfaces) - 1)
        delta_y_oe = pyo.value(oe_length_x) / (len(oe_xfaces) - 1)

        # Equations to estimate force balance for factor p of fuel electrode
        @soec.Expression(fs.time, fe_iznodes, doc='Equations to estimate force balance for fuel electrode')
        def force_balance_fe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (feE/ (1-fepc) * 1e-6) * sum(delta_y_fe * (b.creep_rate_fe[t, fe_xfaces.at(j - 1), iz] + \
                                                                    b.creep_rate_fe[t, fe_xfaces.at(j), iz]) \
                                                      for j in range(2, len(fe_xfaces) + 1))

        @soec.Expression(fs.time, oe_iznodes, doc='Equations to estimate force balance for oxygen electrode')
        def force_balance_oe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (oeE / (1-oepc) * 1e-6) * sum(delta_y_oe * (b.creep_rate_oe[t, oe_xfaces.at(j - 1), iz] + \
                                                                    b.creep_rate_oe[t, oe_xfaces.at(j), iz]) \
                                                      for j in range(2, len(oe_xfaces) + 1))

        @soec.Expression(fs.time, e_iznodes, doc='Equations to estimate force balance for electrolyte')
        def force_balance_e(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (eE / (1-epc) * 1e-6) * sum(delta_y_e * (b.creep_rate_e[t, e_xfaces.at(j - 1), iz] + \
                                                                  b.creep_rate_e[t, e_xfaces.at(j), iz]) \
                                                     for j in range(2, len(e_xfaces) + 1))

        @soec.Expression(fs.time, fe_iznodes,
                         doc='Equations to estimate force balance due to thermal gradient for fuel electrode')
        def thermal_grad_force_balance_fe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (feE/ (1-fepc) * 1e-6) * fecte * sum(delta_y_fe * (b.temperature_fe_grad[t, ix, iz]) \
                                                              for ix in fe_xfaces)

        @soec.Expression(fs.time, oe_iznodes,
                            doc='Equations to estimate force balance due to thermal gradient for oxygen electrode')
        def thermal_grad_force_balance_oe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (oeE / (1-oepc) * 1e-6) * oecte * sum(delta_y_oe * (b.temperature_oe_grad[t, ix, iz]) \
                                                              for ix in oe_xfaces)

        @soec.Expression(fs.time, e_iznodes,
                            doc='Equations to estimate force balance due to thermal gradient for electrolyte')
        def thermal_grad_force_balance_e(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (eE / (1-epc) * 1e-6) * ecte * sum(delta_y_e * (b.temperature_e_grad[t, ix, iz]) \
                                                              for ix in e_xfaces)

        # Equations to estimate moment balance for factor p of fuel electrode
        @soec.Expression(fs.time, fe_iznodes, doc='Equations to estimate moment balance for fuel electrode')
        def moment_balance_fe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (feE/ (1-fepc) * 1e-6) * sum(delta_y_fe * (
                    b.creep_rate_fe[t, fe_xfaces.at(j - 1), iz] * b.fe_x_tolist[fe_xfaces.at(j - 1)] \
                    + b.creep_rate_fe[t, fe_xfaces.at(j), iz] * b.fe_x_tolist[fe_xfaces.at(j)]) \
                                                      for j in range(2, len(fe_xfaces) + 1))

        @soec.Expression(fs.time, oe_iznodes, doc='Equations to estimate moment balance for oxygen electrode')
        def moment_balance_oe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (oeE / (1-oepc) * 1e-6) * sum(
                delta_y_oe * (b.creep_rate_oe[t, oe_xfaces.at(j - 1), iz] * (
                            fe_length_x + e_length_x + b.oe_x_tolist[oe_xfaces.at(j - 1)]) + \
                              b.creep_rate_oe[t, oe_xfaces.at(j), iz] * (
                                      fe_length_x + e_length_x + b.oe_x_tolist[oe_xfaces.at(j)])) \
                for j in range(2, len(oe_xfaces) + 1))

        @soec.Expression(fs.time, e_iznodes, doc='Equations to estimate moment balance for electrolyte')
        def moment_balance_e(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (eE / (1-epc) * 1e-6) * sum(delta_y_e * (
                    b.creep_rate_e[t, e_xfaces.at(j - 1), iz] * (
                        fe_length_x + b.e_x_tolist[e_xfaces.at(j - 1)]) + \
                    b.creep_rate_e[t, e_xfaces.at(j), iz] * (fe_length_x + b.e_x_tolist[e_xfaces.at(j)])) \
                                                     for j in range(2, len(e_xfaces) + 1))

        @soec.Expression(fs.time, fe_iznodes,
                         doc='Equations to estimate moment balance due to thermal gradient for fuel electrode')
        def thermal_grad_moment_balance_fe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (feE/ (1-fepc) * 1e-6) * fecte * sum(
                delta_y_fe * (b.temperature_fe_grad[t, ix, iz] * b.fe_x_tolist[ix]) for ix in fe_xfaces)

        @soec.Expression(fs.time, oe_iznodes,
                            doc='Equations to estimate moment balance due to thermal gradient for oxygen electrode')
        def thermal_grad_moment_balance_oe(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (oeE / (1-oepc) * 1e-6) * oecte * sum(
                delta_y_oe * (b.temperature_oe_grad[t, ix, iz] * b.oe_x_tolist[ix]) for ix in oe_xfaces)

        @soec.Expression(fs.time, e_iznodes,
                            doc='Equations to estimate moment balance due to thermal gradient for electrolyte')
        def thermal_grad_moment_balance_e(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return 0.5 * (eE / (1-epc) * 1e-6) * ecte * sum(
                delta_y_e * (b.temperature_e_grad[t, ix, iz] * b.e_x_tolist[ix]) for ix in e_xfaces)

        # Equations to estimate force balance for cell
        @soec.Expression(fs.time, fe_iznodes, doc='Equation to estimate force balance for cell')
        def force_balance_cell(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return b.force_balance_fe[t, iz] + b.force_balance_e[t, iz] + b.force_balance_oe[t, iz]

        @soec.Expression(fs.time, fe_iznodes,
                         doc="Equation to estimate force balance due to thermal gradient for cell")
        def thermal_grad_force_balance_cell(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return b.thermal_grad_force_balance_fe[t, iz] + b.thermal_grad_force_balance_e[t, iz] + \
                b.thermal_grad_force_balance_oe[t, iz]

        # Equations to estimate moment balance for cell
        @soec.Expression(fs.time, fe_iznodes, doc='Equations to estimate moment balance for cell')
        def moment_balance_cell(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return b.moment_balance_fe[t, iz] + b.moment_balance_e[t, iz] + b.moment_balance_oe[t, iz]

        @soec.Expression(fs.time, fe_iznodes,
                         doc='Equations to estimate moment balance due to thermal gradient for cell')
        def thermal_grad_moment_balance_cell(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            return b.thermal_grad_moment_balance_fe[t, iz] + b.thermal_grad_moment_balance_e[t, iz] + \
                b.thermal_grad_moment_balance_oe[t, iz]

        @soec.Expression(fs.time, fe_iznodes, doc='Equation to estimate the force balance for cell')
        def uniform_strain_rate(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            uniform_strain_rate_org = (b.creep_factor_c2 * b.moment_balance_cell[t, iz] - \
                    b.creep_factor_c3 * b.force_balance_cell[t, iz]) / (
                        b.creep_factor_c2 ** 2 - b.creep_factor_c1 * b.creep_factor_c3)
            uniform_strain_rate_thermal_grad = (b.creep_factor_c2 * b.thermal_grad_moment_balance_cell[t, iz] - \
                    b.creep_factor_c3 * b.thermal_grad_force_balance_cell[t, iz]) / (
                        b.creep_factor_c2 ** 2 - b.creep_factor_c1 * b.creep_factor_c3)
            return uniform_strain_rate_org + uniform_strain_rate_thermal_grad

        @soec.Expression(fs.time, fe_iznodes, doc='Equation to estimate the curvature change rate for cell')
        def curvature_change_rate(b, t, iz):
            if t == 0:
                return pyo.Expression.Skip
            curvature_change_rate_org = (b.creep_factor_c2 * b.force_balance_cell[t, iz] - \
                    b.creep_factor_c1 * b.moment_balance_cell[t, iz]) / (
                        b.creep_factor_c2 ** 2 - b.creep_factor_c1 * b.creep_factor_c3)
            curvature_change_rate_thermal_grad = (b.creep_factor_c2 * b.thermal_grad_force_balance_cell[t, iz] - \
                    b.creep_factor_c1 * b.thermal_grad_moment_balance_cell[t, iz]) / (
                        b.creep_factor_c2 ** 2 - b.creep_factor_c1 * b.creep_factor_c3)
            return curvature_change_rate_org + curvature_change_rate_thermal_grad

        # Equations to estimate the stress rate
        @soec.Constraint(fs.time, fe_xfaces, fe_iznodes, doc='stress accumulation rate due to creep for fuel electrode')
        def stress_accumulation_rate_fe_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.sigma_rate_fe_creep[t, ix, iz] == \
                (feE/ (1-fepc) * 1e-6) * (
                        (b.uniform_strain_rate[t, iz] + b.curvature_change_rate[t, iz] * b.fe_x_tolist[ix] -
                         fecte * b.temperature_fe_grad[t, ix, iz]) - b.creep_rate_fe[t, ix, iz])

        @soec.Constraint(fs.time, oe_xfaces, oe_iznodes,
                         doc='stress accumulation rate due to creep for oxygen electrode')
        def stress_accumulation_rate_oe_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.sigma_rate_oe_creep[t, ix, iz] == \
                (oeE / (1-oepc) * 1e-6) * (
                        (b.uniform_strain_rate[t, iz] + b.curvature_change_rate[t, iz] * (
                                fe_length_x + e_length_x + b.oe_x_tolist[ix]) -
                         oecte * b.temperature_oe_grad[t, ix, iz]
                         ) - b.creep_rate_oe[t, ix, iz])

        @soec.Constraint(fs.time, e_xfaces, e_iznodes, doc='stress accumulation rate due to creep for electrolyte')
        def stress_accumulation_rate_e_eqn(b, t, ix, iz):
            if t == 0:
                return pyo.Constraint.Skip
            return b.sigma_rate_e_creep[t, ix, iz] == \
                (eE / (1-epc) * 1e-6) * (
                        (b.uniform_strain_rate[t, iz] + b.curvature_change_rate[t, iz] * (
                                    fe_length_x + b.e_x_tolist[ix]) -
                         ecte * b.temperature_e_grad[t, ix, iz]
                         ) - b.creep_rate_e[t, ix, iz])

        eps = 1e-10
        @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
                        doc='Weibull statistics for Failure probability in fuel electrode layer')
        def fuel_electrode_failure_probability(b, t, ix, iz):
            sigma_fe = b.sigma_fe_creep[t, ix, iz]
            volume = fe_length_x * b.length_y * b.length_z
            return Expr_if(pyo.value(1e6 * sigma_fe + eps) >= 0,
                           1 - pyo.exp(-(volume/reference_volume_fe) * (safe_sqrt(sigma_fe ** 2) / fesigmanot) ** fewebp), eps)

        @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
                            doc='Weibull statistics for Failure probability in oxygen electrode layer')
        def oxygen_electrode_failure_probability(b, t, ix, iz):
            sigma_oe = b.sigma_oe_creep[t, ix, iz]
            volume = oe_length_x * b.length_y * b.length_z
            return Expr_if(pyo.value(1e6 * sigma_oe + eps) >= 0,
                            1 - pyo.exp(-(volume/reference_volume_oe) * (safe_sqrt(sigma_oe ** 2) / oesigmanot) ** oewebp), eps)

        @soec.Expression(fs.time, e_xfaces, e_iznodes,
                        doc='Weibull statistics for Failure probability in electrolyte layer')
        def electrolyte_failure_probability(b, t, ix, iz):
            sigma_el = b.sigma_e_creep[t, ix, iz]
            volume = e_length_x * b.length_y * b.length_z
            return Expr_if(pyo.value(1e6 * sigma_el + eps) >= 0,
                            1 - pyo.exp(-(volume/reference_volume_el) * (safe_sqrt(sigma_el ** 2) / elsigmanot) ** elwebp), eps)
        
        # soec.cell_characteristic_strain = pyo.Var(initialize=0.02)
        # soec.cell_characteristic_strain.fix()

        # eps = 1e-10
        # @soec.Expression(fs.time, fe_xfaces, fe_iznodes,
        #                 doc='Weibull statistics for Failure probability in fuel electrode layer')
        # def fuel_electrode_failure_probability(b, t, ix, iz):
        #     creep_fe = b.creep_strain_fe[t, ix, iz]
        #     volume = fe_length_x * b.length_y * b.length_z
        #     return Expr_if(pyo.value(creep_fe + eps) >= 0,
        #                    1 - pyo.exp(-(volume/reference_volume_fe) * (creep_fe / b.cell_characteristic_strain) ** fewebp), eps)

        # @soec.Expression(fs.time, oe_xfaces, oe_iznodes,
        #                     doc='Weibull statistics for Failure probability in oxygen electrode layer')
        # def oxygen_electrode_failure_probability(b, t, ix, iz):
        #     creep_oe = b.creep_strain_oe[t, ix, iz]
        #     volume = oe_length_x * b.length_y * b.length_z
        #     return Expr_if(pyo.value(creep_oe + eps) >= 0,
        #                     1 - pyo.exp(-(volume/reference_volume_oe) * (creep_oe / b.cell_characteristic_strain) ** oewebp), eps)

        # @soec.Expression(fs.time, e_xfaces, e_iznodes,
        #                 doc='Weibull statistics for Failure probability in electrolyte layer')
        # def electrolyte_failure_probability(b, t, ix, iz):
        #     creep_e = b.creep_strain_e[t, ix, iz]
        #     volume = e_length_x * b.length_y * b.length_z
        #     return Expr_if(pyo.value(creep_e + eps) >= 0,
        #                     1 - pyo.exp(-(volume/reference_volume_el) * (creep_e / b.cell_characteristic_strain) ** elwebp), eps)
        
        # Applying scaling method
        for t in fs.time:
            if t == 0:
                continue
            else:
                for ix in fe_xfaces:
                    for iz in fe_iznodes:
                        iscale.set_scaling_factor(soec.sigma_fe_creep[t, ix, iz], 1e-6)
                        iscale.set_scaling_factor(soec.creep_strain_fe[t, ix, iz], 1e-6)
                        iscale.constraint_scaling_transform(soec.creep_rate_fe_eqn[t, ix, iz], 1e4)
                    if pyo_dae:
                        iscale.set_scaling_factor(soec.creep_rate_fe[t, ix, iz], 1e4)
                for ix in e_xfaces:
                    for iz in e_iznodes:
                        iscale.set_scaling_factor(soec.sigma_e_creep[t, ix, iz], 1e-6)
                        iscale.set_scaling_factor(soec.creep_strain_e[t, ix, iz], 1e-6)
                        iscale.constraint_scaling_transform(soec.creep_rate_e_eqn[t, ix, iz], 1e5)
                    if pyo_dae:
                        iscale.set_scaling_factor(soec.creep_rate_e[t, ix, iz], 1e5)
                for ix in oe_xfaces:
                    for iz in oe_iznodes:
                        iscale.set_scaling_factor(soec.sigma_oe_creep[t, ix, iz], 1e-6)
                        iscale.set_scaling_factor(soec.creep_strain_oe[t, ix, iz], 1e-6)
                        iscale.constraint_scaling_transform(soec.creep_rate_oe_eqn[t, ix, iz], 1e4)
                    if pyo_dae:
                        iscale.set_scaling_factor(soec.creep_rate_oe[t, ix, iz], 1e4)

        # Fixing the initial values of the stress
        soec.sigma_fe_creep[0, :, :].fix()
        soec.sigma_e_creep[0, :, :].fix()
        soec.sigma_oe_creep[0, :, :].fix()
        soec.creep_strain_fe[0, :, :].fix()
        soec.creep_strain_e[0, :, :].fix()
        soec.creep_strain_oe[0, :, :].fix()

        # reference: 'Molla et al., 2018'
        a_fe = 8.68e-8 / 3600
        a_e = 6.44e-11 / 3600
        a_oe = 4.57e-9 / 3600
        soec.a_fe.set_value(a_fe)
        soec.a_e.set_value(a_e)
        soec.a_oe.set_value(a_oe)
        ##########
        import pickle
        # folder = "health_submodels/optimization/creep_model"
        creep_relaxation_data = pickle.load(open("stress_creep_submodels/relaxation_creep_results.pkl", "rb"))

        x_list_fe, x_list_e, x_list_oe, eps_creep_fe, eps_creep_e, eps_creep_oe = creep_relaxation_data

        if stress_relaxation:
            # fix initial creep strain and stress for each layer
            for ix in fe_xfaces:
                for iz in fe_iznodes:
                    soec.creep_strain_fe[0, ix, iz].fix(eps_creep_fe[x_list_fe[ix]])
                    soec.sigma_fe_creep[0, ix, iz].fix(0)

            for ix in e_xfaces:
                for iz in e_iznodes:
                    soec.creep_strain_e[0, ix, iz].fix(eps_creep_e[soec.fe_x_tolist[-1] + soec.e_x_tolist[ix]])
                    soec.sigma_e_creep[0, ix, iz].fix(0)

            for ix in oe_xfaces:
                for iz in oe_iznodes:
                    soec.creep_strain_oe[0, ix, iz].fix(eps_creep_oe[soec.fe_x_tolist[-1] + soec.e_x_tolist[-1] +
                                                                        soec.oe_x_tolist[ix]])
                    soec.sigma_oe_creep[0, ix, iz].fix(0)

def save_thermal_stress_results(blk, file_name):
    import json
    t0 = blk.fs.time.first()
    tf = blk.fs.time.last()
    soec = blk.fs.soc_module.solid_oxide_cell
    fe_fs = soec.fuel_electrode
    oe_fs = soec.oxygen_electrode
    e_fs = soec.electrolyte
    oe_sigma = soec.oxygen_electrode_residual_thermal_stress
    fe_sigma = soec.fuel_electrode_residual_thermal_stress
    e_sigma = soec.electrolyte_residual_thermal_stress
    # set alias with variables
    fe_xfaces = soec.fe_xfaces
    e_xfaces = soec.e_xfaces
    oe_xfaces = soec.oe_xfaces
    fe_weibull_failure = soec.fuel_electrode_failure_probability
    oe_weibull_failure = soec.oxygen_electrode_failure_probability
    el_weibull_failure = soec.electrolyte_failure_probability
    data = {
        'z_list': [pyo.value(soec.fuel_channel.zfaces.at(izf) * soec.length_z
                             ) for izf in soec.fuel_channel.izfaces],

        'fe_residual_thermal_stress_t0_x1, MPa': [pyo.value(fe_sigma[t0, fe_xfaces.last(), iz]
        ) * 1e-6 for iz in fe_fs.iznodes],
        'fe_residual_thermal_stress_tf_x1, MPa': [pyo.value(fe_sigma[tf, fe_xfaces.last(), iz]
        ) * 1e-6 for iz in fe_fs.iznodes],
        'oe_residual_thermal_stress_t0, MPa': [pyo.value(oe_sigma[t0, 0, iz]
        ) * 1e-6 for iz in oe_fs.iznodes],
        'oe_residual_thermal_stress_tf, MPa': [pyo.value(oe_sigma[tf, 0, iz]
        ) * 1e-6 for iz in oe_fs.iznodes],
        'e_residual_thermal_stress_t0, MPa': [pyo.value(e_sigma[t0, 0, iz]
                                                            ) * 1e-6 for iz in oe_fs.iznodes],
        'e_residual_thermal_stress_tf, MPa': [pyo.value(e_sigma[tf, 0, iz]
                                                           ) * 1e-6 for iz in oe_fs.iznodes],
        'fe_weibull_failure_t0_x1': [pyo.value(fe_weibull_failure[t0, fe_xfaces.last(), iz]
                                               ) for iz in fe_fs.iznodes],
        'fe_weibull_failure_tf_x1': [pyo.value(fe_weibull_failure[tf, fe_xfaces.last(), iz]
                                               ) for iz in fe_fs.iznodes],
        'oe_weibull_failure_t0': [pyo.value(oe_weibull_failure[t0, 0, iz]
                                            ) for iz in oe_fs.iznodes],
        'oe_weibull_failure_tf': [pyo.value(oe_weibull_failure[tf, 0, iz]
                                            ) for iz in oe_fs.iznodes],
        'el_weibull_failure_t0': [pyo.value(el_weibull_failure[t0, 0, iz]
                                            ) for iz in e_fs.iznodes],
        'el_weibull_failure_tf': [pyo.value(el_weibull_failure[tf, 0, iz]
                                            ) for iz in e_fs.iznodes],
    }

    with open('thermal_stress_data_' + file_name + '.json', 'w') as fp:
        json.dump(data, fp, indent=4)

    print('Done! Save data for thermal stress results')

# save temperature and thermal stress profiles using full discretization
def _save_temperature_stress_flowsheet(blk, filename):
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    soec = blk.fs.soc_module.solid_oxide_cell
    # list of time points
    time_set = sorted(blk.fs.time)
    # last point
    tf = blk.fs.time.last()


    data = {
        "time_list": [],
        "H2_production": [],
        "cell potential": [],
        "feed_heater_duty": [],
        "sweep_heater_duty": [],
        "cell_temperature_znode_average": [],
        "total_electric_power": [],
        "Cell_average_temperature": [],
        "cell_inlet_temperature": [],
        "cell_outlet_temperature": [],
        "oxygen inlet temperature": [],
        "fuel inlet temperature": [],
        "fuel_electrode_inlet_temperature": [],
        "fuel_electrode_outlet_temperature": [],
        "oxygen_electrode_inlet_temperature": [],
        "oxygen_electrode_outlet_temperature": [],
        "electrolyte_inlet_temperature": [],
        "electrolyte_outlet_temperature": [],
    # ###############################################################################
        "electrolyte_first_node_temperature": [],
        "electrolyte_second_node_temperature": [],
        "electrolyte_third_node_temperature": [],
        "electrolyte_fourth_node_temperature": [],
        "electrolyte_fifth_node_temperature": [],
        "electrolyte_sixth_node_temperature": [],
        "electrolyte_seventh_node_temperature": [],
        "electrolyte_eighth_node_temperature": [],
        "electrolyte_ninth_node_temperature": [],
        "electrolyte_tenth_node_temperature": [],

    # ###############################################################################
        "oxygen_electrode_first_node_temperature": [],
        "oxygen_electrode_second_node_temperature": [],
        "oxygen_electrode_third_node_temperature": [],
        "oxygen_electrode_fourth_node_temperature": [],
        "oxygen_electrode_fifth_node_temperature": [],
        "oxygen_electrode_sixth_node_temperature": [],
        "oxygen_electrode_seventh_node_temperature": [],
        "oxygen_electrode_eighth_node_temperature": [],
        "oxygen_electrode_ninth_node_temperature": [],
        "oxygen_electrode_tenth_node_temperature": [],

    # ###############################################################################
        "fuel_electrode_first_node_temperature": [],
        "fuel_electrode_second_node_temperature": [],
        "fuel_electrode_third_node_temperature": [],
        "fuel_electrode_fourth_node_temperature": [],
        "fuel_electrode_fifth_node_temperature": [],
        "fuel_electrode_sixth_node_temperature": [],
        "fuel_electrode_seventh_node_temperature": [],
        "fuel_electrode_eighth_node_temperature": [],
        "fuel_electrode_ninth_node_temperature": [],
        "fuel_electrode_tenth_node_temperature": [],

        "fuel_electrode_temperature_deviation_x_first_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_second_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_third_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_fourth_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_fifth_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_sixth_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_seventh_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_eighth_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_ninth_node_temperature": [],
        "fuel_electrode_temperature_deviation_x_tenth_node_temperature": [],

    # ###############################################################################
        "fuel_electrode_inletfaces_first_node_temperature": [],
        "fuel_electrode_inletfaces_second_node_temperature": [],
        "fuel_electrode_inletfaces_third_node_temperature": [],
        "fuel_electrode_inletfaces_fourth_node_temperature": [],
        "fuel_electrode_inletfaces_fifth_node_temperature": [],
        "fuel_electrode_inletfaces_sixth_node_temperature": [],
        "fuel_electrode_inletfaces_seventh_node_temperature": [],
        "fuel_electrode_inletfaces_eighth_node_temperature": [],
        "fuel_electrode_inletfaces_ninth_node_temperature": [],
        "fuel_electrode_inletfaces_tenth_node_temperature": [],

    # ###############################################################################
        "electrolyte_thermal_stress": [],
        "oxygen_electrode_thermal_stress": [],

    # ###############################################################################

        "fuel_electrode_faces_first_node_thermal_stress": [],
        "fuel_electrode_faces_second_node_thermal_stress": [],
        "fuel_electrode_faces_third_node_thermal_stress": [],
        "fuel_electrode_faces_fourth_node_thermal_stress": [],
        "fuel_electrode_faces_fifth_node_thermal_stress": [],
        "fuel_electrode_faces_sixth_node_thermal_stress": [],
        "fuel_electrode_faces_seventh_node_thermal_stress": [],
        "fuel_electrode_faces_eighth_node_thermal_stress": [],
        "fuel_electrode_faces_ninth_node_thermal_stress": [],
        "fuel_electrode_faces_tenth_node_thermal_stress": [],

        "fuel_electrode_second_faces_first_node_thermal_stress": [],
        "fuel_electrode_second_faces_second_node_thermal_stress": [],
        "fuel_electrode_second_faces_third_node_thermal_stress": [],
        "fuel_electrode_second_faces_fourth_node_thermal_stress": [],
        "fuel_electrode_second_faces_fifth_node_thermal_stress": [],
        "fuel_electrode_second_faces_sixth_node_thermal_stress": [],
        "fuel_electrode_second_faces_seventh_node_thermal_stress": [],
        "fuel_electrode_second_faces_eighth_node_thermal_stress": [],
        "fuel_electrode_second_faces_ninth_node_thermal_stress": [],
        "fuel_electrode_second_faces_tenth_node_thermal_stress": [],

    # ###############################################################################
        "fuel_electrode_faces_first_node_failure_probability": [],
        "fuel_electrode_faces_second_node_failure_probability": [],
        "fuel_electrode_faces_third_node_failure_probability": [],
        "fuel_electrode_faces_fourth_node_failure_probability": [],
        "fuel_electrode_faces_fifth_node_failure_probability": [],
        "fuel_electrode_faces_sixth_node_failure_probability": [],
        "fuel_electrode_faces_seventh_node_failure_probability": [],
        "fuel_electrode_faces_eighth_node_failure_probability": [],
        "fuel_electrode_faces_ninth_node_failure_probability": [],
        "fuel_electrode_faces_tenth_node_failure_probability": [],

        "fuel_electrode_second_faces_first_node_failure_probability": [],
        "fuel_electrode_second_faces_second_node_failure_probability": [],
        "fuel_electrode_second_faces_third_node_failure_probability": [],
        "fuel_electrode_second_faces_fourth_node_failure_probability": [],
        "fuel_electrode_second_faces_fifth_node_failure_probability": [],
        "fuel_electrode_second_faces_sixth_node_failure_probability": [],
        "fuel_electrode_second_faces_seventh_node_failure_probability": [],
        "fuel_electrode_second_faces_eighth_node_failure_probability": [],
        "fuel_electrode_second_faces_ninth_node_failure_probability": [],
        "fuel_electrode_second_faces_tenth_node_failure_probability": [],

    # ###############################################################################
        "effiency_hhv": [],
        'cell_integral_efficiency': [],
        "electrolyte_failure_probability": [],

    # ###############################################################################
        # save creep model results
        "strain_creep_fe_free": [],
        "strain_creep_fe_e": [],
        "strain_creep_e": [],
        "strain_creep_oe_free": [],
        "strain_creep_oe_e": [],
        "sigma_creep_fe_free": [],
        "sigma_creep_fe_e": [],
        "sigma_creep_e": [],
        "sigma_creep_oe_free": [],
        "sigma_creep_oe_e": [],
        "creep_rate_fe_free": [],
        "creep_rate_fe_e": [],
        "creep_rate_oe_free": [],
        "creep_rate_e": [],

    }

    # Loop through your time_set and append data to each list in the dictionary
    for t in time_set:
        data["time_list"].append(t)
        data["H2_production"].append(np.array(pyo.value(blk.fs.h2_mass_production[t])))
        data["cell potential"].append(np.array(pyo.value(soec.potential[t])))
        data["feed_heater_duty"].append(np.array(pyo.value(blk.fs.feed_heater.electric_heat_duty[t])))
        data["sweep_heater_duty"].append(np.array(pyo.value(blk.fs.sweep_heater.electric_heat_duty[t])))
        data["cell_temperature_znode_average"].append(np.array(
            [np.mean([pyo.value(soec.temperature_z[t, iz]) for iz in soec.iznodes])]))
        data["total_electric_power"].append(np.array(pyo.value(blk.fs.total_electric_power[t])))
        data["Cell_average_temperature"].append(np.array(
            [np.mean([pyo.value(soec.temperature_z[t, iz]) for iz in soec.iznodes])]))
        data["cell_inlet_temperature"].append(np.array([pyo.value(soec.temperature_z[t, soec.iznodes.first()])]))
        data["cell_outlet_temperature"].append(np.array([pyo.value(soec.temperature_z[t, soec.iznodes.last()])]))
        data["oxygen inlet temperature"].append(np.array([pyo.value(soec.oxygen_inlet.temperature[t])]))
        data["fuel inlet temperature"].append(np.array([pyo.value(soec.fuel_inlet.temperature[t])]))
        data["fuel_electrode_inlet_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first()])]))
        data["fuel_electrode_outlet_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.last()])]))
        data["oxygen_electrode_inlet_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first()])]))
        data["oxygen_electrode_outlet_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.last()])]))
        data["electrolyte_inlet_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first()])]))
        data["electrolyte_outlet_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.last()])]))
        # ###############################################################################
        data["electrolyte_first_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first()])]))
        data["electrolyte_second_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 1])]))
        data["electrolyte_third_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 2])]))
        data["electrolyte_fourth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 3])]))
        data["electrolyte_fifth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 4])]))
        data["electrolyte_sixth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 5])]))
        data["electrolyte_seventh_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 6])]))
        data["electrolyte_eighth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 7])]))
        data["electrolyte_ninth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 8])]))
        data["electrolyte_tenth_node_temperature"].append(np.array(
            [pyo.value(soec.electrolyte.temperature_z[t, soec.iznodes.first() + 9])]))

        # ###############################################################################
        data["oxygen_electrode_first_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first()])]))
        data["oxygen_electrode_second_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 1])]))
        data["oxygen_electrode_third_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 2])]))
        data["oxygen_electrode_fourth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 3])]))
        data["oxygen_electrode_fifth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 4])]))
        data["oxygen_electrode_sixth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 5])]))
        data["oxygen_electrode_seventh_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 6])]))
        data["oxygen_electrode_eighth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 7])]))
        data["oxygen_electrode_ninth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 8])]))
        data["oxygen_electrode_tenth_node_temperature"].append(np.array(
            [pyo.value(soec.oxygen_electrode.temperature_z[t, soec.iznodes.first() + 9])]))

        # ###############################################################################
        data["fuel_electrode_first_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first()])]))
        data["fuel_electrode_second_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 1])]))
        data["fuel_electrode_third_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 2])]))
        data["fuel_electrode_fourth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 3])]))
        data["fuel_electrode_fifth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 4])]))
        data["fuel_electrode_sixth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 5])]))
        data["fuel_electrode_seventh_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 6])]))
        data["fuel_electrode_eighth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 7])]))
        data["fuel_electrode_ninth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 8])]))
        data["fuel_electrode_tenth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 9])]))

        data["fuel_electrode_temperature_deviation_x_first_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first()])]))
        data["fuel_electrode_temperature_deviation_x_second_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 1])]))
        data["fuel_electrode_temperature_deviation_x_third_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 2])]))
        data["fuel_electrode_temperature_deviation_x_fourth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 3])]))
        data["fuel_electrode_temperature_deviation_x_fifth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 4])]))
        data["fuel_electrode_temperature_deviation_x_sixth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 5])]))
        data["fuel_electrode_temperature_deviation_x_seventh_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 6])]))
        data["fuel_electrode_temperature_deviation_x_eighth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 7])]))
        data["fuel_electrode_temperature_deviation_x_ninth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 8])]))
        data["fuel_electrode_temperature_deviation_x_tenth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 9])]))

        # ###############################################################################
        data["fuel_electrode_inletfaces_first_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first()])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first()])]))
        data["fuel_electrode_inletfaces_second_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 1])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 1])]))
        data["fuel_electrode_inletfaces_third_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 2])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 2])]))
        data["fuel_electrode_inletfaces_fourth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 3])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 3])]))
        data["fuel_electrode_inletfaces_fifth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 4])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 4])]))
        data["fuel_electrode_inletfaces_sixth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 5])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 5])]))
        data["fuel_electrode_inletfaces_seventh_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 6])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 6])]))
        data["fuel_electrode_inletfaces_eighth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 7])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 7])]))
        data["fuel_electrode_inletfaces_ninth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 8])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 8])]))
        data["fuel_electrode_inletfaces_tenth_node_temperature"].append(np.array(
            [pyo.value(soec.fuel_electrode.temperature_z[t, soec.iznodes.first() + 9])]) + \
            np.array([pyo.value(soec.fuel_electrode.temperature_deviation_x[t, 1, soec.iznodes.first() + 9])]))

        # ###############################################################################
        data["electrolyte_thermal_stress"].append(np.array(
            [pyo.value(soec.electrolyte_residual_thermal_stress[t, 0, iz]) * 1e-6 for iz in soec.iznodes]))
        data["oxygen_electrode_thermal_stress"].append(np.array(
            [pyo.value(soec.oxygen_electrode_residual_thermal_stress[t, 0, iz]) * 1e-6 for iz in soec.iznodes]))

        # ###############################################################################
        # set alias with variables
        fe_xfaces = soec.fe_xfaces
        e_xfaces = soec.e_xfaces
        oe_xfaces = soec.oe_xfaces

        data["fuel_electrode_faces_first_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first()]) * 1e-6]))
        data["fuel_electrode_faces_second_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 1]) * 1e-6]))
        data["fuel_electrode_faces_third_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 2]) * 1e-6]))
        data["fuel_electrode_faces_fourth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 3]) * 1e-6]))
        data["fuel_electrode_faces_fifth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 4]) * 1e-6]))
        data["fuel_electrode_faces_sixth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 5]) * 1e-6]))
        data["fuel_electrode_faces_seventh_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 6]) * 1e-6]))
        data["fuel_electrode_faces_eighth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 7]) * 1e-6]))
        data["fuel_electrode_faces_ninth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 8]) * 1e-6]))
        data["fuel_electrode_faces_tenth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, 0, soec.iznodes.first() + 9]) * 1e-6]))

        # ###############################################################################
        data["fuel_electrode_second_faces_first_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first()]) * 1e-6]))
        data["fuel_electrode_second_faces_second_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 1]) * 1e-6]))
        data["fuel_electrode_second_faces_third_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 2]) * 1e-6]))
        data["fuel_electrode_second_faces_fourth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 3]) * 1e-6]))
        data["fuel_electrode_second_faces_fifth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 4]) * 1e-6]))
        data["fuel_electrode_second_faces_sixth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 5]) * 1e-6]))
        data["fuel_electrode_second_faces_seventh_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 6]) * 1e-6]))
        data["fuel_electrode_second_faces_eighth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 7]) * 1e-6]))
        data["fuel_electrode_second_faces_ninth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 8]) * 1e-6]))
        data["fuel_electrode_second_faces_tenth_node_thermal_stress"].append(np.array(
            [pyo.value(soec.fuel_electrode_residual_thermal_stress[t, fe_xfaces.last(), soec.iznodes.first() + 9]) * 1e-6]))

        # ###############################################################################
        data["fuel_electrode_faces_first_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first()])))
        data["fuel_electrode_faces_second_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 1])))
        data["fuel_electrode_faces_third_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 2])))
        data["fuel_electrode_faces_fourth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 3])))
        data["fuel_electrode_faces_fifth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 4])))
        data["fuel_electrode_faces_sixth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 5])))
        data["fuel_electrode_faces_seventh_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 6])))
        data["fuel_electrode_faces_eighth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 7])))
        data["fuel_electrode_faces_ninth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 8])))
        data["fuel_electrode_faces_tenth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, 0, soec.iznodes.first() + 9])))
        # ###############################################################################
        data["fuel_electrode_second_faces_first_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first()])))
        data["fuel_electrode_second_faces_second_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 1])))
        data["fuel_electrode_second_faces_third_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 2])))
        data["fuel_electrode_second_faces_fourth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 3])))
        data["fuel_electrode_second_faces_fifth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 4])))
        data["fuel_electrode_second_faces_sixth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 5])))
        data["fuel_electrode_second_faces_seventh_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 6])))
        data["fuel_electrode_second_faces_eighth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 7])))
        data["fuel_electrode_second_faces_ninth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 8])))
        data["fuel_electrode_second_faces_tenth_node_failure_probability"].append(np.array(pyo.value(
            soec.fuel_electrode_failure_probability[t, fe_xfaces.last(), soec.iznodes.first() + 9])))

        # ###############################################################################
        data["effiency_hhv"].append(np.array(pyo.value(blk.fs.efficiency_hhv[t])))
        data['cell_integral_efficiency'].append(pyo.value(blk.fs.integral_efficiency))
        data["electrolyte_failure_probability"].append(
            np.array([pyo.value(soec.electrolyte_failure_probability[t, 0, iz]) for iz in soec.iznodes])
        )

        # ###############################################################################
        data["strain_creep_fe_free"].append(np.array(
            [pyo.value(soec.creep_strain_fe[t, 0, iz]) for iz in soec.iznodes]))
        data["strain_creep_fe_e"].append(np.array(
            [pyo.value(soec.creep_strain_fe[t, fe_xfaces.last(), iz]) for iz in soec.iznodes]))
        data["strain_creep_e"].append(np.array(
            [pyo.value(soec.creep_strain_e[t, 0, iz]) for iz in soec.iznodes]))
        data["strain_creep_oe_free"].append(np.array(
            [pyo.value(soec.creep_strain_oe[t, 0, iz]) for iz in soec.iznodes]))
        data["strain_creep_oe_e"].append(np.array(
            [pyo.value(soec.creep_strain_oe[t, oe_xfaces.last(), iz]) for iz in soec.iznodes]))
        data["sigma_creep_fe_free"].append(np.array(
            [pyo.value(soec.sigma_fe_creep[t, 0, iz]) for iz in soec.iznodes]))
        data["sigma_creep_fe_e"].append(np.array(
            [pyo.value(soec.sigma_fe_creep[t, fe_xfaces.last(), iz]) for iz in soec.iznodes]))
        data["sigma_creep_e"].append(np.array(
            [pyo.value(soec.sigma_e_creep[t, 0, iz]) for iz in soec.iznodes]))
        data["sigma_creep_oe_free"].append(np.array(
            [pyo.value(soec.sigma_oe_creep[t, 0, iz]) for iz in soec.iznodes]))
        data["sigma_creep_oe_e"].append(np.array(
            [pyo.value(soec.sigma_oe_creep[t, oe_xfaces.last(), iz]) for iz in soec.iznodes]))
        if t == 0:
            continue
        else:
            data["creep_rate_fe_free"].append(np.array(
                [pyo.value(soec.creep_rate_fe[t, 0, iz]) for iz in soec.iznodes]))
            data["creep_rate_fe_e"].append(np.array(
                [pyo.value(soec.creep_rate_fe[t, fe_xfaces.last(), iz]) for iz in soec.iznodes]))
            data["creep_rate_oe_free"].append(np.array(
                [pyo.value(soec.creep_rate_oe[t, 0, iz]) for iz in soec.iznodes]))
            data["creep_rate_e"].append(np.array(
                [pyo.value(soec.creep_rate_e[t, 0, iz]) for iz in soec.iznodes]))

    # Save the dictionary to a json file
    with open(filename + '.json', 'w') as fp:
        json.dump(data, fp, cls=NumpyEncoder, indent=4)

    print('Finishing saving temperature & stress results')