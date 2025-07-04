##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
1-D Cross Flow Heat Exchanger Model With Wall Temperatures

Discretization based on tube rows
"""
from __future__ import division

# Import Python libraries
import math

# Import Pyomo libraries
from pyomo.environ import (
    SolverFactory,
    Var,
    Param,
    Constraint,
    value,
    TerminationCondition,
    exp,
    sqrt,
    log,
    sin,
    cos,
    SolverStatus,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES cores
from idaes.core import (
    ControlVolume1DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    FlowDirection,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.constants import Constants as const
import idaes.core.util.scaling as iscale
from pyomo.dae import DerivativeVar
from pyomo.environ import units as pyunits
from idaes.core.solvers import get_solver
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
import idaes.logger as idaeslog
from idaes.core.util.tables import create_stream_table_dataframe

__author__ = "Jinliang Ma, Douglas Allan"

# Set up logger
_log = idaeslog.getLogger(__name__)


def _make_geometry_common(blk, shell_units):
    # Number of tube columns in the cross section plane perpendicular to shell side fluid flow (y direction)
    blk.ncol_tube = Var(
        initialize=10.0, doc="number of tube columns", units=pyunits.dimensionless
    )

    # Number of segments of tube bundles
    blk.nseg_tube = Var(
        initialize=10.0, doc="number of tube segments", units=pyunits.dimensionless
    )

    # Number of inlet tube rows
    blk.nrow_inlet = Var(
        initialize=1, doc="number of inlet tube rows", units=pyunits.dimensionless
    )

    # Inner diameter of tubes
    blk.di_tube = Var(
        initialize=0.05, doc="inner diameter of tube", units=shell_units["length"]
    )

    # Thickness of tube
    blk.thickness_tube = Var(
        initialize=0.005, doc="tube thickness", units=shell_units["length"]
    )

    # Pitch of tubes between two neighboring columns (in y direction). Always greater than tube outside diameter
    blk.pitch_y = Var(
        initialize=0.1,
        doc="pitch between two neighboring columns",
        units=shell_units["length"],
    )

    # Pitch of tubes between two neighboring rows (in x direction). Always greater than tube outside diameter
    blk.pitch_x = Var(
        initialize=0.1,
        doc="pitch between two neighboring rows",
        units=shell_units["length"],
    )

    # Length of tube per segment in z direction
    blk.length_tube_seg = Var(
        initialize=1.0, doc="length of tube per segment", units=shell_units["length"]
    )

    # Minimum cross-sectional area on shell side
    blk.area_flow_shell_min = Var(
        initialize=1.0, doc="minimum flow area on shell side", units=shell_units["area"]
    )

    # total number of tube rows
    @blk.Expression(doc="total number of tube rows")
    def nrow_tube(b):
        return b.nseg_tube * b.nrow_inlet

    # Tube outside diameter
    @blk.Expression(doc="outside diameter of tube")
    def do_tube(b):
        return b.di_tube + b.thickness_tube * 2.0

    # Mean beam length for radiation
    if blk.config.has_radiation:

        @blk.Expression(doc="mean bean length")
        def mbl(b):
            return 3.6 * (
                b.pitch_x * b.pitch_y / const.pi / b.do_tube - b.do_tube / 4.0
            )

        # Mean beam length for radiation divided by sqrt(2)
        @blk.Expression(doc="sqrt(1/2) of mean bean length")
        def mbl_div2(b):
            return b.mbl / sqrt(2.0)

        # Mean beam length for radiation multiplied by sqrt(2)
        @blk.Expression(doc="sqrt(2) of mean bean length")
        def mbl_mul2(b):
            return b.mbl * sqrt(2.0)

    # Ratio of pitch_x/do_tube
    @blk.Expression(doc="ratio of pitch in x direction to tube outside diameter")
    def pitch_x_to_do(b):
        return b.pitch_x / b.do_tube

    # Ratio of pitch_y/do_tube
    @blk.Expression(doc="ratio of pitch in y direction to tube outside diameter")
    def pitch_y_to_do(b):
        return b.pitch_y / b.do_tube

    # Total cross-sectional area of tube metal per segment
    @blk.Expression(doc="total cross section area of tube metal per segment")
    def area_wall_seg(b):
        return (
            0.25
            * const.pi
            * (b.do_tube**2 - b.di_tube**2)
            * b.ncol_tube
            * b.nrow_inlet
        )

    # Length of shell side flow
    @blk.Constraint(doc="Length of shell side flow")
    def length_flow_shell_eqn(b):
        return b.length_flow_shell == b.nrow_tube * b.pitch_x

    # Average flow area on shell side
    @blk.Constraint(doc="Average cross section area of shell side flow")
    def area_flow_shell_eqn(b):
        return (
            b.length_flow_shell * b.area_flow_shell
            == b.length_tube_seg * b.length_flow_shell * b.pitch_y * b.ncol_tube
            - b.ncol_tube
            * b.nrow_tube
            * 0.25
            * const.pi
            * b.do_tube**2
            * b.length_tube_seg
        )

    # Minimum flow area on shell side
    @blk.Constraint(doc="Minimum flow area on shell side")
    def area_flow_shell_min_eqn(b):
        return (
            b.area_flow_shell_min
            == b.length_tube_seg * (b.pitch_y - b.do_tube) * b.ncol_tube
        )

    @blk.Expression()
    def total_heat_transfer_area(b):
        return (
            const.pi
            * b.do_tube
            * b.nrow_inlet
            * b.ncol_tube
            * b.nseg_tube
            * b.length_tube_seg
        )


def _make_geometry_tube(blk, shell_units):
    # Elevation difference (outlet - inlet) for static pressure calculation
    blk.delta_elevation = Var(
        initialize=0.0,
        units=shell_units["length"],
        doc="Elevation increase used for static pressure calculation",
    )

    # Length of tube side flow
    @blk.Constraint(doc="Length of tube side flow")
    def length_flow_tube_eqn(b):
        return (
            pyunits.convert(b.length_flow_tube, to_units=shell_units["length"])
            == b.nseg_tube * b.length_tube_seg
        )

    # Total flow area on tube side
    @blk.Constraint(doc="Total area of tube flow")
    def area_flow_tube_eqn(b):
        return (
            b.area_flow_tube
            == 0.25 * const.pi * b.di_tube**2.0 * b.ncol_tube * b.nrow_inlet
        )


def _make_performance_common(
    blk, shell, shell_units, shell_has_pressure_change, make_reynolds, make_nusselt
):
    # We need the Reynolds number for pressure change, even if we don't need it for heat transfer
    if shell_has_pressure_change:
        make_reynolds = True

    add_object_reference(blk, "heat_shell", shell.heat)

    if shell_has_pressure_change:
        add_object_reference(blk, "deltaP_shell", shell.deltaP)

    # Parameters
    if blk.config.has_radiation:
        # tube wall emissivity, converted from parameter to variable
        # TODO is this dimensionless?
        blk.emissivity_wall = Var(initialize=0.7, doc="shell side wall emissivity")

    # Wall thermal conductivity
    blk.therm_cond_wall = Param(
        initialize=1.0,
        mutable=True,
        units=shell_units["thermal_conductivity"],
        doc="loss coefficient of a tube u-turn should be 43.0",
    )

    # Wall heat capacity
    blk.cp_wall = Param(
        initialize=502.4,
        mutable=True,
        units=shell_units["heat_capacity_mass"],
        doc="metal wall heat capacity",
    )

    # Wall density
    blk.density_wall = Param(
        initialize=7800.0,
        mutable=True,
        units=shell_units["density_mass"],
        doc="metal wall density",
    )

    # Heat transfer resistance due to the fouling on shell side
    blk.rfouling_shell = Param(
        initialize=0.0001, mutable=True, doc="fouling resistance on tube side"
    )

    # Correction factor for convective heat transfer coefficient on shell side
    blk.fcorrection_htc_shell = Var(
        initialize=1.0, doc="correction factor for convective HTC on shell"
    )

    # Correction factor for shell side pressure drop due to friction
    if shell_has_pressure_change:
        blk.fcorrection_dp_shell = Var(
            initialize=1.0, doc="correction factor for shell side pressure drop"
        )

    # Performance variables
    if blk.config.has_radiation:
        # Gas emissivity at mbl
        blk.gas_emissivity = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=0.5,
            doc="emissivity at given mean beam length",
        )

        # Gas emissivity at mbl/sqrt(2)
        blk.gas_emissivity_div2 = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=0.4,
            doc="emissivity at mean beam length divided by sqrt of 2",
        )

        # Gas emissivity at mbl*sqrt(2)
        blk.gas_emissivity_mul2 = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=0.6,
            doc="emissivity at mean beam length multiplied by sqrt of 2",
        )

        # Gray fraction of gas in entire spectrum
        blk.gas_gray_fraction = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=0.5,
            doc="gray fraction of gas in entire spectrum",
        )

        # Gas-surface radiation exchange factor for shell side wall
        blk.frad_gas_shell = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=0.5,
            doc="gas-surface radiation exchange factor for shell side wall",
        )

        # Shell side equivalent convective heat transfer coefficient due to radiation
        blk.hconv_shell_rad = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=100.0,
            bounds=(0, None),
            units=shell_units["heat_transfer_coefficient"],
            doc="shell side convective heat transfer coefficient due to radiation",
        )

    # Shell side convective heat transfer coefficient due to convection only
    blk.hconv_shell_conv = Var(
        blk.flowsheet().config.time,
        shell.length_domain,
        initialize=100.0,
        bounds=(0, None),
        units=shell_units["heat_transfer_coefficient"],
        doc="shell side convective heat transfer coefficient due to convection",
    )

    # Boundary wall temperature on shell side
    blk.temp_wall_shell = Var(
        blk.flowsheet().config.time,
        shell.length_domain,
        initialize=500,
        units=shell_units["temperature"],
        doc="boundary wall temperature on shell side",
    )

    # Central wall temperature of tube metal, used to calculate energy contained by tube metal
    blk.temp_wall_center = Var(
        blk.flowsheet().config.time,
        shell.length_domain,
        initialize=500,
        units=shell_units["temperature"],
        doc="tube wall temperature at center",
    )

    # Tube wall heat holdup per length of shell
    if blk.config.has_holdup:
        blk.heat_holdup = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=1e6,
            units=shell_units["energy"] / shell_units["length"],
            doc="tube wall heat holdup per length of shell",
        )
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="heat holdup of tube metal",
        )
        def heat_holdup_eqn(b, t, x):
            return (
                b.heat_holdup[t, x]
                == b.cp_wall
                * b.density_wall
                * b.area_wall_seg
                * pyunits.convert(b.length_flow_tube, to_units=shell_units["length"])
                / b.length_flow_shell
                * b.temp_wall_center[t, x]
            )

    # Tube wall heat accumulation term
    if blk.config.dynamic:
        blk.heat_accumulation = DerivativeVar(
            blk.heat_holdup,
            initialize=0,
            wrt=blk.flowsheet().config.time,
            units=shell_units["energy"] / shell_units["length"] / shell_units["time"],
            doc="Tube wall heat accumulation per unit length of shell",
        )

    if blk.config.has_radiation:
        # TODO Make units consistent for radiation
        # Constraints for gas emissivity
        @blk.Constraint(
            blk.flowsheet().config.time, shell.length_domain, doc="Gas emissivity"
        )
        def gas_emissivity_eqn(b, t, x):
            X1 = shell.properties[t, x].temperature
            X2 = b.mbl
            X3 = shell.properties[t, x].pressure
            try:
                X4 = shell.properties[t, x].mole_frac_comp["CO2"]
            except KeyError:
                X4 = 0
            try:
                X5 = shell.properties[t, x].mole_frac_comp["H2O"]
            except KeyError:
                X5 = 0
            try:
                X6 = shell.properties[t, x].mole_frac_comp["O2"]
            except KeyError:
                X6 = 0
            return (
                b.gas_emissivity[t, x]
                == -0.000116906 * X1
                + 1.02113 * X2
                + 4.81687e-07 * X3
                + 0.922679 * X4
                - 0.0708822 * X5
                - 0.0368321 * X6
                + 0.121843 * log(X1)
                + 0.0353343 * log(X2)
                + 0.0346181 * log(X3)
                + 0.0180859 * log(X5)
                - 0.256274 * exp(X2)
                - 0.674791 * exp(X4)
                - 0.724802 * sin(X2)
                - 0.0206726 * cos(X2)
                - 9.01012e-05 * cos(X3)
                - 3.09283e-05 * X1 * X2
                - 5.44339e-10 * X1 * X3
                - 0.000196134 * X1 * X5
                + 4.54838e-05 * X1 * X6
                + 7.57411e-07 * X2 * X3
                + 0.0395456 * X2 * X4
                + 0.726625 * X2 * X5
                - 0.034842 * X2 * X6
                + 4.00056e-06 * X3 * X5
                + 5.71519e-09 * (X1 * X2) ** 2
                - 1.27853 * (X2 * X5) ** 2
            )

        # Constraints for gas emissivity at mbl/sqrt(2)
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="Gas emissivity at a lower mean beam length",
        )
        def gas_emissivity_div2_eqn(b, t, x):
            X1 = shell.properties[t, x].temperature
            X2 = b.mbl_div2
            X3 = shell.properties[t, x].pressure
            try:
                X4 = shell.properties[t, x].mole_frac_comp["CO2"]
            except KeyError:
                X4 = 0
            try:
                X5 = shell.properties[t, x].mole_frac_comp["H2O"]
            except KeyError:
                X5 = 0
            try:
                X6 = shell.properties[t, x].mole_frac_comp["O2"]
            except KeyError:
                X6 = 0
            return (
                b.gas_emissivity_div2[t, x]
                == -0.000116906 * X1
                + 1.02113 * X2
                + 4.81687e-07 * X3
                + 0.922679 * X4
                - 0.0708822 * X5
                - 0.0368321 * X6
                + 0.121843 * log(X1)
                + 0.0353343 * log(X2)
                + 0.0346181 * log(X3)
                + 0.0180859 * log(X5)
                - 0.256274 * exp(X2)
                - 0.674791 * exp(X4)
                - 0.724802 * sin(X2)
                - 0.0206726 * cos(X2)
                - 9.01012e-05 * cos(X3)
                - 3.09283e-05 * X1 * X2
                - 5.44339e-10 * X1 * X3
                - 0.000196134 * X1 * X5
                + 4.54838e-05 * X1 * X6
                + 7.57411e-07 * X2 * X3
                + 0.0395456 * X2 * X4
                + 0.726625 * X2 * X5
                - 0.034842 * X2 * X6
                + 4.00056e-06 * X3 * X5
                + 5.71519e-09 * (X1 * X2) ** 2
                - 1.27853 * (X2 * X5) ** 2
            )

        # Constraints for gas emissivity at mbl*sqrt(2)
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="Gas emissivity at a higher mean beam length",
        )
        def gas_emissivity_mul2_eqn(b, t, x):
            X1 = shell.properties[t, x].temperature
            X2 = b.mbl_mul2
            X3 = shell.properties[t, x].pressure
            try:
                X4 = shell.properties[t, x].mole_frac_comp["CO2"]
            except KeyError:
                X4 = 0
            try:
                X5 = shell.properties[t, x].mole_frac_comp["H2O"]
            except KeyError:
                X5 = 0
            try:
                X6 = shell.properties[t, x].mole_frac_comp["O2"]
            except KeyError:
                X6 = 0
            return (
                b.gas_emissivity_mul2[t, x]
                == -0.000116906 * X1
                + 1.02113 * X2
                + 4.81687e-07 * X3
                + 0.922679 * X4
                - 0.0708822 * X5
                - 0.0368321 * X6
                + 0.121843 * log(X1)
                + 0.0353343 * log(X2)
                + 0.0346181 * log(X3)
                + 0.0180859 * log(X5)
                - 0.256274 * exp(X2)
                - 0.674791 * exp(X4)
                - 0.724802 * sin(X2)
                - 0.0206726 * cos(X2)
                - 9.01012e-05 * cos(X3)
                - 3.09283e-05 * X1 * X2
                - 5.44339e-10 * X1 * X3
                - 0.000196134 * X1 * X5
                + 4.54838e-05 * X1 * X6
                + 7.57411e-07 * X2 * X3
                + 0.0395456 * X2 * X4
                + 0.726625 * X2 * X5
                - 0.034842 * X2 * X6
                + 4.00056e-06 * X3 * X5
                + 5.71519e-09 * (X1 * X2) ** 2
                - 1.27853 * (X2 * X5) ** 2
            )

        # fraction of gray gas spectrum
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="fraction of gray gas spectrum",
        )
        def gas_gray_fraction_eqn(b, t, x):
            return (
                b.gas_gray_fraction[t, x]
                * (2 * b.gas_emissivity_div2[t, x] - b.gas_emissivity_mul2[t, x])
                == b.gas_emissivity_div2[t, x] ** 2
            )

        # gas-surface radiation exchange factor between gas and shell side wall
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="gas-surface radiation exchange factor between gas and shell side wall",
        )
        def frad_gas_shell_eqn(b, t, x):
            return (
                b.frad_gas_shell[t, x]
                * (
                    (1 / b.emissivity_wall - 1) * b.gas_emissivity[t, x]
                    + b.gas_gray_fraction[t, x]
                )
                == b.gas_gray_fraction[t, x] * b.gas_emissivity[t, x]
            )

        # equivalent convective heat transfer coefficient due to radiation
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="equivalent convective heat transfer coefficient due to radiation",
        )
        def hconv_shell_rad_eqn(b, t, x):
            return b.hconv_shell_rad[t, x] == (
                pyunits.convert(
                    const.stefan_constant,
                    to_units=shell_units["power"]
                    / shell_units["length"] ** 2
                    / shell_units["temperature"] ** 4,
                )
                * b.frad_gas_shell[t, x]
                * (shell.properties[t, x].temperature + b.temp_wall_shell[t, x])
                * (
                    shell.properties[t, x].temperature ** 2
                    + b.temp_wall_shell[t, x] ** 2
                )
            )

    # Pressure drop and heat transfer coefficient on shell side
    # ----------------------------------------------------------
    # Tube arrangement factor
    if blk.config.tube_arrangement == "in-line":
        blk.f_arrangement = Param(
            initialize=0.788, doc="in-line tube arrangement factor"
        )
    elif blk.config.tube_arrangement == "staggered":
        blk.f_arrangement = Param(
            initialize=1.0, doc="staggered tube arrangement factor"
        )
    else:
        raise ConfigurationError()

    if make_reynolds:
        # Velocity on shell side
        blk.v_shell = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=1.0,
            units=shell_units["velocity"],
            doc="velocity on shell side",
        )

        # Reynalds number on shell side
        blk.N_Re_shell = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            bounds=(1e-7, None),
            initialize=10000.0,
            units=pyunits.dimensionless,
            doc="Reynolds number on shell side",
        )

    if shell_has_pressure_change:
        # Friction factor on shell side
        # TODO does this have units?
        blk.friction_factor_shell = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=1.0,
            doc="friction factor on shell side",
        )

    if make_nusselt:
        # Nusselt number on shell side
        blk.N_Nu_shell = Var(
            blk.flowsheet().config.time,
            shell.length_domain,
            initialize=1,
            units=pyunits.dimensionless,
            doc="Nusselts number on shell side",
            bounds=(1e-7, None),
        )

    if make_reynolds:
        # Velocity equation on shell side
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="velocity on shell side",
        )
        def v_shell_eqn(b, t, x):
            return (
                b.v_shell[t, x]
                * shell.properties[t, x].dens_mol_phase["Vap"]
                * b.area_flow_shell_min
                == shell.properties[t, x].flow_mol
            )

        # Reynolds number
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="Reynolds number equation on shell side",
        )
        def N_Re_shell_eqn(b, t, x):
            return (
                b.N_Re_shell[t, x] * shell.properties[t, x].visc_d_phase["Vap"]
                == b.do_tube
                * b.v_shell[t, x]
                * shell.properties[t, x].dens_mol_phase["Vap"]
                * shell.properties[t, x].mw
            )
            # return b.N_Re_shell[t,x] * shell.properties[t,x].visc_d * b.area_flow_shell_min == \
            #     b.do_tube * shell.properties[t,x].dens_mol_phase["Vap"]**2 * sum(
            #         shell.properties[t,x].flow_mol_comp[j]*shell.properties[t,x].mw_comp[j]
            #         for j in shell.properties[t,x].component_list)

    if shell_has_pressure_change == True:
        # Friction factor on shell side
        if blk.config.tube_arrangement == "in-line":

            @blk.Constraint(
                blk.flowsheet().config.time,
                shell.length_domain,
                doc="in-line friction factor on shell side",
            )
            def friction_factor_shell_eqn(b, t, x):
                return (
                    b.friction_factor_shell[t, x] * b.N_Re_shell[t, x] ** 0.15
                    == (
                        0.044
                        + 0.08
                        * b.pitch_x_to_do
                        / (b.pitch_y_to_do - 1.0) ** (0.43 + 1.13 / b.pitch_x_to_do)
                    )
                    * b.fcorrection_dp_shell
                )

        elif blk.config.tube_arrangement == "staggered":

            @blk.Constraint(
                blk.flowsheet().config.time,
                shell.length_domain,
                doc="staggered friction factor on shell side",
            )
            def friction_factor_shell_eqn(b, t, x):
                return (
                    b.friction_factor_shell[t, x] * b.N_Re_shell[t, x] ** 0.16
                    == (0.25 + 0.118 / (b.pitch_y_to_do - 1.0) ** 1.08)
                    * b.fcorrection_dp_shell
                )

        else:
            raise ConfigurationError()

        # Pressure drop on shell side
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="pressure change on shell side",
        )
        def deltaP_shell_eqn(b, t, x):
            return (
                b.deltaP_shell[t, x] * b.pitch_x
                == -1.4
                * b.friction_factor_shell[t, x]
                * shell.properties[t, x].dens_mol_phase["Vap"]
                * shell.properties[t, x].mw
                * b.v_shell[t, x] ** 2
            )

    if make_nusselt:
        # The actual Nusselt number correlation needs to be made by the particular heat exchanger
        @blk.Constraint(
            blk.flowsheet().config.time,
            shell.length_domain,
            doc="Convective heat transfer coefficient equation on shell side due to convection",
        )
        def hconv_shell_conv_eqn(b, t, x):
            return (
                b.hconv_shell_conv[t, x] * b.do_tube
                == b.N_Nu_shell[t, x]
                * shell.properties[t, x].therm_cond_phase["Vap"]
                * b.fcorrection_htc_shell
            )

    # Total convective heat transfer coefficient on shell side
    @blk.Expression(
        blk.flowsheet().config.time,
        shell.length_domain,
        doc="Total convective heat transfer coefficient on shell side",
    )
    def hconv_shell_total(b, t, x):
        if blk.config.has_radiation:
            return b.hconv_shell_conv[t, x] + b.hconv_shell_rad[t, x]
        else:
            return b.hconv_shell_conv[t, x]


def _make_performance_tube(
    blk, tube, tube_units, tube_has_pressure_change, make_reynolds, make_nusselt
):

    # Need Reynolds number for pressure drop, even if we don't need it for heat transfer
    if tube_has_pressure_change:
        make_reynolds = True

    blk.hconv_tube = Var(
        blk.flowsheet().config.time,
        tube.length_domain,
        initialize=100.0,
        doc="tube side convective heat transfer coefficient",
    )

    # Loss coefficient for a 180 degree bend (u-turn), usually related to radius to inside diameter ratio
    blk.kloss_uturn = Param(
        initialize=0.5, mutable=True, doc="loss coefficient of a tube u-turn"
    )

    # Heat transfer resistance due to the fouling on tube side
    blk.rfouling_tube = Param(
        initialize=0.0, mutable=True, doc="fouling resistance on tube side"
    )
    # Correction factor for convective heat transfer coefficient on tube side
    blk.fcorrection_htc_tube = Var(
        initialize=1.0, doc="correction factor for convective HTC on tube side"
    )
    # Correction factor for tube side pressure drop due to friction
    if tube_has_pressure_change:
        blk.fcorrection_dp_tube = Var(
            initialize=1.0, doc="correction factor for tube side pressure drop"
        )

    # Boundary wall temperature on tube side
    blk.temp_wall_tube = Var(
        blk.flowsheet().config.time,
        tube.length_domain,
        initialize=500,
        units=tube_units["temperature"],
        doc="boundary wall temperature on tube side",
    )
    if make_reynolds:
        # Tube side heat transfer coefficient and pressure drop
        # -----------------------------------------------------
        # Velocity on tube side
        blk.v_tube = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=1.0,
            units=tube_units["velocity"],
            doc="velocity on tube side",
        )

        # Reynalds number on tube side
        blk.N_Re_tube = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=10000.0,
            units=pyunits.dimensionless,
            doc="Reynolds number on tube side",
            bounds=(1e-7, None),
        )
    if tube_has_pressure_change == True:
        # Friction factor on tube side
        # TODO does this have units?
        blk.friction_factor_tube = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=1.0,
            doc="friction factor on tube side",
        )

        # Pressure drop due to friction on tube side
        blk.deltaP_tube_friction = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=-10.0,
            units=tube_units["pressure"],
            doc="pressure drop due to friction on tube side",
        )

        # Pressure drop due to 180 degree turn on tube side
        blk.deltaP_tube_uturn = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=-10.0,
            units=tube_units["pressure"],
            doc="pressure drop due to u-turn on tube side",
        )
    if make_nusselt:
        # Nusselt number on tube side
        blk.N_Nu_tube = Var(
            blk.flowsheet().config.time,
            tube.length_domain,
            initialize=1,
            units=pyunits.dimensionless,
            doc="Nusselts number on tube side",
            bounds=(1e-7, None),
        )

    if make_reynolds:
        # Velocity equation
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="tube side velocity equation",
        )
        def v_tube_eqn(b, t, x):
            return (
                b.v_tube[t, x]
                * pyunits.convert(b.area_flow_tube, to_units=tube_units["area"])
                * tube.properties[t, x].dens_mol_phase["Vap"]
                == tube.properties[t, x].flow_mol
            )

        # Reynolds number
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="Reynolds number equation on tube side",
        )
        def N_Re_tube_eqn(b, t, x):
            return (
                b.N_Re_tube[t, x] * tube.properties[t, x].visc_d_phase["Vap"]
                == pyunits.convert(b.di_tube, to_units=tube_units["length"])
                * b.v_tube[t, x]
                * tube.properties[t, x].dens_mol_phase["Vap"]
                * tube.properties[t, x].mw
            )
            # return b.N_Re_tube[t,x] * tube.properties[t,x].visc_d * b.area_flow_tube == \
            #     b.di_tube * tube.properties[t,x].dens_mol_phase["Vap"]**2 * sum(
            #         tube.properties[t,x].flow_mol_comp[j]*tube.properties[t,x].mw_comp[j]
            #         for j in tube.properties[t,x].component_list)

    if tube_has_pressure_change:
        # Friction factor
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="Darcy friction factor on tube side",
        )
        def friction_factor_tube_eqn(b, t, x):
            return (
                b.friction_factor_tube[t, x] * b.N_Re_tube[t, x] ** 0.25
                == 0.3164 * b.fcorrection_dp_tube
            )

        # Pressure drop due to friction per tube length
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="pressure drop due to friction per tube length",
        )
        def deltaP_tube_friction_eqn(b, t, x):
            return (
                b.deltaP_tube_friction[t, x]
                * pyunits.convert(b.di_tube, to_units=tube_units["length"])
                == -0.5
                * tube.properties[t, x].dens_mol_phase["Vap"]
                * tube.properties[t, x].mw
                * b.v_tube[t, x] ** 2
                * b.friction_factor_tube[t, x]
            )

        # Pressure drop due to u-turn
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="pressure drop due to u-turn on tube side",
        )
        def deltaP_tube_uturn_eqn(b, t, x):
            return (
                b.deltaP_tube_uturn[t, x]
                * pyunits.convert(b.length_tube_seg, to_units=tube_units["length"])
                == -0.5
                * tube.properties[t, x].dens_mol_phase["Vap"]
                * tube.properties[t, x].mw
                * b.v_tube[t, x] ** 2
                * b.kloss_uturn
            )

        # Total pressure drop on tube side
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="total pressure drop on tube side",
        )
        def deltaP_tube_eqn(b, t, x):
            return b.deltaP_tube[t, x] == (
                b.deltaP_tube_friction[t, x]
                + b.deltaP_tube_uturn[t, x]
                - pyunits.convert(b.delta_elevation, to_units=tube_units["length"])
                / b.nseg_tube
                * pyunits.convert(
                    const.acceleration_gravity, to_units=tube_units["acceleration"]
                )
                * tube.properties[t, x].dens_mol_phase["Vap"]
                * tube.properties[t, x].mw
                / pyunits.convert(b.length_tube_seg, to_units=tube_units["length"])
            )

    if make_nusselt:
        @blk.Constraint(
            blk.flowsheet().config.time,
            tube.length_domain,
            doc="convective heat transfer coefficient equation on tube side",
        )
        def hconv_tube_eqn(b, t, x):
            return (
                b.hconv_tube[t, x] * b.di_tube
                == b.N_Nu_tube[t, x]
                * tube.properties[t, x].therm_cond_phase["Vap"]
                * b.fcorrection_htc_tube
            )
def _scale_common(blk, shell, shell_has_pressure_change, make_reynolds, make_nusselt):
    def gsf(obj):
        return iscale.get_scaling_factor(obj, default=1, warning=True)
    def ssf(obj, sf):
        iscale.set_scaling_factor(obj, sf, overwrite=False)
    def cst(con, sf):
        iscale.constraint_scaling_transform(con, sf, overwrite=False)
    sgsf = iscale.set_and_get_scaling_factor

    sf_do_tube = iscale.get_scaling_factor(
        blk.do_tube, default=1 / value(blk.do_tube)
    )

    sf_di_tube = iscale.get_scaling_factor(
        blk.do_tube, default=1 / value(blk.di_tube)
    )
    calculate_variable_from_constraint(
        blk.area_flow_shell_min,
        blk.area_flow_shell_min_eqn
    )
    sf_area_flow_shell_min = iscale.get_scaling_factor(
        blk.area_flow_shell_min, default=1/value(blk.area_flow_shell_min)
    )
    for t in blk.flowsheet().time:
        for z in shell.length_domain:
            sf_flow_mol_shell = gsf(shell.properties[t, z].flow_mol)

            if make_reynolds:
                # FIXME get better scaling later
                ssf(blk.v_shell[t, z], 1/10)
                cst(blk.v_shell_eqn[t, z], sf_flow_mol_shell)

                #FIXME should get scaling of N_Re from defining eqn
                sf_N_Re_shell = sgsf(blk.N_Re_shell[t, z], 1e-4)

                sf_visc_d_shell = gsf(shell.properties[t, z].visc_d_phase["Vap"])
                cst(blk.N_Re_shell_eqn[t, z], sf_N_Re_shell * sf_visc_d_shell)
            if make_nusselt:
                sf_k_shell = gsf(shell.properties[t, z].therm_cond_phase["Vap"])

                sf_N_Nu_shell = sgsf(
                    blk.N_Nu_shell[t, z], 1 / 0.33 * sf_N_Re_shell ** 0.6
                )
                cst(blk.N_Nu_shell_eqn[t, z], sf_N_Nu_shell)

                sf_hconv_shell_conv = sgsf(
                    blk.hconv_shell_conv[t, z], sf_N_Nu_shell * sf_k_shell / sf_do_tube
                )
                cst(blk.hconv_shell_conv_eqn[t, z], sf_hconv_shell_conv * sf_do_tube)



            # FIXME estimate from parameters
            if blk.config.has_holdup:
                s_U_holdup = gsf(blk.heat_holdup[t, z])
                cst(blk.heat_holdup_eqn[t, z], s_U_holdup)

def _scale_tube(blk, tube, tube_has_presure_change, make_reynolds, make_nusselt):
    def gsf(obj):
        return iscale.get_scaling_factor(obj, default=1, warning=True)

    def ssf(obj, sf):
        iscale.set_scaling_factor(obj, sf, overwrite=False)

    def cst(con, sf):
        iscale.constraint_scaling_transform(con, sf, overwrite=False)

    sgsf = iscale.set_and_get_scaling_factor

    sf_di_tube = iscale.get_scaling_factor(
        blk.do_tube, default=1 / value(blk.di_tube)
    )
    sf_do_tube = iscale.get_scaling_factor(
        blk.do_tube, default=1 / value(blk.do_tube)
    )

    for t in blk.flowsheet().time:
        for z in tube.length_domain:
            if make_reynolds:
                # FIXME get better scaling later
                ssf(blk.v_tube[t, z], 1/20)
                sf_flow_mol_tube = gsf(tube.properties[t, z].flow_mol)

                cst(blk.v_tube_eqn[t, z], sf_flow_mol_tube)

                # FIXME should get scaling of N_Re from defining eqn
                sf_N_Re_tube = sgsf(blk.N_Re_tube[t, z], 1e-4)

                sf_visc_d_tube = gsf(tube.properties[t, z].visc_d_phase["Vap"])
                cst(blk.N_Re_tube_eqn[t, z], sf_N_Re_tube * sf_visc_d_tube)
            if make_nusselt:
                sf_k_tube = gsf(tube.properties[t, z].therm_cond_phase["Vap"])

                sf_N_Nu_tube = sgsf(
                    blk.N_Nu_tube[t, z], 1 / 0.023 * sf_N_Re_tube ** 0.8
                )
                cst(blk.N_Nu_tube_eqn[t, z], sf_N_Nu_tube)

                sf_hconv_tube = sgsf(
                    blk.hconv_tube[t, z], sf_N_Nu_tube * sf_k_tube / sf_di_tube
                )
                cst(blk.hconv_tube_eqn[t, z], sf_hconv_tube * sf_di_tube)