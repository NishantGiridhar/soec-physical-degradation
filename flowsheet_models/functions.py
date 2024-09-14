# from pyomo.environ import SolverFactory
from idaes.core.util.initialization import initialize_by_time_element
import logging
from idaes.core.solvers import petsc
import idaes.logger as idaeslog
from pyomo.dae.flatten import flatten_dae_components
import logging
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet
import idaes.core.util.scaling as iscale

# Initial logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_scaling(model, solver='ipopt', large_threshold=1E3, small_threshold=5E-3):
    """
    Perform scaling analysis on a Pyomo model.

    Parameters:
    - model: The Pyomo model on which to perform scaling analysis.
    - solver: The solver to use for the optimization (default is 'ipopt').
    - large_threshold: Threshold for identifying large elements in the Jacobian (default is 1E3).
    - small_threshold: Threshold for identifying small elements in the Jacobian (default is 5E-3).
    """
    solver = SolverFactory(solver)
    solver.options['max_iter'] = 0
    results_opt = solver.solve(model, tee=True)

    jac, nlp = iscale.get_jacobian(model, scaled=False, equality_constraints_only=True)
    variables = nlp.get_pyomo_variables()
    constraints = nlp.get_pyomo_equality_constraints()

    print("Badly scaled variables:")
    bad_vars = iscale.extreme_jacobian_columns(jac=jac, nlp=nlp, large=large_threshold, small=small_threshold)
    for var_value, var_name in bad_vars:
        print(f"    {var_value:.2e}, [{var_name}]")

    print("\n\n" + "Badly scaled constraints:")
    bad_cons = iscale.extreme_jacobian_rows(jac=jac, nlp=nlp, large=large_threshold, small=small_threshold)
    for con_value, con_name in bad_cons:
        print(f"    {con_value:.2e}, [{con_name}]")

    with open('scaling_issues.txt', 'w') as file:
        file.write("Badly scaled variables:\n")
        for var_value, var_name in bad_vars:
            file.write(f"    {var_value:.2e}, [{var_name}]\n")
        file.write("\n\nBadly scaled constraints:\n")
        for con_value, con_name in bad_cons:
            file.write(f"    {con_value:.2e}, [{con_name}]\n")
def calculate_degrees_of_freedom(model):
    """
    Calculate the degrees of freedom in the model.
    Degrees of freedom are defined as the difference between the number of active, unfixed variables
    and the number of active equality constraints.

    Parameters:
    - model: The Pyomo model instance.

    Returns:
    - int: The calculated degrees of freedom.
    """
    # Count active, unfixed variables
    active_vars = sum(1 for v in model.component_objects(pyo.Var, active=True)
                      for idx in v if not v[idx].fixed)

    # Count active equality constraints
    active_eqs = sum(1 for c in model.component_objects(pyo.Constraint, active=True)
                     for idx in c if c[idx].equality)

    # Calculate degrees of freedom
    dof = active_vars - active_eqs
    return dof

def adjust_model_component_activation(model, active_time_segment):
    """
    Adjusts the activation of model components based on the specified active time segment.

    Parameters:
    - model: The Pyomo model or a specific part of the model (e.g., model.fs) containing the dynamic components.
    - active_time_segment: A tuple specifying the start and end of the active time segment (start, end).
    """
    start, end = active_time_segment  # These should already be floats

    # Iterate over all components in the model
    for component in model.component_objects(pyo.Constraint, active=None):
        if component.is_indexed():
            for index in component:
                # Convert time_index to float for comparison, if it's a string
                time_index = index[0] if isinstance(index, tuple) else index
                try:
                    # Attempt to convert time_index to float, if necessary
                    time_index = float(time_index)
                except ValueError:
                    # If conversion fails, log a warning or error, and skip this index
                    continue

                if start <= time_index <= end:
                    component[index].activate()
                else:
                    component[index].deactivate()

    # {for component in model.component_objects(pyo.Constraint, active=None):
    #         if component.is_indexed():
    #             for index in component:
    #                 try:
    #                     time_index = float(index[0] if isinstance(index, tuple) else index)
    #                     if start <= time_index <= end:
    #                         component[index].activate()
    #                         logger.debug(f"Activated constraint {component.name}[{index}] for segment {start} to {end}.")
    #                     else:
    #                         component[index].deactivate()
    #                         logger.debug(f"Deactivated constraint {component.name}[{index}] outside segment {start} to {end}.")
    #                 except ValueError:
    #                     continue}

def initialize_by_time_element_with_dof_check(model, time_segments, solver_options=None, log_level=idaeslog.INFO):
    """
    Initializes the model by segments after adjusting component activation and checking degrees of freedom.

    Parameters:
    - model: The Pyomo model instance to initialize.
    - time_segments: A list of tuples defining the start and end times of each segment.
    - solver_options: Dictionary of solver options, if any.
    - log_level: Logging level.
    """
    solver = pyo.SolverFactory('ipopt')  # Example: Using IPOPT solver

    # Optionally, set solver options here if provided
    if solver_options:
        for key, value in solver_options.items():
            solver.options[key] = value

    for start, end in time_segments:
        adjust_model_component_activation(model, (start, end))

        # Perform DoF check
        dof = calculate_degrees_of_freedom(model)
        if dof != 0:
            raise ValueError(
                f"Non-zero degrees of freedom ({dof}) found for segment [{start}, {end}]. Model is not ready for initialization.")

        print(f"Initializing model for time segment: {start} to {end}")


def solve_with_petsc_at_time_points(m, ts_options, solver_options=None, log_level=idaeslog.INFO):
    time_points = list(m.fs.time)  # Assuming m.fs.time is sorted

    # Initial solver setup (if not provided)
    if solver_options is None:
        solver_options = {
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-6,
            'halt_on_ampl_error': 'yes',
        }

    # Loop through each time point, solving for each and initializing the next with the result
    for i in range(1, len(time_points)):
        current_time = time_points[i]
        previous_time = time_points[i - 1]

        logger.info(f"Solving from {previous_time} to {current_time}")

        # Modify ts_options for current segment
        ts_options_modified = ts_options.copy()
        ts_options_modified['--ts_dt'] = current_time - previous_time

        # # Modify the model's time domain for the current time segment
        # if hasattr(m.fs, 'time_set'):
        #     del m.fs.time_set  # Remove existing ContinuousSet to avoid accumulation
        # m.fs.time_set = ContinuousSet(bounds=(previous_time, current_time))

        try:
            # Solve using PETSc for the time interval [previous_time, current_time]
            print('Solving model with Petsc at interval = ', previous_time, current_time)
            petsc_result = petsc.petsc_dae_by_time_element(
                m,
                between=[previous_time, current_time],
                time=m.fs.time,
                keepfiles=True,
                symbolic_solver_labels=True,
                ts_options=ts_options_modified,
                initial_solver="ipopt",
                initial_solver_options=solver_options,
            )
            logger.info("Solver completed successfully for the current segment.")

            # # Initialize the model at the current time interval
            # initialize_by_time_element(m.fs, time=m.fs.time_set, solver=pyo.SolverFactory('ipopt'), outlvl=log_level)

        except Exception as e:
            logger.error(f"Solver failed for segment {previous_time} to {current_time}: {e}")
            break  # Consider more nuanced error recovery or logging strategies

    logger.info("Completed PETSc solver execution for all time steps.")


def copy_initial_steady_state(m):
    """
    Propagates initial steady state guesses to future time points in a dynamic Pyomo model.

    Parameters:
    - m: The Pyomo model instance containing a dynamic model with a defined time set.
    """

    # Ensure the model and its time set are correctly specified
    if not hasattr(m, 'fs') or not hasattr(m.fs, 'time'):
        raise ValueError("Model does not have the expected structure with m.fs.time defined.")

    try:
        # Extract active variables, categorized as regular or time-dependent
        regular_vars, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)

        # Copy initial conditions forward for time-dependent variables
        for var in time_vars:
            initial_value = var[m.fs.time.first()].value  # Store initial value
            for t in m.fs.time:
                if t != m.fs.time.first():
                    var[t].value = initial_value

        print("Copied initial conditions to all future time points.")

    except Exception as e:
        # Handle potential errors in the flattening process or variable handling
        print(f"An error occurred while copying initial conditions: {e}")


def copy_values_to_next_time_step(m):
    """
    For each time-dependent variable in the dynamic Pyomo model, copies the value from
    the previous time step to the next, for all time steps after the initial one.

    Parameters:
    - m: The Pyomo model instance containing a dynamic model with a defined time set.
    """

    # Ensure the model and its time set are correctly specified
    if not hasattr(m, 'fs') or not hasattr(m.fs, 'time'):
        raise ValueError("Model does not have the expected structure with m.fs.time defined.")

    try:
        # Extract active variables, categorized as regular or time-dependent
        regular_vars, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)

        # Obtain a sorted list of the time points to ensure correct sequential processing
        time_points = sorted(m.fs.time)

        # Copy values from the previous time step to the next for time-dependent variables
        for var in time_vars:
            for i, t in enumerate(time_points):
                if i == 0:  # Skip the first time point
                    continue
                # Copy value from the previous time step
                var[t].value = var[time_points[i - 1]].value

        print("Copied values from each previous time step to the next.")

    except Exception as e:
        # Handle potential errors
        print(f"An error occurred while copying values to next time steps: {e}")


def initialize_model_with_propagation(m, solver_options=None, log_level=idaeslog.INFO):
    """
    Initializes a dynamic Pyomo model by time element and propagates the variable values
    from each initialized time step to the next.

    Parameters:
    - m: The Pyomo model instance containing a dynamic model with a defined time set.
    - solver_options: Dictionary of solver options.
    - log_level: Logging level from IDAES logger.
    """

    # Ensure solver options are provided, else default to a basic setup
    if solver_options is None:
        solver_options = {"tol": 1e-6, "max_iter": 1000}

    # Setup solver
    solver = pyo.SolverFactory('ipopt')
    solver.options.update(solver_options)

    # Initialize the model at each time element
    initialize_by_time_element(m.fs, time=m.fs.time, solver=solver, outlvl=log_level)

    # Extract active variables, categorized as regular or time-dependent
    _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)

    # Obtain a sorted list of the time points to ensure correct sequential processing
    time_points = sorted(m.fs.time)

    # Propagate values from the previous time step to the next for time-dependent variables
    for var in time_vars:
        for i, t in enumerate(time_points):
            if i == 0:  # Skip the first time point
                continue
            # Copy value from the previous time step
            var[t].value = var[time_points[i - 1]].value

    # # Initialize the model at each time element
    # initialize_by_time_element(m.fs, time=m.fs.time, solver=solver, outlvl=log_level)

    print("Model initialized and values propagated to next time steps.")

import idaes.logger as idaeslog
from pyomo.environ import SolverFactory, value
from pyomo.dae import ContinuousSet

def solve_and_initialize_with_petsc(m, ts_options, solver_options=None, log_level=idaeslog.INFO):
    """
    Solves a dynamic Pyomo model across multiple time intervals using PETSc,
    then initializes the solution for the next time interval.

    Parameters:
    - m: The Pyomo model instance containing a dynamic model with a defined time set.
    - ts_options: A dictionary of options for the PETSc time stepper.
    - solver_options: A dictionary of solver options for initial solver and PETSc, if None, default options are used.
    - log_level: Logging level for the process.
    """
    # Ensure logger is set up with the specified log level
    logger = idaeslog.getLogger(__name__, level=log_level)

    # Define default solver options if none provided
    if solver_options is None:
        solver_options = {
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-6,
            'halt_on_ampl_error': 'yes',
        }

    time_points = list(m.fs.time)  # Assuming m.fs.time is sorted

    for i in range(1, len(time_points)):
        current_time = time_points[i]
        previous_time = time_points[i - 1]

        logger.info(f"Solving from {previous_time} to {current_time}")

        # Modify ts_options for the current time segment
        ts_options_modified = ts_options.copy()
        # ts_options_modified['--ts_dt'] = current_time - previous_time

        try:
            # Solve using PETSc for the time interval
            logger.info(f'Solving model with Petsc at interval = [{previous_time}, {current_time}]')
            petsc_result = solve_with_petsc_at_time_points(m, ts_options_modified, solver_options, log_level)

            # Copy values to next time step as initial guess for PETSc
            copy_values_to_next_time_step(m)

        except Exception as e:
            logger.error(f"Solver or initialization failed for segment [{previous_time}, {current_time}]: {e}")
            break  # Consider more nuanced error recovery or logging strategies

    logger.info("Completed dynamic solving and initialization for all time steps.")


# def initialize_model_with_propagation_and_petsc_solver(m, ts_options, solver_options=None, log_level=idaeslog.INFO):
#     """
#     Initializes a dynamic Pyomo model by time element, propagates variable values
#     from each initialized time step to the next, and solves the model using PETSc solver.
#
#     Parameters:
#     - m: The Pyomo model instance containing a dynamic model with a defined time set.
#     - ts_options: Dictionary of PETSc time-stepping options.
#     - solver_options: Dictionary of initial solver options for IPOPT.
#     - log_level: Logging level from IDAES logger.
#     """
#
#     # Ensure solver options for IPOPT are provided, else default to a basic setup
#     if solver_options is None:
#         solver_options = {
#             'nlp_scaling_method': 'user-scaling',
#             'linear_solver': 'ma57',
#             'OF_ma57_automatic_scaling': 'yes',
#             'max_iter': 300,
#             'tol': 1e-6,
#             'halt_on_ampl_error': 'yes',
#         }
#     # time_points = sorted(m.fs.time)
#     time_points = list(m.fs.time)
#     for i in range(1, len(time_points)):
#         # Initialize the model at each time element using IDAES's utility
#         initialize_by_time_element(m.fs, time=m.fs.time, solver=pyo.SolverFactory('ipopt'), outlvl=log_level)
#
#         # Setup and execute the PETSc solver with the provided ts_options
#         # This part needs to be adapted based on how you're integrating PETSc with Pyomo/IDAES
#         # The following is a placeholder for the PETSc solver execution
#
#         idaeslog.solver_log.tee = True
#
#         # Setup basic logging
#         logging.basicConfig(level=logging.INFO)
#         logger = logging.getLogger(__name__)
#
#         current_time = time_points[i]
#         previous_time = time_points[i - 1]
#
#         # Activate model components for the current segment
#         activate_model_components(m, start=previous_time, end=current_time)
#
#         # Setup PETSc solver with modified ts_options for the current segment
#         ts_options_modified = ts_options.copy()
#         ts_options_modified['--ts_dt'] = current_time - previous_time
#
#         # Solve the model for the current time segment
#         try:
#             petsc_result = petsc.petsc_dae_by_time_element(
#                 m,
#                 time=m.fs.time,  # Ensuring the entire ContinuousSet is available
#                 keepfiles=True,
#                 symbolic_solver_labels=True,
#                 ts_options=ts_options_modified,
#                 initial_solver="ipopt",
#                 initial_solver_options=solver_options,
#             )
#             logger.info("Solver completed successfully for the current segment.")
#         except Exception as e:
#             logger.error(f"Solver failed for segment {previous_time} to {current_time}: {e}")
#             deactivate_model_components(m, start=previous_time, end=current_time)  # Ensure to clean up before next attempt
#             continue  # or break with some recovery logic
#
#         # Deactivate model components outside of the current segment
#         deactivate_model_components(m, start=previous_time, end=current_time)
#
#         # Propagate variable values from the previous time step to the next
#         _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)
#         # time_points = sorted(m.fs.time)
#         for var in time_vars:
#             for i, t in enumerate(time_points):
#                 if i == 0:
#                     continue
#                 # var[t].value = var[time_points[i - 1]].value
#                 var[t].set_value(var[time_points[i - 1]].value)
#         print("Model initialized and values propagated to next time steps.")

# def initialize_model_with_propagation_and_petsc_solver(m, ts_options, solver_options=None, log_level=idaeslog.INFO):
#     # Ensure solver options for IPOPT are provided, with a default setup if none provided
#     if solver_options is None:
#         solver_options = {
#             'nlp_scaling_method': 'user-scaling',
#             'linear_solver': 'ma57',
#             'OF_ma57_automatic_scaling': 'yes',
#             'max_iter': 300,
#             'tol': 1e-6,
#             'halt_on_ampl_error': 'yes',
#         }
#
#     time_points = list(m.fs.time)
#
#     # Propagate variable values as before
#     _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)
#     for var in time_vars:
#         for i, t in enumerate(time_points):
#             if i > 0:
#                 var[t].set_value(var[time_points[i - 1]].value)
#     print("Model initialized and values propagated to next time steps.")
#
#     # Basic logging setup
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#
#     # Iterate through time segments, activating components, attempting solution with PETSc, and handling errors
#     for i in range(1, len(time_points)):
#         current_time = time_points[i]
#         previous_time = time_points[i - 1]
#
#         activate_model_components(m, start=previous_time, end=current_time)
#
#         ts_options_modified = ts_options.copy()
#         ts_options_modified['--ts_dt'] = current_time - previous_time
#
#         try:
#             petsc_result = petsc.petsc_dae_by_time_element(
#                 m,
#                 time=m.fs.time,  # Ensuring the entire ContinuousSet is available
#                 keepfiles=True,
#                 symbolic_solver_labels=True,
#                 ts_options=ts_options_modified,
#                 initial_solver="ipopt",
#                 initial_solver_options=solver_options,
#             )
#             logger.info("Solver completed successfully for the current segment.")
#         except Exception as e:
#             logger.error(f"Solver failed for segment {previous_time} to {current_time}: {e}")
#             # deactivate_model_components(m, start=previous_time, end=current_time)
#             continue
#
#         deactivate_model_components(m, start=previous_time, end=current_time)
#         # activate_model_components(m, start=previous_time, end=current_time)
#
#     # Call to the custom initialization function that includes DoF checks
#     # This assumes you've defined time_segments based on your model's specifics
#     time_segments = [(time_points[i - 1], time_points[i]) for i in range(1, len(time_points))]
#     initialize_by_time_element_with_dof_check(m, time_segments, solver_options, log_level)
#
#     # initialize the model at each time element using IDAES's utility
#     initialize_by_time_element(m.fs, time=m.fs.time, solver=pyo.SolverFactory('ipopt'), outlvl=log_level)

def activate_model_components(model, start, end):
    start = float(start)
    end = float(end)

    for component in model.component_objects(pyo.Constraint, active=None):
        if component.is_indexed():
            for index in component:
                time_index = index[0] if isinstance(index, tuple) else index
                try:
                    time_index = float(time_index)
                except ValueError:
                    continue
                if start <= time_index <= end:
                    component[index].activate()
                else:
                    component[index].deactivate()
        else:
            # For non-indexed constraints, decide based on your model's logic
            pass


def deactivate_model_components(model, start, end):
    start = float(start)
    end = float(end)

    for component in model.component_objects(pyo.Constraint, active=None):
        if component.is_indexed():
            for index in component:
                time_index = index[0] if isinstance(index, tuple) else index
                try:
                    time_index = float(time_index)
                except ValueError:
                    continue
                if start <= time_index <= end:
                    component[index].deactivate()
        else:
            # For non-indexed constraints, handle as needed
            pass

# def initialize_model_with_propagation_and_petsc_solver(m, ts_options, solver_options=None, log_level=idaeslog.INFO):
#     if solver_options is None:
#         solver_options = {
#             'nlp_scaling_method': 'user-scaling',
#             'linear_solver': 'ma57',
#             'OF_ma57_automatic_scaling': 'yes',
#             'max_iter': 300,
#             'tol': 1e-6,
#             'halt_on_ampl_error': 'yes',
#         }
#
#     time_points = list(m.fs.time)
#     logging.basicConfig(level=log_level)
#     logger = logging.getLogger(__name__)
#
#     for i in range(1, len(time_points)):
#         current_time = time_points[i]
#         previous_time = time_points[i - 1]
#
#         # Activate model components for the current segment
#         activate_model_components(m, start=previous_time, end=current_time)
#
#         # Check degrees of freedom (Function calculate_degrees_of_freedom must be defined elsewhere)
#         dof = calculate_degrees_of_freedom(m)
#         if dof != 0:
#             logger.error(f"Non-zero degrees of freedom ({dof}) found for segment {previous_time} to {current_time}. Model is not ready for initialization.")
#             break
#
#         # Initialize the model for the current segment if needed
#         print(f"Initializing model for time segment: {previous_time} to {current_time}")
#         initialize_by_time_element(m.fs, time=m.fs.time, solver=SolverFactory('ipopt'), outlvl=log_level)
#
#         # Adjust solver options for the current time segment and execute the PETSc solver
#         ts_options_modified = ts_options.copy()
#         ts_options_modified['--ts_dt'] = current_time - previous_time
#
#         try:
#             petsc_result = petsc.petsc_dae_by_time_element(
#                 m,
#                 time=m.fs.time,
#                 keepfiles=True,
#                 symbolic_solver_labels=True,
#                 ts_options=ts_options_modified,
#                 initial_solver="ipopt",
#                 initial_solver_options=solver_options,
#             )
#             logger.info(f"Solver completed successfully for segment {previous_time} to {current_time}.")
#         except Exception as e:
#             logger.error(f"Solver failed for segment {previous_time} to {current_time}: {e}")
#             deactivate_model_components(m, start=previous_time, end=current_time)
#             continue
#
#         # Propagate variable values from the previous time step to the next
#         _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)
#         for var in time_vars:
#             if i < len(time_points) - 1:  # Ensure there is a next time point
#                 next_time = time_points[i + 1]
#                 var[next_time].set_value(var[current_time].value)
#
#         # Deactivate model components after processing this segment
#         deactivate_model_components(m, start=previous_time, end=current_time)
#
#     print("Model initialization and values propagated to next time steps.")

def initialize_model_with_propagation_and_petsc_solver(m, ts_options, solver_options=None, log_level=idaeslog.INFO):

    # Initial solver setup (if not provided)
    if solver_options is None:
        solver_options = {
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-6,
            'halt_on_ampl_error': 'yes',
        }

    initialize_by_time_element(m.fs, time=m.fs.time, solver=solver_options, outlvl=idaeslog.INFO)
    print('Initializing model with Petsc solver at time points')

    # Loop through each time point, solving for each and initializing the next with the result
    time_points = list(m.fs.time)  # Assuming m.fs.time is sorted
    for i in range(1, len(time_points)):
        current_time = time_points[i]
        previous_time = time_points[i - 1]

        logger.info(f"Solving from {previous_time} to {current_time}")

        # Modify ts_options for current segment
        ts_options_modified = ts_options.copy()
        # ts_options_modified['--ts_dt'] = current_time - previous_time

        try:
            # Solve using PETSc for the time interval [previous_time, current_time]
            print('Solving model with Petsc at interval = ', previous_time, current_time)
            petsc_result = petsc.petsc_dae_by_time_element(
                m,
                between=[previous_time, current_time],
                time=m.fs.time,
                keepfiles=True,
                symbolic_solver_labels=True,
                ts_options=ts_options_modified,
                initial_solver="ipopt",
                initial_solver_options=solver_options,
            )
            # pyo.assert_optimal_termination(petsc_result)
            logger.info("Solver completed successfully for the current segment.")

            # # Propagate variable values from the previous time step to the next
            # _, time_vars = flatten_dae_components(m, m.fs.time, pyo.Var, active=True)
            # for var in time_vars:
            #     if i < len(time_points) - 1:  # Ensure there is a next time point
            #         next_time = time_points[i + 1]
            #         var[next_time].set_value(var[current_time].value)
            #         print('Propagating variable values from the previous time step to the next')

            # # Call to the custom initialization function that includes DoF checks
            # # This assumes you've defined time_segments based on your model's specifics
            # # Modify the model's time domain for the current time segment
            # if hasattr(m.fs, 'time_set'):
            #     del m.fs.time_set  # Remove existing ContinuousSet to avoid accumulation
            # m.fs.time_set = ContinuousSet(bounds=(previous_time, current_time))

        except Exception as e:
            logger.error(f"Solver failed for segment {previous_time} to {current_time}: {e}")
            break  # Consider more nuanced error recovery or logging strategies

    logger.info("Completed PETSc solver execution for all time steps.")

def generate_time_set():
    # Define the time intervals in seconds
    t_start = 1 * 60 * 60  # Start time
    t_ramp = 5 * 60  # Ramp time
    t_settle = 5 * 60 * 60  # Settle time
    t_end = 3 * 60 * 60  # End time
    dt_set = [t_start, t_ramp, t_settle, t_ramp, t_end]

    # Calculate the cumulative sum of the time intervals to create the time set
    time_set_org = [sum(dt_set[:j]) for j in range(len(dt_set) + 1)]

    # Additional time elements for more granularity during transitions
    number_elements_first_interval = 3
    number_elements_second_interval = 3
    time_set_add1 = [time_set_org[0] + j * (time_set_org[1] - time_set_org[0]) / number_elements_first_interval for j in range(number_elements_first_interval)]
    time_set_add2 = [time_set_org[1] + j * (time_set_org[2] - time_set_org[1]) / number_elements_second_interval for j in range(number_elements_second_interval)]

    # Combine and sort all time points
    time_set = time_set_org + time_set_add1[1:] + time_set_add2[1:]
    time_set.sort()

    return time_set

def generate_corrected_ramp_setpoints(time_set):
    corrected_ramp_setpoints = []
    for time in time_set:
        if time <= 3600:  # t = 0 to 3600s
            corrected_ramp_setpoints.append("maximum_H2")
        elif 3600 < time <= 3900:  # t = 3600 to 3900s, transient time
            corrected_ramp_setpoints.append("minimum_H2")
        elif 3900 < time <= 21900:  # t = 3900 to 21900s
            corrected_ramp_setpoints.append("minimum_H2")
        elif 21900 < time <= 22200:  # t = 21900 to 22200, transient time
            corrected_ramp_setpoints.append("maximum_H2")
        else:  # t = 22200 to 33000
            corrected_ramp_setpoints.append("maximum_H2")
    return corrected_ramp_setpoints

# Generate the time set
time_set = generate_time_set()

# Generate corrected ramp setpoints based on the time set
corrected_ramp_setpoints = generate_corrected_ramp_setpoints(time_set)

# # Example output for verification
# print("First 10 Time Points and Setpoints:")
# for i in range(10):
#     print(f"Time: {time_set[i]}s, Setpoint: {corrected_ramp_setpoints[i]}")
#
# print("\nLast 5 Time Points and Setpoints:")
# for i in range(-5, 0):
#     print(f"Time: {time_set[i]}s, Setpoint: {corrected_ramp_setpoints[i]}")

# function to plot stress over time
import matplotlib.pyplot as plt

def plot_stress_over_time(model, iznode=None):
    times = list(model.fs.time)  # Replace 'model.fs.time' with your specific time indexing
    stresses = []

    # Assuming 'electrolyte_residual_thermal_stress' is indexed by time and possibly by 'iznode'
    if iznode is not None:
        stresses = [1e-6 * pyo.value(model.fs.soc_module.solid_oxide_cell.electrolyte_residual_thermal_stress[t, iznode]) for t in times]
    else:
        stresses = [1e-6 * pyo.value(model.fs.soc_module.solid_oxide_cell.electrolyte_residual_thermal_stress[t]) for t in times]  # Modify if your indexing differs

    plt.figure(figsize=(10, 5))
    plt.plot(times, stresses, marker='o', linestyle='-', color='b')
    plt.title('Electrolyte Residual Thermal Stress Over Time')
    plt.xlabel('Time')
    plt.ylabel('Stress (MPa)')
    plt.grid(True)
    plt.show()

# Example of using this function:
# plot_stress_over_time(m, iznode=0)  # Replace 'm' with your model object and specify the iznode if required

# def run_plot_sensitivity_analysys(m, solver):
# # deactivate heater and sweep heater duty constraints
# m.fs.feed_heater_duty_constraint_1.deactivate()
# m.fs.feed_heater_duty_constraint_2.deactivate()
# # Define the variable paths and their intervals
# min_value, max_value = 1e6, 10e6
# interval_change = 1e6
# ipopt_solver_count = 0  # Initialize the counter
# folder = "health_submodels/optimization/sensitivity_analysis"
#
# # Iterate over the intervals for both variables
# for i in range(1, 11):  # Loop over 10 intervals
#     # Calculate the new values for both variables
#     feed_heater_value = min_value + i * interval_change
#     sweep_heater_value = min_value + i * interval_change
#
#     if i == 1:
#         # load full discretization results from dynamic simulation
#         print("Loading full discretization results from existing file")
#         ms.from_json(m, fname="health_submodels/optimization" + "/" +
#                               "Dynamics_Full_Discretization_BaseCase.json.gz")
#     else:
#         # load the previous results from the previous interval
#         ms.from_json(m, fname=folder + "/" + "Sensitivity_analysis_" + str(i - 1) + ".json.gz")
#
#     ipopt_solver_count += 1
#     # Solve the model
#     solver.options['max_iter'] = 200
#     results_sens = solver.solve(m, tee=True)
#     pyo.assert_optimal_termination(results_sens)
#     print('solving model at iteration:', ipopt_solver_count)
#
#     # save the models to json file
#     ms.to_json(m, fname=folder + "/" + "Sensitivity_analysis_" + str(i) + ".json.gz")
#
#     # Save results of stress for each change of heater duty interval
#     # Initialize an empty DataFrame before the loop
#     full_df = pd.DataFrame()
#
#     for t in m.fs.time:
#         # Fix the values of both variables
#         m.fs.feed_heater.electric_heat_duty[t].fix(feed_heater_value)
#         m.fs.sweep_heater.electric_heat_duty[t].fix(sweep_heater_value)
#
#         # Record the stress of electrolyte at node z = 1
#         electrolyte_stress = pyo.value(
#             m.fs.soc_module.solid_oxide_cell.electrolyte_residual_thermal_stress[t, 1]) * 1e-6
#         df = pd.DataFrame({
#             # get the time values
#             'time': [pyo.value(t / 3600)],
#             'stress': [electrolyte_stress],
#             'feed_heater_duty': [feed_heater_value],
#             'sweep_heater_duty': [sweep_heater_value]
#         })
#
#         # Append to the full DataFrame using pd.concat
#         full_df = pd.concat([full_df, df], ignore_index=True)
#
#     # Write the full DataFrame to CSV after the loop
#     full_df.to_csv(
#         folder + "/" + f"stress_electrolyte_node_z1_{i}.csv",
#         index=False)
#
# # # Plot the stress of electrolyte at node z = 1 for each change of heater duty interval
# fig = plt.figure()
# ax = fig.subplots()
# for i in range(1, 11):
#     df = pd.read_csv(folder + "/" + f"stress_electrolyte_node_z1_{i}.csv")
#     ax.plot(df['time'], df['stress'], label=f'Heater Duty: {1e-6 * (min_value + i * interval_change)} MW')
# ax.set_xlabel(r'Time (h)')
# ax.set_ylabel(r'${\sigma}_{Electrolyte} (MPa)$')
# ax.set_title('Stress of Electrolyte at Node z = 1')
# ax.legend()
# plt.tight_layout()
# plt.savefig(folder + "/" + "stress_electrolyte_node_z1.png")
#
# plt.show()


