# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107, D200, D203, D400, D401, T201

__all__ = [
    "Model",
    "Options",
    "ScanResult",
    "elim_and_recalc",
    "rationalize_all_numbers",
    "strike_goldd",
]


from dataclasses import dataclass, field
from datetime import datetime
from math import ceil, inf
from pathlib import Path
from time import time
from typing import cast

import numpy as np
import symbtools as st
import sympy as sp
from sympy.matrices import zeros


@dataclass
class Model:
    x: list[list[sp.Symbol]]  # known variables
    p: list[list[sp.Symbol]]  # unknown parameters
    w: list  # unknown symbols
    u: list  # known symbols
    f: list  # dynamic equations
    h: list  # outputs


@dataclass
class Options:
    name: str
    check_obser = 1
    max_lie_time = inf
    nnz_der_u: list[float] = field(default_factory=lambda: [inf])
    nnz_der_w: list[float] = field(default_factory=lambda: [inf])
    prev_ident_pars: list = field(default_factory=list)


@dataclass
class ScanResult: ...


def rationalize_all_numbers(expr: sp.Matrix) -> sp.Matrix:
    numbers_atoms = list(expr.atoms(sp.Number))
    rationalized_number_tpls = [(n, sp.Rational(n)) for n in numbers_atoms]
    return cast(sp.Matrix, expr.subs(rationalized_number_tpls))


def elim_and_recalc(
    unmeas_xred_indices,
    rangoinicial,
    numonx,
    p,
    x,
    unidflag,
    w1vector,
    *args,
):
    numonx = rationalize_all_numbers(sp.Matrix(numonx))
    # Depending on the number of arguments you pass to the function, there are two cases:

    # called when there is no 'w'
    if len(args) == 0:
        pred = p
        xred = x
        wred = w1vector
        identifiables = []
        obs_states = []
        obs_inputs = []
        q = len(pred)
        n = len(xred)
        nw = len(wred)

    # called when there are 'w'
    if len(args) == 3:
        pred = p
        xred = x
        wred = w1vector
        identifiables = args[0]
        obs_states = args[1]
        obs_inputs = args[2]
        q = len(pred)
        n = len(xred)
        nw = len(wred)

    # before: q+n+nw; but with unknown inputs there may also be derivatives
    r = sp.shape(sp.Matrix(numonx))[1]
    new_ident_pars = identifiables
    new_nonid_pars = []
    new_obs_states = obs_states
    new_unobs_states = []
    new_obs_in = obs_inputs
    new_unobs_in = []

    # ========================================================================
    # ELIMINATE A PARAMETER:
    # ========================================================================
    # At each iteration we remove a different column (= parameter) from onx:
    for ind in range(q):  # for each parameter of p...
        if q <= 1:  # check if the parameter has already been marked as identifiable
            isidentifiable = pred[ind] in identifiables
        else:
            isidentifiable = any(pred[ind] in arr for arr in identifiables)
        if isidentifiable:
            print(
                f"\n Parameter {pred[ind]} has already been classified as identifiable."
            )
        else:
            indices = []
            for i in range(r):
                indices.append(i)
            indices.pop(n + ind)
            column_del_numonx = sp.Matrix(numonx).col(indices)  # one column is removed
            num_rank = st.generic_rank(
                sp.Matrix(column_del_numonx)
            )  # the range is calculated without that column
            if num_rank == rangoinicial:
                if unidflag == 1:
                    print(
                        f"\n    => Parameter {pred[ind]} is structurally unidentifiable"
                    )
                    new_nonid_pars.append(pred[ind])
                else:
                    print(
                        f"\n    => We cannot decide about parameter {pred[ind]} at the moment"
                    )
            else:
                print(f"\n    => Parameter {pred[ind]} is structurally identifiable")
                new_ident_pars.append(pred[ind])

    # ========================================================================
    # ELIMINATE A STATE:
    # ========================================================================
    # At each iteration we try removing a different state from 'xred':
    if options.checkObser == 1:
        for ind in range(len(unmeas_xred_indices)):  # for each unmeasured state
            original_index = unmeas_xred_indices[ind]
            if len(obs_states) <= 1:
                isobservable = xred[original_index] in obs_states
            else:
                isobservable = any(xred[original_index] in arr for arr in obs_states)
            if isobservable:
                print("\n State %s has already been classified as observable.".format())
            else:
                indices = []
                for i in range(r):
                    indices.append(i)
                indices.pop(original_index)  # remove the column that we want to check
                column_del_numonx = sp.Matrix(numonx).col(indices)
                num_rank = st.generic_rank(sp.Matrix(column_del_numonx))
                if num_rank == rangoinicial:
                    if unidflag == 1:
                        print(f"\n    => State {xred[original_index]} is unobservable")
                        new_unobs_states.append(xred[original_index])
                    else:  # if this function was called because the necessary number of derivatives was not calculated...
                        print(
                            f"\n    => We cannot decide about state {xred[original_index]} at the moment"
                        )
                else:
                    print(f"\n    => State {xred[original_index]} is observable")
                    new_obs_states.append(xred[original_index])

    # ========================================================================
    # ELIMINATE AN UNKNOWN INPUT:
    # ========================================================================
    # At each iteration we try removing a different column from onx:
    for ind in range(nw):  # for each unknown input...
        if (
            len(obs_inputs) <= 1
        ):  # check if the unknown input has already been marked as observable
            isobservable = wred[ind] in obs_inputs
        else:
            isobservable = any(wred[ind] in arr for arr in obs_inputs)
        if isobservable:
            print("\n Input %s has already been classified as observable.".format())
        else:
            indices = []
            for i in range(r):
                indices.append(i)
            indices.pop(n + q + ind)  # remove the column that we want to check
            column_del_numonx = sp.Matrix(numonx).col(indices)
            num_rank = st.generic_rank(sp.Matrix(column_del_numonx))
            if num_rank == rangoinicial:
                if unidflag == 1:
                    print(f"\n    => Input {wred[ind]} is unobservable")
                    new_unobs_in.append(wred[ind])
                else:
                    print(
                        f"\n    => We cannot decide about input {wred[ind]} at the moment"
                    )
            else:
                print(f"\n    => Input {wred[ind]} is observable")
                new_obs_in.append(wred[ind])
    return (
        new_ident_pars,
        new_nonid_pars,
        new_obs_states,
        new_unobs_states,
        new_obs_in,
        new_unobs_in,
    )


def strike_goldd(model: Model, options: Options) -> ScanResult:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize variables:
    identifiables = []  # identifiable parameters.
    nonidentif = []  # unidentifiable parameters.
    obs_states = []  # observable states.
    unobs_states = []  # unobservable states.
    obs_inputs = []  # observable inputs.
    unobs_inputs = []  # unobservable inputs.
    lastrank = None
    unidflag = 0
    skip_elim = 0
    is_fispo = 0

    # Dimensions of the problem:
    m = len(model.h)  # number of outputs
    n = len(model.x)  # number of states
    q = len(model.p)  # number of unknown parameters
    nw = len(model.w)
    r = n + q + nw  # number of unknown variables to observe / identify
    nd = ceil((r - m) / m)  # minimum number of Lie derivatives for Oi to have full rank

    # Check which states are directly measured, if any.
    # Basically it is checked if any state is directly on the output,
    # then that state is directly measurable.
    saidas = model.h if m == 1 else [model.h[i] for i in range(m)]
    estados = model.x if n == 1 else [model.x[i][0] for i in range(n)]
    ismeasured = [0 for i in range(n)]

    if len(saidas) == 1:
        for i in range(n):
            if estados[i] in saidas:
                ismeasured[i] = 1
    else:
        for i in range(n):
            if any(estados[i] in arr for arr in saidas):
                ismeasured[i] = 1

    measured_states_idx = [i for i in range(n) if ismeasured[i] == 1]
    unmeasured_states_idx = [i for i in range(n) if ismeasured[i] == 0]

    # names of the measured states
    meas_x = []
    if len(measured_states_idx) == 1 and n == 1:
        meas_x = estados
    if len(measured_states_idx) == 1 and n != 1:
        meas_x.append(estados[measured_states_idx[0]])
    if len(measured_states_idx) > 1:
        for i in range(len(measured_states_idx)):
            meas_x.append([estados[measured_states_idx[i]]])

    print(
        f"Building the observability-identifiability matrix requires at least {nd} Lie derivatives"
    )
    print("Calculating derivatives: ")

    ########################################################################
    # Check if the size of nnzDerU and nnzDerW are appropriate
    if len(model.u) > len(options.nnz_der_u):
        msg = """ The number of known inputs is higher than the size of nnzDerU and must have the same size.
        Go to the options file and modify it.
        For more information about the error see point 7 of the StrikePy instruction manual."""
        raise ValueError(msg)
    if len(model.w) > len(options.nnz_der_w):
        msg = """ The number of unknown inputs is higher than the size of nnzDerW and must have the same size.
        Go to the options file and modify it.
        For more information about the error see point 7 of the StrikePy instruction manual. """
        raise ValueError(msg)

    ########################################################################
    # Input derivates:

    # Create array of known inputs and set certain derivatives to zero:
    input_der = []
    if len(model.u) > 0:
        for ind_u in range(len(model.u)):  # create array of derivatives of the inputs
            if len(model.u) == 1:
                locals()[f"{model.u[ind_u]}"] = sp.Symbol(
                    f"{model.u[ind_u]}"
                )  # the first element is the underived input
                auxiliar = [locals()[f"{model.u[ind_u]}"]]
            else:
                locals()[f"{model.u[ind_u][0]}"] = sp.Symbol(
                    f"{model.u[ind_u][0]}"
                )  # the first element is the underived input
                auxiliar = [locals()[f"{model.u[ind_u][0]}"]]
            for k in range(nd):
                if len(model.u) == 1:
                    locals()[f"{model.u[ind_u]}_d{k + 1}"] = sp.Symbol(
                        f"{model.u[ind_u]}_d{k + 1}"
                    )
                    auxiliar.append(locals()[f"{model.u[ind_u]}_d{k + 1}"])
                else:
                    locals()[f"{model.u[ind_u][0]}_d{k + 1}"] = sp.Symbol(
                        f"{model.u[ind_u][0]}_d{k + 1}"
                    )
                    auxiliar.append(locals()[f"{model.u[ind_u][0]}_d{k + 1}"])
            if len(model.u) == 1:
                input_der = auxiliar
                if len(input_der) >= options.nnz_der_u[0] + 1:
                    for i in range(len(input_der[(options.nnz_der_u[0] + 1) :])):
                        input_der[(options.nnz_der_u[0] + 1) + i] = 0
            else:
                input_der.append(auxiliar)
                if len(input_der[0]) >= options.nnz_der_u[ind_u] + 1:
                    for i in range(len(input_der[0][(options.nnz_der_u[ind_u] + 1) :])):
                        input_der[ind_u][(options.nnz_der_u[ind_u] + 1) + i] = 0
    zero_input_der_dummy_name = sp.Symbol("zero_input_der_dummy_name")

    # Create array of unknown inputs and set certain derivatives to zero:
    w_der = []
    if len(model.w) > 0:
        for ind_w in range(len(model.w)):  # create array of derivatives of the inputs
            if len(model.w) == 1:
                locals()[f"{model.w[ind_w]}"] = sp.Symbol(
                    f"{model.w[ind_w]}"
                )  # the first element is the underived input
                auxiliar = [locals()[f"{model.w[ind_w]}"]]
            else:
                locals()[f"{model.w[ind_w][0]}"] = sp.Symbol(
                    f"{model.w[ind_w][0]}"
                )  # the first element is the underived input
                auxiliar = [locals()[f"{model.w[ind_w][0]}"]]
            for k in range(nd + 1):
                if len(model.w) == 1:
                    locals()[f"{model.w[ind_w]}_d{k + 1}"] = sp.Symbol(
                        f"{model.w[ind_w]}_d{k + 1}"
                    )
                    auxiliar.append(locals()[f"{model.w[ind_w]}_d{k + 1}"])
                else:
                    locals()[f"{model.w[ind_w][0]}_d{k + 1}"] = sp.Symbol(
                        f"{model.w[ind_w][0]}_d{k + 1}"
                    )
                    auxiliar.append(locals()[f"{model.w[ind_w][0]}_d{k + 1}"])
            if len(model.w) == 1:
                w_der = auxiliar
                if len(w_der) >= options.nnz_der_w[0] + 1:
                    for i in range(len(w_der[(options.nnz_der_w[0] + 1) :])):
                        w_der[(options.nnz_der_w[0] + 1) + i] = 0
            else:
                w_der.append(auxiliar)
                if len(w_der[0]) >= options.nnz_der_w[ind_w] + 1:
                    for i in range(len(w_der[0][(options.nnz_der_w[ind_w] + 1) :])):
                        w_der[ind_w][(options.nnzDerW[ind_w] + 1) + i] = 0

        if sp.shape(sp.Matrix(w_der).T)[0] == 1:
            w1vector = [[w_der[i]] for i in range(len(w_der) - 1)]
            w1vector_dot = [[w_der[i]] for i in range(1, len(w_der))]

        else:
            w1vector = []
            for k in range(sp.shape(sp.Matrix(w_der))[1] - 1):
                for i in w_der:
                    w1vector.append([i[k]])
            w1vector_dot = []
            for k in range(sp.shape(sp.Matrix(w_der))[1]):
                for i in w_der:
                    if k != 0:
                        w1vector_dot.append([i[k]])

        # -- Include as states only nonzero inputs / derivatives:
        nzi = [[fila] for fila in range(len(w1vector)) if w1vector[fila][0] != 0]
        nzj = [[1] for fila in range(len(w1vector)) if w1vector[fila][0] != 0]
        nz_w1vec = [
            w1vector[fila] for fila in range(len(w1vector)) if w1vector[fila][0] != 0
        ]
        w1vector = nz_w1vec
        w1vector_dot = w1vector_dot[0 : len(nzi)]

    else:
        w1vector = []
        w1vector_dot = []

    ########################################################################
    # Augment state vector, dynamics:
    if len(model.x) == 1:
        xaug = []
        xaug.append(model.x)
        xaug = np.append(xaug, model.p, axis=0)
        if len(w1vector) != 0:
            xaug = np.append(xaug, w1vector, axis=0)

        faug = []
        faug.append(model.f)
        faug = np.append(faug, zeros(len(model.p), 1), axis=0)
        if len(w1vector) != 0:
            faug = np.append(faug, w1vector_dot, axis=0)

    else:
        xaug = model.x
        xaug = np.append(xaug, model.p, axis=0)
        if len(w1vector) != 0:
            xaug = np.append(xaug, w1vector, axis=0)

        faug = model.f
        faug = np.append(faug, zeros(len(model.p), 1), axis=0)
        if len(w1vector) != 0:
            faug = np.append(faug, w1vector_dot, axis=0)
    ########################################################################
    # Build Oi:
    onx = np.array(zeros(m * (1 + nd), n + q + len(w1vector)))
    jacobiano = sp.Matrix(model.h).jacobian(xaug)
    onx[0 : len(model.h)] = np.array(
        jacobiano
    )  # first row(s) of onx (derivative of the output with respect to the vector states+unknown parameters).
    ind = 0  # Lie derivative index (sometimes called 'k')

    ########################################################################
    past_Lie = model.h
    extra_term = np.array(0)

    # loop as long as I don't complete the preset Lie derivatives or go over the maximum time set for each derivative
    while ind < nd:
        Lieh = sp.Matrix((onx[(ind * m) : (ind + 1) * m][:]).dot(faug))
        if ind > 0 and len(model.u) > 0:
            for i in range(ind):
                if len(model.u) == 1:
                    column = len(input_der) - 1
                    if i < column:
                        lo_u_der = input_der[i]
                        if lo_u_der == 0:
                            lo_u_der = zero_input_der_dummy_name
                        lo_u_der = np.array([lo_u_der])
                        hi_u_der = input_der[i + 1]
                        hi_u_der = sp.Matrix([hi_u_der])

                        intermedio = sp.Matrix([past_Lie]).jacobian(lo_u_der) * hi_u_der
                        if extra_term:
                            extra_term = extra_term + intermedio
                        else:
                            extra_term = intermedio
                else:
                    column = len(input_der[0]) - 1
                    if i < column:
                        lo_u_der = []
                        hi_u_der = []
                        for fila in input_der:
                            lo_u_der.append(fila[i])
                            hi_u_der.append(fila[i + 1])
                        for i in range(len(lo_u_der)):
                            if lo_u_der[i] == 0:
                                lo_u_der[i] = zero_input_der_dummy_name
                        lo_u_der = np.array(lo_u_der)
                        hi_u_der = sp.Matrix(hi_u_der)
                        intermedio = sp.Matrix([past_Lie]).jacobian(lo_u_der) * hi_u_der
                        if extra_term:
                            extra_term = extra_term + intermedio
                        else:
                            extra_term = intermedio
        ext_Lie = Lieh + extra_term if extra_term else Lieh
        past_Lie = ext_Lie
        onx[((ind + 1) * m) : (ind + 2) * m] = sp.Matrix(ext_Lie).jacobian(xaug)

        ind = ind + 1
        print(end=f" {ind}")

    if (
        ind == nd
    ):  # If I have done all the minimum derivatives to build onx (I have not exceeded the time limit)....
        increaseLie = 1
        while (
            increaseLie == 1
        ):  # while increaseLie is 1 I will increase the size of onx
            print(
                f"\n >>> Observability-Identifiability matrix built with {nd} Lie derivatives"
            )
            # =============================================================================================
            # The observability/identifiability matrix is saved in a .txt file

            with (
                results_dir / f"obs_ident_matrix_{options.name}_{nd}_Lie_deriv.txt"
            ).open("w") as file:
                file.write(f"onx = {onx.tolist()!s}")

            # =============================================================================================
            # Check identifiability by calculating rank:
            print(
                f" >>> Calculating rank of matrix with size {sp.shape(sp.Matrix(onx))[0]}x{sp.shape(sp.Matrix(onx))[1]}..."
            )
            rational_onx = rationalize_all_numbers(sp.Matrix(onx))
            rango = st.generic_rank(sp.Matrix(rational_onx))
            print(f"     Rank = {rango} (calculated in {toc} seconds)")
            if (
                rango == len(xaug)
            ):  # If the onx matrix already has full rank... all is observable and identifiable
                obs_states = model.x
                obs_inputs = model.w
                identifiables = model.p
                increaseLie = (
                    0  # stop increasing the number of onx rows with derivatives
                )

            else:  # With that number of Lie derivatives the array is not full rank.
                # ----------------------------------------------------------
                # If there are unknown inputs, we may want to check id/obs of (x,p,w) and not of dw/dt:
                if len(model.w) > 0:
                    [
                        identifiables,
                        nonidentif,
                        obs_states,
                        unobs_states,
                        obs_inputs,
                        unobs_inputs,
                    ] = elim_and_recalc(
                        unmeasured_states_idx,
                        rango,
                        onx,
                        model.p,
                        model.x,
                        unidflag,
                        w1vector,
                        identifiables,
                        obs_states,
                        obs_inputs,
                    )

                    # Check which unknown inputs are observable:
                    obs_in_no_der = []
                    if len(model.w) == 1 and len(obs_inputs) > 0:
                        if model.w == obs_inputs:
                            obs_in_no_der = model.w
                    if len(model.w) > 1 and len(obs_inputs) > 0:
                        for elemento in model.w:
                            if len(obs_inputs) == 1:
                                if elemento == obs_inputs:
                                    obs_in_no_der = elemento
                            else:
                                for input_ in obs_inputs:
                                    if elemento == input_:
                                        obs_in_no_der.append(elemento[0])
                    if (
                        len(identifiables) == len(model.p)
                        and len(obs_states) + len(meas_x) == len(model.x)
                        and len(obs_in_no_der) == len(model.w)
                    ):
                        obs_states = model.x
                        obs_inputs = obs_in_no_der
                        identifiables = model.p
                        increaseLie = 0  # -> with this we skip the next 'if' block and jump to the end of the algorithm
                        is_fispo = 1
                # ----------------------------------------------------------
                # If possible (& necessary), calculate one more Lie derivative and retry:
                if (
                    nd < len(xaug)
                    and lasttime < options.max_lie_time
                    and rango != lastrank
                    and increaseLie == 1
                ):
                    ind = nd
                    nd = (
                        nd + 1
                    )  # One is added to the number of derivatives already made
                    extra_term = np.array(0)  # reset for each new Lie derivative
                    # - Known input derivatives: ----------------------------------
                    if len(model.u) > 0:  # Extra terms of extended Lie derivatives
                        # may have to add extra input derivatives (note that 'nd' has grown):
                        input_der = []
                        for ind_u in range(
                            len(model.u)
                        ):  # create array of derivatives of the inputs
                            if len(model.u) == 1:
                                locals()[f"{model.u[ind_u]}"] = sp.Symbol(
                                    f"{model.u[ind_u]}"
                                )  # the first element is the underived input
                                auxiliar = [locals()[f"{model.u[ind_u]}"]]
                            else:
                                locals()[f"{model.u[ind_u][0]}"] = sp.Symbol(
                                    f"{model.u[ind_u][0]}"
                                )  # the first element is the underived input
                                auxiliar = [locals()[f"{model.u[ind_u][0]}"]]
                            for k in range(nd):
                                if len(model.u) == 1:
                                    locals()[f"{model.u[ind_u]}_d{k + 1}"] = sp.Symbol(
                                        f"{model.u[ind_u]}_d{k + 1}"
                                    )
                                    auxiliar.append(
                                        locals()[f"{model.u[ind_u]}_d{k + 1}"]
                                    )
                                else:
                                    locals()[f"{model.u[ind_u][0]}_d{k + 1}"] = (
                                        sp.Symbol(f"{model.u[ind_u][0]}_d{k + 1}")
                                    )
                                    auxiliar.append(
                                        locals()[f"{model.u[ind_u][0]}_d{k + 1}"]
                                    )
                            if len(model.u) == 1:
                                input_der = auxiliar
                                if len(input_der) >= options.nnz_der_u[0] + 1:
                                    for i in range(
                                        len(input_der[(options.nnz_der_u[0] + 1) :])
                                    ):
                                        input_der[(options.nnz_der_u[0] + 1) + i] = 0
                            else:
                                input_der.append(auxiliar)
                                if len(input_der[0]) >= options.nnz_der_u[ind_u] + 1:
                                    for i in range(
                                        len(
                                            input_der[0][
                                                (options.nnz_der_u[ind_u] + 1) :
                                            ]
                                        )
                                    ):
                                        input_der[ind_u][
                                            (options.nnzDerU[ind_u] + 1) + i
                                        ] = 0

                        for i in range(ind):
                            if len(model.u) == 1:
                                column = len(input_der) - 1
                                if i < column:
                                    lo_u_der = input_der[i]
                                    if lo_u_der == 0:
                                        lo_u_der = zero_input_der_dummy_name
                                    lo_u_der = np.array([lo_u_der])
                                    hi_u_der = input_der[i + 1]
                                    hi_u_der = sp.Matrix([hi_u_der])

                                    intermedio = (
                                        sp.Matrix([past_Lie]).jacobian(lo_u_der)
                                        * hi_u_der
                                    )
                                    if extra_term:
                                        extra_term = extra_term + intermedio
                                    else:
                                        extra_term = intermedio
                            else:
                                column = len(input_der[0]) - 1
                                if i < column:
                                    lo_u_der = []
                                    hi_u_der = []
                                    for fila in input_der:
                                        lo_u_der.append(fila[i])
                                        hi_u_der.append(fila[i + 1])
                                    for i in range(len(lo_u_der)):
                                        if lo_u_der[i] == 0:
                                            lo_u_der[i] = zero_input_der_dummy_name
                                    lo_u_der = np.array(lo_u_der)
                                    hi_u_der = sp.Matrix(hi_u_der)
                                    intermedio = (
                                        sp.Matrix([past_Lie]).jacobian(lo_u_der)
                                        * hi_u_der
                                    )
                                    if extra_term:
                                        extra_term = extra_term + intermedio
                                    else:
                                        extra_term = intermedio

                    # - Unknown input derivatives:----------------
                    # add new derivatives, if they are not zero:
                    if len(model.w) > 0:
                        prev_size = len(w1vector)
                        w_der = []
                        for ind_w in range(
                            len(model.w)
                        ):  # create array of derivatives of the inputs
                            if len(model.w) == 1:
                                locals()[f"{model.w[ind_w]}"] = sp.Symbol(
                                    f"{model.w[ind_w]}"
                                )  # the first element is the underived input
                                auxiliar = [locals()[f"{model.w[ind_w]}"]]
                            else:
                                locals()[f"{model.w[ind_w][0]}"] = sp.Symbol(
                                    f"{model.w[ind_w][0]}"
                                )  # the first element is the underived input
                                auxiliar = [locals()[f"{model.w[ind_w][0]}"]]
                            for k in range(nd + 1):
                                if len(model.w) == 1:
                                    locals()[f"{model.w[ind_w]}_d{k + 1}"] = sp.Symbol(
                                        f"{model.w[ind_w]}_d{k + 1}"
                                    )
                                    auxiliar.append(
                                        locals()[f"{model.w[ind_w]}_d{k + 1}"]
                                    )
                                else:
                                    locals()[f"{model.w[ind_w][0]}_d{k + 1}"] = (
                                        sp.Symbol(f"{model.w[ind_w][0]}_d{k + 1}")
                                    )
                                    auxiliar.append(
                                        locals()[f"{model.w[ind_w][0]}_d{k + 1}"]
                                    )
                            if len(model.w) == 1:
                                w_der = auxiliar
                                if len(w_der) >= options.nnz_der_w[0] + 1:
                                    for i in range(
                                        len(w_der[(options.nnz_der_w[0] + 1) :])
                                    ):
                                        w_der[(options.nnz_der_w[0] + 1) + i] = 0
                            else:
                                w_der.append(auxiliar)
                                if len(w_der[0]) >= options.nnz_der_w[ind_w] + 1:
                                    for i in range(
                                        len(w_der[0][(options.nnz_der_w[ind_w] + 1) :])
                                    ):
                                        w_der[ind_w][
                                            (options.nnzDerW[ind_w] + 1) + i
                                        ] = 0

                        if sp.shape(sp.Matrix(w_der).T)[0] == 1:
                            w1vector = []
                            for i in range(len(w_der) - 1):
                                w1vector.append([w_der[i]])
                            w1vector_dot = []
                            for i in range(len(w_der)):
                                if i != 0:
                                    w1vector_dot.append([w_der[i]])

                        else:
                            w1vector = []
                            for k in range(sp.shape(sp.Matrix(w_der))[1] - 1):
                                for i in w_der:
                                    w1vector.append([i[k]])
                            w1vector_dot = []
                            for k in range(sp.shape(sp.Matrix(w_der))[1]):
                                for i in w_der:
                                    if k != 0:
                                        w1vector_dot.append([i[k]])

                        # -- Include as states only nonzero inputs / derivatives:
                        nzi = []
                        for fila in range(len(w1vector)):
                            if w1vector[fila][0] != 0:
                                nzi.append([fila])
                        nzj = []
                        for fila in range(len(w1vector)):
                            if w1vector[fila][0] != 0:
                                nzj.append([1])
                        nz_w1vec = []
                        for fila in range(len(w1vector)):
                            if w1vector[fila][0] != 0:
                                nz_w1vec.append(w1vector[fila])
                        w1vector = nz_w1vec
                        w1vector_dot = w1vector_dot[0 : len(nzi)]

                        ########################################################################
                        # Augment state vector, dynamics:
                        if len(model.x) == 1:
                            xaug = []
                            xaug.append(model.x)
                            xaug = np.append(xaug, model.p, axis=0)
                            if len(w1vector) != 0:
                                xaug = np.append(xaug, w1vector, axis=0)

                            faug = []
                            faug.append(model.f)
                            faug = np.append(faug, zeros(len(model.p), 1), axis=0)
                            if len(w1vector) != 0:
                                faug = np.append(faug, w1vector_dot, axis=0)

                        else:
                            xaug = model.x
                            xaug = np.append(xaug, model.p, axis=0)
                            if len(w1vector) != 0:
                                xaug = np.append(xaug, w1vector, axis=0)

                            faug = model.f
                            faug = np.append(faug, zeros(len(model.p), 1), axis=0)
                            if len(w1vector) != 0:
                                faug = np.append(faug, w1vector_dot, axis=0)
                        ########################################################################
                        # -- Augment size of the Obs-Id matrix if needed:
                        new_size = len(w1vector)
                        onx = np.append(
                            onx, zeros((ind + 1) * m, new_size - prev_size), axis=1
                        )
                    ########################################################################
                    newLie = sp.Matrix((onx[(ind * m) : (ind + 1) * m][:]).dot(faug))
                    past_Lie = newLie + extra_term if extra_term else newLie
                    newOnx = sp.Matrix(past_Lie).jacobian(xaug)
                    onx = np.append(onx, newOnx, axis=0)

                    lastrank = rango

                # If that is not possible, there are several possible causes:
                # This is the case when you have onx with all possible derivatives done and it is not full rank, the maximum time for the next derivative has passed
                # or the matrix no longer increases in rank as derivatives are increased.
                else:
                    if nd >= len(
                        xaug
                    ):  # The maximum number of Lie derivatives has been reached
                        unidflag = 1
                        print(
                            "\n >>> The model is structurally unidentifiable as a whole"
                        )
                    elif rango == lastrank:
                        onx = onx[0 : (-1 - (m - 1))]
                        nd = (
                            nd - 1
                        )  # It is indicated that the number of derivatives needed was one less than the number of derivatives made
                        unidflag = 1
                    elif lasttime >= options.max_lie_time:
                        print(
                            "\n => More Lie derivatives would be needed to see if the model is structurally unidentifiable as a whole."
                        )
                        print(
                            "    However, the maximum computation time allowed for calculating each of them has been reached."
                        )
                        print(
                            f"    You can increase it by changing <<maxLietime>> in options (currently maxLietime = {options.max_lie_time})"
                        )
                        unidflag = 0
                    if skip_elim == 0 and is_fispo == 0:
                        # Eliminate columns one by one to check identifiability of the associated parameters:
                        [
                            identifiables,
                            nonidentif,
                            obs_states,
                            unobs_states,
                            obs_inputs,
                            unobs_inputs,
                        ] = elim_and_recalc(
                            unmeasured_states_idx,
                            rango,
                            onx,
                            model.p,
                            model.x,
                            unidflag,
                            w1vector,
                            identifiables,
                            obs_states,
                            obs_inputs,
                        )

                        # Check which unknown inputs are observable:
                        obs_in_no_der = []
                        if (
                            len(model.w) == 1
                            and len(obs_inputs) > 0
                            and model.w == obs_inputs
                        ):
                            obs_in_no_der = model.w
                        if len(model.w) > 1 and len(obs_inputs) > 0:
                            for elemento in model.w:  # for each unknown input
                                if len(obs_inputs) == 1:
                                    if elemento == obs_inputs:
                                        obs_in_no_der = elemento
                                else:
                                    for input in obs_inputs:
                                        if elemento == input:
                                            obs_in_no_der.append(elemento[0])

                        if (
                            len(identifiables) == len(model.p)
                            and (len(obs_states) + len(meas_x)) == len(model.x)
                            and len(obs_in_no_der) == len(model.w)
                        ):
                            obs_states = model.x
                            obs_inputs = obs_in_no_der
                            identifiables = model.p
                            increaseLie = 0  # -> with this we skip the next 'if' block and jump to the end of the algorithm
                            is_fispo = 1
                        increaseLie = 0

    else:  # If the maxLietime has been reached, but the minimum of Lie derivatives has not been calculated:
        print("\n => More Lie derivatives would be needed to analyse the model.")
        print(
            "    However, the maximum computation time allowed for calculating each of them has been reached."
        )
        print(
            f"    You can increase it by changing <<maxLietime>> in options (currently maxLietime = {options.max_lie_time})"
        )
        print(
            f"\n >>> Calculating rank of matrix with size {sp.shape(sp.Matrix(onx))[0]}x{sp.shape(sp.Matrix(onx))[1]}..."
        )
        # =============================================================================================
        # The observability/identifiability matrix is saved in a .txt file
        file_path = results_dir / f"obs_ident_matrix_{options.name}_{nd}_Lie_deriv.txt"
        with file_path.open("w") as file:
            file.write(f"onx = {onx.tolist()!s}")

        # =============================================================================================
        rational_onx = rationalize_all_numbers(sp.Matrix(onx))
        rango = st.generic_rank(sp.Matrix(rational_onx))

        print(f"\n     Rank = {rango}")
        (
            identifiables,
            nonidentif,
            obs_states,
            unobs_states,
            obs_inputs,
            unobs_inputs,
        ) = elim_and_recalc(
            unmeasured_states_idx, rango, onx, identifiables, obs_states, obs_inputs
        )
    # ======================================================================================
    # Build the vectors of identifiable / unidentifiable parameters, and of observable / unobservable states and inputs:
    if len(identifiables) != 0:
        p_id = sp.Matrix(identifiables).T
        p_id = np.array(p_id).tolist()[0]
    else:
        p_id = []

    if len(nonidentif) != 0:
        p_un = sp.Matrix(nonidentif).T
        p_un = np.array(p_un).tolist()[0]
    else:
        p_un = []

    if len(obs_states) != 0:
        obs_states = sp.Matrix(obs_states).T
        obs_states = np.array(obs_states).tolist()[0]

    if len(unobs_states) != 0:
        unobs_states = sp.Matrix(unobs_states).T
        unobs_states = np.array(unobs_states).tolist()[0]

    if len(obs_inputs) != 0:
        obs_inputs = sp.Matrix(obs_inputs).T
        obs_inputs = np.array(obs_inputs).tolist()[0]

    if len(unobs_inputs) != 0:
        unobs_inputs = sp.Matrix(unobs_inputs).T
        unobs_inputs = np.array(unobs_inputs).tolist()[0]
    # ========================================================================================
    # The observability/identifiability matrix is saved in a .txt file

    file_path = results_dir / f"obs_ident_matrix_{options.name}_{nd}_Lie_deriv.txt"
    with file_path.open("w") as file:
        file.write(f"onx = {onx.tolist()!s}")

    # The summary of the results is saved in a .txt file
    file_path = (
        results_dir
        / f"id_results_{options.name}_{datetime.today().strftime('%d-%m-%Y')}.txt"
    )
    with file_path.open("w") as file:
        file.write("\n RESULTS SUMMARY:")

    # Report results:
    # result
    # fispo: bool

    print("\n ------------------------ ")
    print("     RESULTS SUMMARY:")
    print(" ------------------------ ")
    if (
        len(p_id) == len(model.p)
        and len(obs_states) == len(model.x)
        and len(obs_inputs) == len(model.w)
    ):
        print("\n >>> The model is Fully Input-State-Parameter Observable (FISPO):")
        if len(model.w) > 0:
            print("\n     All its unknown inputs are observable.")
            file.write("\n     All its unknown inputs are observable.")
        print("\n     All its states are observable.")
        print("\n     All its parameters are locally structurally identifiable.")
    else:
        if len(p_id) == len(model.p):
            print("\n >>> The model is structurally identifiable:")
            print("\n     All its parameters are structurally identifiable.")
            file.write(
                "\n >>> The model is structurally identifiable:\n     All its parameters are structurally identifiable."
            )
        elif unidflag:
            print("\n >>> The model is structurally unidentifiable.")
            print(f"\n >>> These parameters are identifiable:\n      {p_id} ")
            print(f"\n >>> These parameters are unidentifiable:\n      {p_un}")
            file.write(
                f"\n >>> The model is structurally unidentifiable.\n >>> These parameters are identifiable:\n      {p_id}\n >>> These parameters are unidentifiable:\n      {p_un}"
            )
        else:
            print(f"\n >>> These parameters are identifiable:\n      {p_id}")
            file.write(f"\n >>> These parameters are identifiable:\n      {p_id}")

        if len(obs_states) > 0:
            print(
                f"\n >>> These states are observable (and their initial conditions, if unknown, are identifiable):\n      {obs_states}"
            )
            file.write(
                f"\n >>> These states are observable (and their initial conditions, if unknown, are identifiable):\n      {obs_states}"
            )
        if len(unobs_states) > 0:
            print(
                f"\n >>> These states are unobservable (and their initial conditions, if unknown, are unidentifiable):\n      {unobs_states}"
            )
            file.write(
                f"\n >>> These states are unobservable (and their initial conditions, if unknown, are unidentifiable):\n      {unobs_states}"
            )

        if len(meas_x) != 0:  # para mostrarlo en una fila, como el resto
            meas_x = sp.Matrix(meas_x).T
            meas_x = np.array(meas_x).tolist()[0]
        else:
            meas_x = []

        if len(meas_x) > 0:
            print(f"\n >>> These states are directly measured:\n      {meas_x}")
            file.write(f"\n >>> These states are directly measured:\n      {meas_x}")
        if len(obs_inputs) > 0:
            print(f"\n >>> These unmeasured inputs are observable:\n      {obs_inputs}")
            file.write(
                f"\n >>> These unmeasured inputs are observable:\n      {obs_inputs}"
            )
        if len(unobs_inputs) > 0:
            print(
                f"\n >>> These unmeasured inputs are unobservable:\n      {unobs_inputs}"
            )
            file.write(
                f"\n >>> These unmeasured inputs are unobservable:\n      {unobs_inputs}"
            )
        if len(model.u) > 0:
            print(f"\n >>> These inputs are known:\n      {model.u}")
            file.write(f"\n >>> These inputs are known:\n      {model.u}")

    return ScanResult()
