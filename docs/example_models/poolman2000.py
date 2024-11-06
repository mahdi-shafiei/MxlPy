from __future__ import annotations

from modelbase2 import Model


def rapid_equilibrium_1_1(
    s1: float,
    p1: float,
    kre: float,
    q: float,
) -> float:
    return kre * (s1 - p1 / q)


def rapid_equilibrium_2_1(
    s1: float,
    s2: float,
    p1: float,
    kre: float,
    q: float,
) -> float:
    return kre * (s1 * s2 - p1 / q)


def rapid_equilibrium_2_2(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    kre: float,
    q: float,
) -> float:
    return kre * (s1 * s2 - (p1 * p2) / q)


def v_out(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
) -> float:
    return (vmax_efflux * s1) / (n_total * k_efflux)


def v1(
    RUBP: float,
    PGA: float,
    FBP: float,
    SBP: float,
    P: float,
    V1: float,
    Km1: float,
    Ki11: float,
    Ki12: float,
    Ki13: float,
    Ki14: float,
    Ki15: float,
    NADPH_pool: float,
) -> float:
    return (V1 * RUBP) / (
        RUBP
        + Km1
        * (
            1
            + (PGA / Ki11)
            + (FBP / Ki12)
            + (SBP / Ki13)
            + (P / Ki14)
            + (NADPH_pool / Ki15)
        )
    )


def v3(
    BPGA: float,
    GAP: float,
    phosphate_pool: float,
    proton_pool_stroma: float,
    NADPH_pool: float,
    NADP_pool: float,
    kRE: float,
    q3: float,
) -> float:
    return kRE * (
        (NADPH_pool * BPGA * proton_pool_stroma)
        - (1 / q3) * (GAP * NADP_pool * phosphate_pool)
    )


def v6(
    FBP: float,
    F6P: float,
    P: float,
    V6: float,
    Km6: float,
    Ki61: float,
    Ki62: float,
) -> float:
    return (V6 * FBP) / (FBP + Km6 * (1 + (F6P / Ki61) + (P / Ki62)))


def v9(
    SBP: float,
    P: float,
    V9: float,
    Km9: float,
    Ki9: float,
) -> float:
    return (V9 * SBP) / (SBP + Km9 * (1 + (P / Ki9)))


def v13(
    RU5P: float,
    ATP: float,
    Phosphate_pool: float,
    PGA: float,
    RUBP: float,
    ADP: float,
    V13: float,
    Km131: float,
    Km132: float,
    Ki131: float,
    Ki132: float,
    Ki133: float,
    Ki134: float,
    Ki135: float,
) -> float:
    return (V13 * RU5P * ATP) / (
        (RU5P + Km131 * (1 + (PGA / Ki131) + (RUBP / Ki132) + (Phosphate_pool / Ki133)))
        * (ATP * (1 + (ADP / Ki134)) + Km132 * (1 + (ADP / Ki135)))
    )


def v16(
    ADP: float,
    Phosphate_i: float,
    V16: float,
    Km161: float,
    Km162: float,
) -> float:
    return (V16 * ADP * Phosphate_i) / ((ADP + Km161) * (Phosphate_i + Km162))


def vStarchProduction(
    G1P: float,
    ATP: float,
    ADP: float,
    Phosphate_pool: float,
    PGA: float,
    F6P: float,
    FBP: float,
    Vst: float,
    Kmst1: float,
    Kmst2: float,
    Kist: float,
    Kast1: float,
    Kast2: float,
    Kast3: float,
) -> float:
    return (Vst * G1P * ATP) / (
        (G1P + Kmst1)
        * (
            (1 + (ADP / Kist)) * (ATP + Kmst2)
            + ((Kmst2 * Phosphate_pool) / (Kast1 * PGA + Kast2 * F6P + Kast3 * FBP))
        )
    )


def ADP(
    ATP: float,
    AP_total: float,
) -> float:
    return AP_total - ATP


def P_i(
    PGA: float,
    BPGA: float,
    GAP: float,
    DHAP: float,
    FBP: float,
    F6P: float,
    G6P: float,
    G1P: float,
    SBP: float,
    S7P: float,
    E4P: float,
    X5P: float,
    R5P: float,
    RUBP: float,
    RU5P: float,
    ATP: float,
    phosphate_total: float,
) -> float:
    return phosphate_total - (
        PGA
        + 2 * BPGA
        + GAP
        + DHAP
        + 2 * FBP
        + F6P
        + G6P
        + G1P
        + 2 * SBP
        + S7P
        + E4P
        + X5P
        + R5P
        + 2 * RUBP
        + RU5P
        + ATP
    )


def N(
    Phosphate_pool: float,
    PGA: float,
    GAP: float,
    DHAP: float,
    Kpxt: float,
    Pext: float,
    Kpi: float,
    Kpga: float,
    Kgap: float,
    Kdhap: float,
) -> float:
    return 1 + (1 + (Kpxt / Pext)) * (
        (Phosphate_pool / Kpi) + (PGA / Kpga) + (GAP / Kgap) + (DHAP / Kdhap)
    )


parameters = {
    "Vmax_1": 2.72,  # [mM/s], Pettersson 1988
    "Vmax_6": 1.6,  # [mM/s], Pettersson 1988
    "Vmax_9": 0.32,  # [mM/s], Pettersson 1988
    "Vmax_13": 8.0,  # [mM/s], Pettersson 1988
    "Vmax_16": 2.8,  # [mM/s], Pettersson 1988
    "Vmax_starch": 0.32,  # [mM/s], Pettersson 1988
    "Vmax_efflux": 2.0,  # [mM/s], Pettersson 1988
    "Km_1": 0.02,  # [mM], Pettersson 1988
    "Km_6": 0.03,  # [mM], Pettersson 1988
    "Km_9": 0.013,  # [mM], Pettersson 1988
    "Km_13_1": 0.05,  # [mM], Pettersson 1988
    "Km_13_2": 0.05,  # [mM], Pettersson 1988
    "Km_16_1": 0.014,  # [mM], Pettersson 1988
    "Km_16_2": 0.3,  # [mM], Pettersson 1988
    "Km_starch_1": 0.08,  # [mM], Pettersson 1988
    "Km_starch_2": 0.08,  # [mM], Pettersson 1988
    "K_pga": 0.25,  # [mM], Pettersson 1988
    "K_gap": 0.075,  # [mM], Pettersson 1988
    "K_dhap": 0.077,  # [mM], Pettersson 1988
    "K_pi": 0.63,  # [mM], Pettersson 1988
    "K_pxt": 0.74,  # [mM], Pettersson 1988
    "Ki_1_1": 0.04,  # [mM], Pettersson 1988
    "Ki_1_2": 0.04,  # [mM], Pettersson 1988
    "Ki_1_3": 0.075,  # [mM], Pettersson 1988
    "Ki_1_4": 0.9,  # [mM], Pettersson 1988
    "Ki_1_5": 0.07,  # [mM], Pettersson 1988
    "Ki_6_1": 0.7,  # [mM], Pettersson 1988
    "Ki_6_2": 12.0,  # [mM], Pettersson 1988
    "Ki_9": 12.0,  # [mM], Pettersson 1988
    "Ki_13_1": 2.0,  # [mM], Pettersson 1988
    "Ki_13_2": 0.7,  # [mM], Pettersson 1988
    "Ki_13_3": 4.0,  # [mM], Pettersson 1988
    "Ki_13_4": 2.5,  # [mM], Pettersson 1988
    "Ki_13_5": 0.4,  # [mM], Pettersson 1988
    "Ki_starch": 10.0,  # [mM], Pettersson 1988
    "Ka_starch_1": 0.1,  # [mM], Pettersson 1988
    "Ka_starch_2": 0.02,  # [mM], Pettersson 1988
    "Ka_starch_3": 0.02,  # [mM], Pettersson 1988
    "k_rapid_eq": 800000000.0,  # Rapid Equilibrium speed
    "q2": 0.00031,  # [], Pettersson 1988
    "q3": 16000000.0,  # [], Pettersson 1988
    "q4": 22.0,  # [], Pettersson 1988
    "q5": 7.1,  # [1/mM]], Pettersson 1988
    "q7": 0.084,  # [], Pettersson 1988
    "q8": 13.0,  # [1/mM]], Pettersson 1988
    "q10": 0.85,  # [], Pettersson 1988
    "q11": 0.4,  # [], Pettersson 1988
    "q12": 0.67,  # [], Pettersson 1988
    "q14": 2.3,  # [], Pettersson 1988
    "q15": 0.058,  # [], Pettersson 1988
    "CO2": 0.2,  # [mM], Pettersson 1988
    "Phosphate_total": 15.0,  # [mM], Pettersson 1988
    "AP_total": 0.5,  # [mM], Pettersson 1988
    "N_total": 0.5,  # [mM], Pettersson 1988
    "Phosphate_pool_ext": 0.5,  # [mM], Pettersson 1988
    "pH_medium": 7.6,  # [], Pettersson 1988
    "pH_stroma": 7.9,  # [], Pettersson 1988
    "proton_pool_stroma": 1.2589254117941661e-05,  # [mM], Pettersson 1988
    "NADPH_pool": 0.21,  # [mM], Pettersson 1988
    "NADP_pool": 0.29,  # [mM], Pettersson 1988
}

variables = {
    "PGA": 0.6437280277346407,
    "BPGA": 0.001360476366780556,
    "GAP": 0.011274125311289358,
    "DHAP": 0.24803073890728228,
    "FBP": 0.019853938009873073,
    "F6P": 1.0950701164493861,
    "G6P": 2.5186612678035734,
    "G1P": 0.14608235353185037,
    "SBP": 0.09193353265673603,
    "S7P": 0.23124426886012006,
    "E4P": 0.028511831060903877,
    "X5P": 0.036372985623662736,
    "R5P": 0.06092475016463224,
    "RUBP": 0.24993009253928708,
    "RU5P": 0.02436989993734177,
    "ATP": 0.43604115800259613,
}


def get_model() -> Model:
    model = Model()
    model.add_parameters(parameters)
    model.add_variables(variables)

    model.add_derived(
        name="ADP",
        fn=ADP,
        args=["ATP", "AP_total"],
    )

    model.add_derived(
        name="Phosphate_pool",
        fn=P_i,
        args=[
            "PGA",
            "BPGA",
            "GAP",
            "DHAP",
            "FBP",
            "F6P",
            "G6P",
            "G1P",
            "SBP",
            "S7P",
            "E4P",
            "X5P",
            "R5P",
            "RUBP",
            "RU5P",
            "ATP",
            "Phosphate_total",
        ],
    )

    model.add_derived(
        name="N_pool",
        fn=N,
        args=[
            "Phosphate_pool",
            "PGA",
            "GAP",
            "DHAP",
            "K_pxt",
            "Phosphate_pool_ext",
            "K_pi",
            "K_pga",
            "K_gap",
            "K_dhap",
        ],
    )

    model.add_reaction(
        name="v1",
        fn=v1,
        stoichiometry={"RUBP": -1, "PGA": 2},
        args=[
            "RUBP",
            "PGA",
            "FBP",
            "SBP",
            "Phosphate_pool",
            "Vmax_1",
            "Km_1",
            "Ki_1_1",
            "Ki_1_2",
            "Ki_1_3",
            "Ki_1_4",
            "Ki_1_5",
            "NADPH_pool",
        ],
    )
    model.add_reaction(
        name="v2",
        fn=rapid_equilibrium_2_2,
        stoichiometry={"PGA": -1, "ATP": -1, "BPGA": 1},
        args=[
            "PGA",
            "ATP",
            "BPGA",
            "ADP",
            "k_rapid_eq",
            "q2",
        ],
    )
    model.add_reaction(
        name="v3",
        fn=v3,
        stoichiometry={"BPGA": -1, "GAP": 1},
        args=[
            "BPGA",
            "GAP",
            "Phosphate_pool",
            "proton_pool_stroma",
            "NADPH_pool",
            "NADP_pool",
            "k_rapid_eq",
            "q3",
        ],
    )
    model.add_reaction(
        name="v4",
        fn=rapid_equilibrium_1_1,
        stoichiometry={"GAP": -1, "DHAP": 1},
        args=[
            "GAP",
            "DHAP",
            "k_rapid_eq",
            "q4",
        ],
    )
    model.add_reaction(
        name="v5",
        fn=rapid_equilibrium_2_1,
        stoichiometry={"GAP": -1, "DHAP": -1, "FBP": 1},
        args=[
            "GAP",
            "DHAP",
            "FBP",
            "k_rapid_eq",
            "q5",
        ],
    )
    model.add_reaction(
        name="v6",
        fn=v6,
        stoichiometry={"FBP": -1, "F6P": 1},
        args=[
            "FBP",
            "F6P",
            "Phosphate_pool",
            "Vmax_6",
            "Km_6",
            "Ki_6_1",
            "Ki_6_2",
        ],
    )
    model.add_reaction(
        name="v7",
        fn=rapid_equilibrium_2_2,
        stoichiometry={"GAP": -1, "F6P": -1, "E4P": 1, "X5P": 1},
        args=[
            "GAP",
            "F6P",
            "E4P",
            "X5P",
            "k_rapid_eq",
            "q7",
        ],
    )
    model.add_reaction(
        name="v8",
        fn=rapid_equilibrium_2_1,
        stoichiometry={"DHAP": -1, "E4P": -1, "SBP": 1},
        args=[
            "DHAP",
            "E4P",
            "SBP",
            "k_rapid_eq",
            "q8",
        ],
    )
    model.add_reaction(
        name="v9",
        fn=v9,
        stoichiometry={"SBP": -1, "S7P": 1},
        args=[
            "SBP",
            "Phosphate_pool",
            "Vmax_9",
            "Km_9",
            "Ki_9",
        ],
    )
    model.add_reaction(
        name="v10",
        fn=rapid_equilibrium_2_2,
        stoichiometry={"GAP": -1, "S7P": -1, "X5P": 1, "R5P": 1},
        args=[
            "GAP",
            "S7P",
            "X5P",
            "R5P",
            "k_rapid_eq",
            "q10",
        ],
    )
    model.add_reaction(
        name="v11",
        fn=rapid_equilibrium_1_1,
        stoichiometry={"R5P": -1, "RU5P": 1},
        args=[
            "R5P",
            "RU5P",
            "k_rapid_eq",
            "q11",
        ],
    )
    model.add_reaction(
        name="v12",
        fn=rapid_equilibrium_1_1,
        stoichiometry={"X5P": -1, "RU5P": 1},
        args=[
            "X5P",
            "RU5P",
            "k_rapid_eq",
            "q12",
        ],
    )
    model.add_reaction(
        name="v13",
        fn=v13,
        stoichiometry={"RU5P": -1, "ATP": -1, "RUBP": 1},
        args=[
            "RU5P",
            "ATP",
            "Phosphate_pool",
            "PGA",
            "RUBP",
            "ADP",
            "Vmax_13",
            "Km_13_1",
            "Km_13_2",
            "Ki_13_1",
            "Ki_13_2",
            "Ki_13_3",
            "Ki_13_4",
            "Ki_13_5",
        ],
    )
    model.add_reaction(
        name="v14",
        fn=rapid_equilibrium_1_1,
        stoichiometry={"F6P": -1, "G6P": 1},
        args=[
            "F6P",
            "G6P",
            "k_rapid_eq",
            "q14",
        ],
    )
    model.add_reaction(
        name="v15",
        fn=rapid_equilibrium_1_1,
        stoichiometry={"G6P": -1, "G1P": 1},
        args=[
            "G6P",
            "G1P",
            "k_rapid_eq",
            "q15",
        ],
    )
    model.add_reaction(
        name="v16",
        fn=v16,
        stoichiometry={"ATP": 1},
        args=[
            "ADP",
            "Phosphate_pool",
            "Vmax_16",
            "Km_16_1",
            "Km_16_2",
        ],
    )
    model.add_reaction(
        name="vPGA_out",
        fn=v_out,
        stoichiometry={"PGA": -1},
        args=[
            "PGA",
            "N_pool",
            "Vmax_efflux",
            "K_pga",
        ],
    )
    model.add_reaction(
        name="vGAP_out",
        fn=v_out,
        stoichiometry={"GAP": -1},
        args=[
            "GAP",
            "N_pool",
            "Vmax_efflux",
            "K_gap",
        ],
    )
    model.add_reaction(
        name="vDHAP_out",
        fn=v_out,
        stoichiometry={"DHAP": -1},
        args=[
            "DHAP",
            "N_pool",
            "Vmax_efflux",
            "K_dhap",
        ],
    )
    model.add_reaction(
        name="vSt",
        fn=vStarchProduction,
        stoichiometry={"G1P": -1, "ATP": -1},
        args=[
            "G1P",
            "ATP",
            "ADP",
            "Phosphate_pool",
            "PGA",
            "F6P",
            "FBP",
            "Vmax_starch",
            "Km_starch_1",
            "Km_starch_2",
            "Ki_starch",
            "Ka_starch_1",
            "Ka_starch_2",
            "Ka_starch_3",
        ],
    )
    return model
