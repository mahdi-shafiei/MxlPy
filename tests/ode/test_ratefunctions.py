# type: ignore

from __future__ import annotations

import unittest
from math import factorial

from modelbase.ode import ratefunctions as rf


class RateFunctionTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(rf.constant(k=1), 1)

    def test_mass_action_1(self):
        self.assertEqual(rf.mass_action_1(S1=1, k_fwd=2), factorial(1) * 2)

    def test_mass_action_2(self):
        self.assertEqual(rf.mass_action_2(S1=1, S2=2, k_fwd=2), factorial(2) * 2)

    def test_mass_action_3(self):
        self.assertEqual(rf.mass_action_3(S1=1, S2=2, S3=3, k_fwd=2), factorial(3) * 2)

    def test_mass_action_4(self):
        self.assertEqual(
            rf.mass_action_4(S1=1, S2=2, S3=3, S4=4, k_fwd=2), factorial(4) * 2
        )

    def test_mass_action_variable(self):
        self.assertEqual(rf.mass_action_variable(1, 2), factorial(1) * 2)
        self.assertEqual(rf.mass_action_variable(1, 2, 2), factorial(2) * 2)
        self.assertEqual(rf.mass_action_variable(1, 2, 3, 2), factorial(3) * 2)
        self.assertEqual(rf.mass_action_variable(1, 2, 3, 4, 2), factorial(4) * 2)
        self.assertEqual(rf.mass_action_variable(1, 2, 3, 4, 5, 2), factorial(5) * 2)

    def test_reversible_mass_action_1_1(self):
        self.assertEqual(
            rf.reversible_mass_action_1_1(S1=1, P1=1, k_fwd=2, k_bwd=0.5),
            factorial(1) * 2 - factorial(1) * 0.5,
        )

    def test_reversible_mass_action_1_2(self):
        self.assertEqual(
            rf.reversible_mass_action_1_2(S1=1, P1=1, P2=2, k_fwd=2, k_bwd=0.5),
            factorial(1) * 2 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_1_3(self):
        self.assertEqual(
            rf.reversible_mass_action_1_3(S1=1, P1=1, P2=2, P3=3, k_fwd=2, k_bwd=0.5),
            factorial(1) * 2 - factorial(3) * 0.5,
        )

    def test_reversible_mass_action_1_4(self):
        self.assertEqual(
            rf.reversible_mass_action_1_4(
                S1=1, P1=1, P2=2, P3=3, P4=4, k_fwd=2, k_bwd=0.5
            ),
            factorial(1) * 2 - factorial(4) * 0.5,
        )

    def test_reversible_mass_action_2_1(self):
        self.assertEqual(
            rf.reversible_mass_action_2_1(S1=1, S2=2, P1=1, k_fwd=2, k_bwd=0.5),
            factorial(2) * 2 - factorial(1) * 0.5,
        )

    def test_reversible_mass_action_2_2(self):
        self.assertEqual(
            rf.reversible_mass_action_2_2(S1=1, S2=2, P1=1, P2=2, k_fwd=2, k_bwd=0.5),
            factorial(2) * 2 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_2_3(self):
        self.assertEqual(
            rf.reversible_mass_action_2_3(
                S1=1, S2=2, P1=1, P2=2, P3=3, k_fwd=2, k_bwd=0.5
            ),
            factorial(2) * 2 - factorial(3) * 0.5,
        )

    def test_reversible_mass_action_2_4(self):
        self.assertEqual(
            rf.reversible_mass_action_2_4(
                S1=1, S2=2, P1=1, P2=2, P3=3, P4=4, k_fwd=2, k_bwd=0.5
            ),
            factorial(2) * 2 - factorial(4) * 0.5,
        )

    def test_reversible_mass_action_3_1(self):
        self.assertEqual(
            rf.reversible_mass_action_3_1(S1=1, S2=2, S3=3, P1=1, k_fwd=2, k_bwd=0.5),
            factorial(3) * 2 - factorial(1) * 0.5,
        )

    def test_reversible_mass_action_3_2(self):
        self.assertEqual(
            rf.reversible_mass_action_3_2(
                S1=1, S2=2, S3=3, P1=1, P2=2, k_fwd=2, k_bwd=0.5
            ),
            factorial(3) * 2 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_3_3(self):
        self.assertEqual(
            rf.reversible_mass_action_3_3(
                S1=1, S2=2, S3=3, P1=1, P2=2, P3=3, k_fwd=2, k_bwd=0.5
            ),
            factorial(3) * 2 - factorial(3) * 0.5,
        )

    def test_reversible_mass_action_3_4(self):
        self.assertEqual(
            rf.reversible_mass_action_3_4(
                S1=1, S2=2, S3=3, P1=1, P2=2, P3=3, P4=4, k_fwd=2, k_bwd=0.5
            ),
            factorial(3) * 2 - factorial(4) * 0.5,
        )

    def test_reversible_mass_action_4_1(self):
        self.assertEqual(
            rf.reversible_mass_action_4_1(
                S1=1, S2=2, S3=3, S4=4, P1=1, k_fwd=2, k_bwd=0.5
            ),
            factorial(4) * 2 - factorial(1) * 0.5,
        )

    def test_reversible_mass_action_4_2(self):
        self.assertEqual(
            rf.reversible_mass_action_4_2(
                S1=1, S2=2, S3=3, S4=4, P1=1, P2=2, k_fwd=2, k_bwd=0.5
            ),
            factorial(4) * 2 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_4_3(self):
        self.assertEqual(
            rf.reversible_mass_action_4_3(
                S1=1, S2=2, S3=3, S4=4, P1=1, P2=2, P3=3, k_fwd=2, k_bwd=0.5
            ),
            factorial(4) * 2 - factorial(3) * 0.5,
        )

    def test_reversible_mass_action_4_4(self):
        self.assertEqual(
            rf.reversible_mass_action_4_4(
                S1=1, S2=2, S3=3, S4=4, P1=1, P2=2, P3=3, P4=4, k_fwd=2, k_bwd=0.5
            ),
            factorial(4) * 2 - factorial(4) * 0.5,
        )

    def test_reversible_mass_action_variable_1(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_1(1, 1, 2, 1, 0.5),
            factorial(1) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_2(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_2(1, 2, 1, 2, 1, 0.5),
            factorial(2) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_3(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_3(1, 2, 3, 1, 2, 1, 0.5),
            factorial(3) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_4(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_4(1, 2, 3, 4, 1, 2, 1, 0.5),
            factorial(4) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_5(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_5(1, 2, 3, 4, 5, 1, 2, 1, 0.5),
            factorial(5) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_6(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_6(1, 2, 3, 4, 5, 6, 1, 2, 1, 0.5),
            factorial(6) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_7(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_7(1, 2, 3, 4, 5, 6, 7, 1, 2, 1, 0.5),
            factorial(7) * 1 - factorial(2) * 0.5,
        )

    def test_reversible_mass_action_variable_8(self):
        self.assertEqual(
            rf.reversible_mass_action_variable_8(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 1, 0.5),
            factorial(8) * 1 - factorial(2) * 0.5,
        )

    def test_michaelis_menten(self):
        self.assertEqual(rf.michaelis_menten(S=1, vmax=1, km=1), 0.5)
        self.assertEqual(rf.michaelis_menten(S=1, vmax=2, km=1), 1)
        self.assertEqual(rf.michaelis_menten(S=1, vmax=1, km=0.25), 0.8)

    def test_reversible_michaelis_menten(self):
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=1, vmax_fwd=1, vmax_bwd=1, kms=1, kmp=1
            ),
            0,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=2, P=1, vmax_fwd=1, vmax_bwd=1, kms=1, kmp=1
            ),
            0.25,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=2, vmax_fwd=1, vmax_bwd=1, kms=1, kmp=1
            ),
            -0.25,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=1, vmax_fwd=4, vmax_bwd=1, kms=1, kmp=1
            ),
            1,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=1, vmax_fwd=1, vmax_bwd=4, kms=1, kmp=1
            ),
            -1,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=1, vmax_fwd=1, vmax_bwd=1, kms=2, kmp=1
            ),
            -0.2,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten(
                S=1, P=1, vmax_fwd=1, vmax_bwd=1, kms=1, kmp=2
            ),
            0.2,
        )

    def test_reversible_michaelis_menten_keq(self):
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=1, P=1, vmax_fwd=1, kms=1, kmp=1, keq=1),
            0,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=2, P=1, vmax_fwd=1, kms=1, kmp=1, keq=1),
            0.25,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=1, P=2, vmax_fwd=1, kms=1, kmp=1, keq=1),
            -0.25,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=1, P=1, vmax_fwd=1, kms=1, kmp=1, keq=4),
            0.25,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(
                S=1, P=1, vmax_fwd=1, kms=1, kmp=1, keq=0.25
            ),
            -1,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=2, P=1, vmax_fwd=1, kms=3, kmp=1, keq=1),
            0.125,
        )
        self.assertEqual(
            rf.reversible_michaelis_menten_keq(S=2, P=1, vmax_fwd=1, kms=1, kmp=3, keq=1),
            0.3,
        )

    def test_competitive_inhibition(self):
        self.assertEqual(rf.competitive_inhibition(S=1, I=1, vmax=1, km=0.5, ki=1), 0.5)

    def test_uncompetitive_inhibition(self):
        self.assertEqual(rf.uncompetitive_inhibition(S=1, I=1, vmax=1, km=0.5, ki=1), 0.4)

    def test_noncompetitive_inhibition(self):
        self.assertEqual(
            rf.noncompetitive_inhibition(S=1, I=1, vmax=1, km=0.6, ki=1), 0.3125
        )

    def test_mixed_inhibition(self):
        self.assertEqual(rf.mixed_inhibition(S=1, I=1, vmax=1, km=0.6, ki=1), 0.3125)

    def test_competitive_activation(self):
        self.assertEqual(rf.competitive_activation(S=1, A=1, vmax=1, km=0.5, ka=1), 0.5)

    def test_uncompetitive_activation(self):
        self.assertEqual(rf.uncompetitive_activation(S=1, A=1, vmax=1, km=0.5, ka=1), 0.4)

    def test_noncompetitive_activation(self):
        self.assertEqual(
            rf.noncompetitive_activation(S=1, A=1, vmax=1, km=0.6, ka=1), 0.3125
        )

    def test_mixed_activation(self):
        self.assertEqual(rf.mixed_activation(S=1, A=1, vmax=1, km=0.6, ka=1), 0.3125)

    def test_reversible_uncompetitive_inhibition(self):
        self.assertEqual(
            rf.reversible_uncompetitive_inhibition(
                S=1, P=0.5, I=0.5, vmax_fwd=2, vmax_bwd=1, kms=1, kmp=1, ki=0.7
            ),
            0.525,
        )

    def test_reversible_noncompetitive_inhibition(self):
        self.assertEqual(
            rf.reversible_noncompetitive_inhibition(
                S=1, P=0.5, I=0.5, vmax_fwd=2, vmax_bwd=1, kms=1, kmp=1, ki=1
            ),
            0.4,
        )

    def test_reversible_uncompetitive_inhibition_keq(self):
        self.assertEqual(
            rf.reversible_uncompetitive_inhibition_keq(
                S=1, P=0.5, I=0.5, vmax_fwd=1, kms=1, kmp=1, ki=0.7, keq=1
            ),
            0.175,
        )

    def test_reversible_noncompetitive_inhibition_keq(self):
        self.assertEqual(
            rf.reversible_noncompetitive_inhibition_keq(
                S=1, P=0.5, I=0.5, vmax_fwd=2, kms=1, kmp=1, ki=0.5, keq=1
            ),
            0.2,
        )

    def test_hill(self):
        self.assertEqual(rf.hill(S=1, vmax=1, kd=1, n=1), 0.5)


class MultiSubstrateTests(unittest.TestCase):
    def test_ordered_2(self):
        self.assertEqual(rf.ordered_2(A=1, B=1, vmax=1, kmA=1, kmB=1, kiA=1), 0.25)

    def test_ordered_2_2(self):
        self.assertEqual(
            rf.ordered_2_2(
                A=1,
                B=1,
                P=0,
                Q=0,
                vmaxf=1,
                vmaxr=1,
                kmA=1,
                kmB=1,
                kmP=1,
                kmQ=1,
                kiA=1,
                kiB=1,
                kiP=1,
                kiQ=1,
            ),
            0.25,
        )

        self.assertEqual(
            rf.ordered_2_2(
                A=1,
                B=1,
                P=1,
                Q=1,
                vmaxf=1,
                vmaxr=1,
                kmA=1,
                kmB=1,
                kmP=1,
                kmQ=1,
                kiA=1,
                kiB=1,
                kiP=1,
                kiQ=1,
            ),
            0.0,
        )

    def test_random_order_2(self):
        self.assertEqual(rf.random_order_2(A=1, B=1, vmax=1, kmA=1, kmB=1, kiA=1), 0.25)

    def test_random_order_2_2(self):
        self.assertEqual(
            rf.random_order_2_2(
                A=1,
                B=1,
                P=0,
                Q=0,
                vmaxf=1,
                vmaxr=1,
                kmB=1,
                kmP=1,
                kiA=1,
                kiB=1,
                kiP=1,
                kiQ=1,
            ),
            0.25,
        )

        self.assertEqual(
            rf.random_order_2_2(
                A=1,
                B=1,
                P=1,
                Q=1,
                vmaxf=1,
                vmaxr=1,
                kmB=1,
                kmP=1,
                kiA=1,
                kiB=1,
                kiP=1,
                kiQ=1,
            ),
            0.0,
        )

    def test_ping_pong_2(self):
        self.assertEqual(rf.ping_pong_2(A=2, B=1, vmax=1, kmA=1, kmB=1), 0.4)

    def test_ping_pong_3(self):
        self.assertEqual(rf.ping_pong_3(A=1, B=1, C=1, vmax=1, kmA=1, kmB=1, kmC=1), 0.25)

    def test_ping_pong_4(self):
        self.assertEqual(
            rf.ping_pong_4(A=1, B=1, C=1, D=1, vmax=1, kmA=1, kmB=1, kmC=1, kmD=1), 0.2
        )
