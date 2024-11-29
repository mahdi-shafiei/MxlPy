"""Label Scope Module.

This module provides functions for creating and managing label scopes in metabolic models.
It includes functionality for initializing label scopes and retrieving reachable label positions.

Functions:
    get_label_scope: Return all label positions that can be reached step by step.
"""

from __future__ import annotations

# def _create_label_scope_seed(
#     self, *, initial_labels: dict[str, int] | dict[str, list[int]]
# ) -> dict[str, int]:
#     """Create initial label scope seed."""
#     # initialise all compounds with 0 (no label)
#     labelled_compounds = {compound: 0 for compound in self.get_compounds()}
#     # Set all unlabelled compounds to 1
#     for name, compound in self.label_compounds.items():
#         num_labels = compound["num_labels"]
#         labelled_compounds[f"{name}__{'0' * num_labels}"] = 1
#     # Also set all non-label compounds to 1
#     for name in self.nonlabel_compounds:
#         labelled_compounds[name] = 1
#     # Set initial label
#     for i in [
#         self.get_compound_isotopomer_with_label_position(
#             base_compound=base_compound, label_position=label_position
#         )
#         for base_compound, label_position in initial_labels.items()
#     ]:
#         labelled_compounds[i] = 1
#     return labelled_compounds


# def get_label_scope(
#     self,
#     initial_labels: dict[str, int] | dict[str, list[int]],
# ) -> dict[int, set[str]]:
#     """Return all label positions that can be reached step by step.

#     Parameters:
#     initial_labels : dict(str: num)

#     Returns:
#     label_scope : dict{step : set of new positions}

#     Examples:
#     >>> l.get_label_scope({"x": 0})
#     >>> l.get_label_scope({"x": [0, 1], "y": 0})

#     """
#     labelled_compounds = self._create_label_scope_seed(initial_labels=initial_labels)
#     new_labels = set("non empty entry to not fulfill while condition")
#     # Loop until no new labels are inserted
#     loop_count = 0
#     result = {}
#     while new_labels != set():
#         new_cpds = labelled_compounds.copy()
#         for rec, cpd_dict in self.get_stoichiometries().items():
#             # Isolate substrates
#             cpds = [i for i, j in cpd_dict.items() if j < 0]
#             # Count how many of the substrates are 1
#             i = 0
#             for j in cpds:
#                 i += labelled_compounds[j]
#             # If all substrates are 1, set all products to 1
#             if i == len(cpds):
#                 for cpd in self.get_stoichiometries()[rec]:
#                     new_cpds[cpd] = 1
#             if self.rates[rec]["reversible"]:
#                 # Isolate substrates
#                 cpds = [i for i, j in cpd_dict.items() if j > 0]
#                 # Count how many of the substrates are 1
#                 i = 0
#                 for j in cpds:
#                     i += labelled_compounds[j]
#                 # If all substrates are 1, set all products to 1
#                 if i == len(cpds):
#                     for cpd in self.get_stoichiometries()[rec]:
#                         new_cpds[cpd] = 1
#         # Isolate "old" labels
#         s1 = pd.Series(labelled_compounds)
#         s1 = cast(pd.Series, s1[s1 == 1])
#         # Isolate new labels
#         s2 = pd.Series(new_cpds)
#         s2 = cast(pd.Series, s2[s2 == 1])
#         # Find new labels
#         new_labels = cast(set[str], set(s2.index).difference(set(s1.index)))
#         # Break the loop once no new labels can be found
#         if new_labels == set():
#             break
#         labelled_compounds = new_cpds
#         result[loop_count] = new_labels
#         loop_count += 1
#     return result
