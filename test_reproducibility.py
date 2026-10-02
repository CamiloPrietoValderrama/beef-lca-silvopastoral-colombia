from LCA_Beef_SSPi_MonteCarlo import simulate, verify_manuscript_outputs, primary_summary, structural_table, shapley_table
import numpy as np

r = simulate(10_000, 42)
verify_manuscript_outputs(r)
s = primary_summary(r)
assert abs(s.loc[s.Scenario=="BAU-F","Mean"].iloc[0]-13.87936240039728) < 1e-8
assert np.allclose(sum(r["BAU-F"][k] for k in ["enteric", "manure", "n2o", "inputs"]), r["BAU-F"]["Eprod"])
assert len(structural_table(r)) == 10
assert len(shapley_table(r)) == 5
assert abs(4*0.42*365-609)/609 < 0.01
assert abs(3.5*0.856*365-1098)/1098 < 0.01
print("PASS: verified revised 450-kg UGG model outputs, decomposition, and source productivity cross-checks")
