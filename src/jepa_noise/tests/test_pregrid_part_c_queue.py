import importlib.util,json,unittest
from pathlib import Path
from copy import deepcopy
spec=importlib.util.spec_from_file_location('queue',str(Path(__file__).resolve().parents[3]/'scripts/jepa_pregrid_part_c_queue.py'));q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def good():
 return {'registered':False,'complete':True,'items':{label:({'full_input':{'vertex_resolution_m':expected},'reproduction':{'expected_m':expected,'got_m':expected,'pass':True}} if label in q.EXPECTED else {'beta':100.0}) for label in q.LABELS for expected in [q.EXPECTED.get(label)]}}
class Gates(unittest.TestCase):
 def test_zero_process_exit_does_not_override_failed_expect(self):
  r=good();r['items']['fix9000']['reproduction']['pass']=False;self.assertEqual(q.verdict(r),'STOP_S2')
 def test_recompute_tolerance_instead_of_trusting_boolean(self):
  r=good();r['items']['fix9000']['full_input']['vertex_resolution_m']+=.002;self.assertEqual(q.verdict(r),'STOP_S2')
 def test_no_missing_checkpoints(self):
  r=good();del r['items']['anc100'];self.assertEqual(q.verdict(r),'ENGINEERING_FAILURE')
 def test_registered_results_are_not_accepted(self):
  r=good();r['registered']=True;self.assertEqual(q.verdict(r),'ENGINEERING_FAILURE')
 def test_exact_checks_pass(self):self.assertEqual(q.verdict(good()),'PART_C_COMPLETE')
 def test_budget_includes_timeout_and_prior_failure(self):
  self.assertTrue(q.within_budget(q.BASE_HOURS,100));self.assertFalse(q.within_budget(29,100))
 def test_incomplete_report_is_not_accepted(self):
  r=good();r['complete']=False;self.assertEqual(q.verdict(r),'ENGINEERING_FAILURE')
 def test_anchored_item_must_carry_beta_100(self):
  r=good();del r['items']['anc100']['beta'];self.assertEqual(q.verdict(r),'ENGINEERING_FAILURE')
  r=good();r['items']['anc100']['beta']=1000.0;self.assertEqual(q.verdict(r),'ENGINEERING_FAILURE')
 def test_rerun_attempt_carries_the_failed_charge(self):
  self.assertAlmostEqual(q.ATTEMPT_BASE_HOURS['20261029']-q.ATTEMPT_BASE_HOURS['20261026'],3894.59/3600)
  self.assertTrue(q.within_budget(q.ATTEMPT_BASE_HOURS['20261029'],q.PLANNED_MINUTES))
if __name__=='__main__': unittest.main()
