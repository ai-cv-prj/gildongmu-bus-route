import unittest
from selection import Selector

def bus(key,box):return {'track_id':key,'bus_box':box}
def approach(s,key=7):
 for t,box in [(0,[10,100,110,200]),(.3,[15,100,140,220]),(.6,[20,100,180,250])]:s.update(t,[bus(key,box)])

class SelectionTests(unittest.TestCase):
 def test_opposite_side_never_selected_even_if_larger(self):
  s=Selector(1000,1000)
  for t in [0,.3,.6,.9]:s.update(t,[bus(1,[600,100,1000,500])])
  self.assertIsNone(s.focus)
 def test_entry_without_approach_is_not_selected(self):
  s=Selector(1000,1000)
  for t in [0,.3,.6,.9]:s.update(t,[bus(1,[0,100,180,300])])
  self.assertIsNone(s.focus)
 def test_approach_is_selected_after_history(self):
  s=Selector(1000,1000);approach(s);self.assertEqual(s.focus,7);self.assertEqual(s.state(.6),'checking')
 def test_read_survives_missing_ocr(self):
  s=Selector(1000,1000);approach(s);s.observe(.6,['7011'])
  for t in [.9,1.2,1.5,1.8,2.1]:s.update(t,[bus(7,[20,100,180,250])]);s.observe(t,[])
  self.assertEqual(s.state(2.1),'recognized');self.assertEqual(s.number,'7011')
 def test_never_read_can_fail_and_recover(self):
  s=Selector(1000,1000);approach(s);self.assertEqual(s.state(2),'failed');s.observe(2,['271']);self.assertEqual(s.state(2),'recognized')
 def test_exit_removes_known_bus_without_failure_or_reentry(self):
  s=Selector(1000,1000);approach(s);s.observe(.6,['7011']);s.update(.9,[bus(7,[0,100,500,500])]);s.update(1.2,[bus(7,[200,100,1000,800])]);self.assertIsNone(s.focus);self.assertIsNone(s.number);self.assertEqual(s.state(1.2),'search');self.assertIn(7,s.retired)
 def test_identity_jump_never_inherits_number(self):
  s=Selector(1000,1000);approach(s);s.observe(.6,['7011']);s.update(.9,[bus(7,[800,700,980,900])]);self.assertIsNone(s.number);self.assertIsNone(s.focus)
 def test_conflicting_read_clears_known_track(self):
  s=Selector(1000,1000);approach(s);s.observe(.6,['7011']);s.observe(.9,['7017']);self.assertIsNone(s.number);self.assertEqual(s.state(.9),'search')
 def test_disappeared_identity_not_reused(self):
  s=Selector(1000,1000);approach(s);s.observe(.6,['7011']);s.update(.9,[]);s.update(1.2,[bus(8,[20,100,180,250])]);self.assertIsNone(s.number)

if __name__=='__main__':unittest.main()
