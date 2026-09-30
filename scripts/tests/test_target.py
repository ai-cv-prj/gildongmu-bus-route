import unittest
from bus_route.target import TargetMatcher,exact_route


def obs(text,score=.99,eligible=True):return {'text':text,'token_scores':[score]*(len(text)+1),'eligible':eligible}


class TargetTests(unittest.TestCase):
    def test_single_whole_number_at_arbitrary_time_and_legacy_is_unchanged(self):
        for t in [0,8.8,30.2]:
            self.assertEqual(TargetMatcher('7016',single_score=.98).update(1,t,[obs('7016')])['state'],'recognized_single')
            self.assertEqual(TargetMatcher('7016').update(1,t,[obs('7016')])['state'],'pending')

    def test_single_cannot_bypass_ambiguity_conflict_or_incomplete_scan(self):
        for observations,complete in [([obs('7016',.979)],True),([obs('7016',eligible=False)],True),
                                      ([obs('7016')],False),([obs('7016'),obs('7017',.8)],True),
                                      ([obs('701')],True)]:
            m=TargetMatcher('7016',['7017'],single_score=.98)
            self.assertNotIn(m.update(1,0,observations,complete=complete)['state'],{'recognized_single','matched_candidate'})
        m=TargetMatcher('7016',single_score=.98);m.update(1,0,[obs('7017')])
        self.assertEqual(m.update(1,.2,[obs('7016')])['state'],'conflict')

    def test_single_uses_eos_and_upgrades_without_retaining_absent_number(self):
        m=TargetMatcher('7016',single_score=.98)
        p=obs('7016');p['token_scores'][-1]=.95
        self.assertEqual(m.update(1,0,[p])['state'],'pending')
        self.assertEqual(m.update(1,.2,[obs('7016')])['state'],'recognized_single')
        self.assertEqual(m.update(1,.4,[obs('7016')])['state'],'matched_candidate')
        self.assertEqual(m.update(1,.6,[])['state'],'unverified')
        for score in [.89,float('nan'),1.1]:
            with self.assertRaises(ValueError):TargetMatcher('7016',single_score=score)

    def test_aliases_preserve_zero_letters(self):
        self.assertTrue(exact_route('07','마포07'))
        for text,target in [('7','마포07'),('110A','110B'),('6628','M6628'),('강서07','마포07')]:
            self.assertFalse(exact_route(text,target))

    def test_partial_never_completes(self):
        m=TargetMatcher('7016',['7017'])
        for i in range(20):
            r=m.update(1,i*.2,[obs('701'),obs('701?'),obs('?016')])
            self.assertEqual(r['state'],'partial')
            self.assertEqual(r['support_samples'],0)

    def test_exact_repeat_only_distinct_frames(self):
        m=TargetMatcher('7016')
        for _ in range(5):self.assertEqual(m.update(1,0,[obs('7016')]*4)['state'],'pending')
        self.assertEqual(m.update(1,.2,[obs('7016')])['state'],'pending')
        self.assertEqual(m.update(1,.4,[obs('7016')])['state'],'matched_candidate')
        self.assertEqual(m.update(1,.6,[])['state'],'unverified')

    def test_recent_7017_vetoes_7016_then_requires_fresh_full_evidence(self):
        m=TargetMatcher('7016',['7017']);m.update(1,0,[obs('7017')])
        for i in range(1,7):self.assertEqual(m.update(1,i*.2,[obs('7016')])['state'],'conflict')
        self.assertEqual(m.update(1,1.4,[obs('7016')])['state'],'matched_candidate')

    def test_continuing_similar_competitors_never_pass(self):
        m=TargetMatcher('7016',['7017'])
        for i in range(30):
            r=m.update(1,i*.2,[obs('7016' if i%2 else '7017')])
            self.assertNotEqual(r['state'],'matched_candidate')

    def test_single_weak_misread_requires_confident_distinguishing_digits(self):
        for distinguishing,expected in [(.99,'matched_candidate'),(.94,'conflict')]:
            m=TargetMatcher('7016',['7017'])
            for t in [0,.2]:m.update(1,t,[obs('7016',distinguishing)])
            m.update(1,.4,[obs('7017',.78)])
            self.assertEqual(m.update(1,.6,[obs('7016',distinguishing)])['state'],expected)

    def test_repeated_weak_or_simultaneous_competitor_blocks(self):
        for repeated in [False,True]:
            m=TargetMatcher('7016',['7017'])
            for t in [0,.2]:m.update(1,t,[obs('7016'),*([obs('7017',.8)] if repeated else [])])
            r=m.update(1,.4,[obs('7016'),obs('7017',.8)])
            self.assertEqual(r['state'],'conflict')

    def test_unrelated_text_invalid_suffix_and_plate_do_not_veto(self):
        m=TargetMatcher('271')
        for t in [0,.2,.4]:
            r=m.update(1,t,[obs('271'),obs('100'),obs('27L'),obs('277',eligible=False)])
        self.assertEqual(r['state'],'matched_candidate')
        m=TargetMatcher('271')
        for t in [0,.2,.4]:r=m.update(1,t,[obs('27L')])
        self.assertNotEqual(r['state'],'matched_candidate')

    def test_explicit_dissimilar_competitor_still_blocks(self):
        m=TargetMatcher('7016',['7770'])
        for t in [0,.2,.4]:r=m.update(1,t,[obs('7016'),obs('7770')])
        self.assertEqual(r['state'],'conflict')

    def test_two_tracks_do_not_share_evidence(self):
        m=TargetMatcher('7016')
        for i in range(3):self.assertEqual(m.update(i,i*.2,[obs('7016')])['state'],'pending')
        m.update(9,0,[obs('7017')])
        for i in range(3):r=m.update(1,1+i*.2,[obs('7016')])
        self.assertEqual(r['state'],'matched_candidate')

    def test_ambiguous_low_score_or_truncated_never_pass(self):
        for observations,complete in [([obs('7016',.89)],True),([obs('7016',eligible=False)],True),([obs('7016')],False)]:
            m=TargetMatcher('7016')
            for i in range(10):self.assertNotEqual(m.update(1,i*.2,observations,complete=complete)['state'],'matched_candidate')

    def test_gap_and_jump_clear_evidence(self):
        m=TargetMatcher('7016');m.update(1,0,[obs('7016')]);m.update(1,.2,[obs('7016')])
        self.assertEqual(m.update(1,2,[obs('7016')])['state'],'pending')
        m=TargetMatcher('7016')
        for t in [0,.2]:m.update(1,t,[obs('7016')],(0,0,20,20))
        self.assertEqual(m.update(1,.4,[obs('7016')],(100,100,120,120))['state'],'hold')

    def test_collision_untracked_and_bad_score(self):
        with self.assertRaises(ValueError):TargetMatcher('마포07',['강서07'])
        m=TargetMatcher('7016')
        self.assertEqual(m.update(None,0,[obs('7016')])['state'],'untracked')
        self.assertEqual(m.update(1,0,[obs('7016',float('nan'))])['state'],'hold')


if __name__=='__main__':unittest.main()
