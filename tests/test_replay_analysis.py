"""Coaching risk must use pre-action evidence and treat flags as hypotheses."""
from types import SimpleNamespace
import unittest
from minesweeper_ai.game import FLAGGED, UNKNOWN, MinesweeperGame
from minesweeper_ai.local_game import LocalGameSession
from minesweeper_ai.local_replay import ReplayTimeline
from minesweeper_ai.replay_analysis import assess_move, observed_effect
from minesweeper_ai.solver import Analysis


def move(kind="reveal", row=0, col=0):
    return dict(action=kind, row=row, col=col)


class ReplayAnalysisTests(unittest.TestCase):
    def test_exact_guess_is_not_judged_by_its_terminal_outcome(self):
        assessments=[]
        for mine in ((0,0),(0,1)):
            session=LocalGameSession(MinesweeperGame(2,2,1,mine_positions={mine}))
            session.apply("reveal",(1,1)); session.apply("reveal",(0,0))
            timeline=ReplayTimeline.from_session(session,seed=0)
            previous=timeline.board_at(1)
            assessments.append(assess_move(previous.observation,1,move()))
        self.assertEqual(assessments[0],assessments[1])
        self.assertAlmostEqual(assessments[0].mine_probability,1/3)
        self.assertTrue(assessments[0].exact)

    def test_wrong_flag_does_not_manufacture_safety(self):
        result=assess_move(((UNKNOWN,UNKNOWN),(FLAGGED,1)),1,move())
        self.assertAlmostEqual(result.mine_probability,1/3)
        flag=assess_move(((UNKNOWN,UNKNOWN),(FLAGGED,1)),1,move("flag",1,0))
        self.assertAlmostEqual(flag.mine_probability,1/3)
        self.assertIn("取消旗子",flag.text)

    def test_proven_safer_alternative(self):
        result=assess_move(((UNKNOWN,1,UNKNOWN),(UNKNOWN,UNKNOWN,UNKNOWN),(UNKNOWN,UNKNOWN,UNKNOWN)),1,move())
        self.assertAlmostEqual(result.mine_probability,.2)
        self.assertEqual(result.best_probability,0)
        self.assertEqual(result.safer_count,3)
        self.assertIn("已證明安全",result.text)

    def test_chord_does_not_mislabel_marginals_as_joint_risk(self):
        result=assess_move(((UNKNOWN,UNKNOWN),(FLAGGED,1)),1,move("chord",1,1))
        self.assertIsNone(result.mine_probability)
        self.assertIn("最高單格雷率 33.33%",result.text)
        self.assertIn("不是整次連開",result.text)
        inactive=assess_move(((UNKNOWN,UNKNOWN),(UNKNOWN,1)),1,move("chord",1,1))
        self.assertIn("未觸發",inactive.text)

    def test_estimated_risk_keeps_proofs_and_labels_approximation(self):
        analysis=Analysis(frozenset({(1,0)}),frozenset(),{(0,0):.5,(0,1):.5,(1,0):0},False,None,3)
        solver=SimpleNamespace(analyse=lambda *args:analysis)
        result=assess_move(((UNKNOWN,UNKNOWN),(UNKNOWN,1)),1,move(),solver=solver,
                           predict_risks=lambda *args:{(0,0):.2,(0,1):.8,(1,0):.9})
        self.assertAlmostEqual(result.mine_probability,.305)
        self.assertFalse(result.exact)
        self.assertEqual(result.best_probability,0)
        self.assertIn("V44 混合估計",result.text)

    def test_first_click_noop_and_observed_effect_are_separate(self):
        result=assess_move(((UNKNOWN,UNKNOWN),(UNKNOWN,UNKNOWN)),1,move())
        self.assertIsNone(result.mine_probability)
        noop=assess_move(((UNKNOWN,UNKNOWN),(UNKNOWN,1)),1,move("reveal",1,1))
        self.assertIn("沒有開啟新格",noop.text)
        step=SimpleNamespace(changes=((1,1,UNKNOWN,1),),action=dict(action="reveal",effective=True))
        self.assertIn("實際展開 1 個安全格",observed_effect(step))
        self.assertIn("事後效果",observed_effect(step))
