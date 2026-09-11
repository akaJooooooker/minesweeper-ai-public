import copy
import random
import unittest
from minesweeper_ai.game import UNKNOWN, GameStatus, MinesweeperGame
from minesweeper_ai.local_game import LocalGameSession
from minesweeper_ai.local_feedback import build_feedback
from minesweeper_ai.local_replay import ReplayTimeline

class ReplayTests(unittest.TestCase):
    def session(self):
        return LocalGameSession(MinesweeperGame(2, 2, 1, mine_positions={(0,0)}))

    def test_flags_redundant_actions_chord_and_random_seeks(self):
        session=self.session()
        expected=[session.game.observation]
        for action,coord in [('flag',(0,0)),('reveal',(1,1)),('reveal',(0,0)),('chord',(1,1))]:
            session.apply(action,coord)
            expected.append(session.game.observation)
        timeline=ReplayTimeline(build_feedback(session,seed=7))
        self.assertEqual(len(timeline),4)
        for index in [4,0,2,1,3,4,2,0]:
            self.assertEqual(timeline.board_at(index).observation,expected[index])
            self.assertEqual(timeline.steps[index].summary['total_clicks'],index)
        self.assertEqual(timeline.steps[4].status,GameStatus.WON)
        self.assertFalse(timeline.steps[3].action['effective'])
        self.assertIsNone(timeline.steps[0].summary['three_bv'])
        self.assertIsNone(timeline.steps[1].summary['three_bv'])

    def test_losing_guess_rewinds_explosion_and_retains_reason(self):
        session=self.session()
        session.apply('reveal',(1,1))
        session.apply('reveal',(0,0),source='V44',decision=dict(mine_probability=1/3,exact=True,reason='minimum exact mine probability'))
        timeline=ReplayTimeline(build_feedback(session,seed=7))
        self.assertEqual(timeline.board_at(2).exploded,(0,0))
        self.assertEqual(timeline.board_at(1).status,GameStatus.ACTIVE)
        self.assertIsNone(timeline.board_at(1).exploded)
        self.assertEqual(timeline.board_at(1).observation[0][0],UNKNOWN)
        self.assertIn('33.33%',timeline.description(2))
        self.assertIn('最低精確雷率',timeline.reason(2))

    def test_seed_only_legacy_and_long_checkpoint_seeks(self):
        session=LocalGameSession(MinesweeperGame(30,20,130,seed=81,first_click_zero=True))
        session.apply('reveal',(10,15))
        expected=[tuple(tuple(UNKNOWN for c in range(30)) for r in range(20)),session.game.observation]
        for i in range(90):
            session.apply('flag',(0,i%30))
            expected.append(session.game.observation)
        payload=dict(seed=81,first_click_zero=True,**session.summary(),actions=session.history)
        timeline=ReplayTimeline(payload)
        order=list(range(92)); random.Random(3).shuffle(order)
        for index in order:
            self.assertEqual(timeline.board_at(index).observation,expected[index])
        self.assertEqual(len(timeline.checkpoints),3)
        with self.assertRaises(IndexError): timeline.board_at(-1)

    def test_invalid_import_is_rejected(self):
        with self.assertRaises(ValueError): ReplayTimeline([])
        session=self.session(); session.apply('reveal',(1,1))
        payload=ReplayTimeline.from_session(session,seed=7).payload
        for change in ({'actions':['invalid']},{'actions':[dict(action='reveal',row=1,col=1,decision='bad')]},{'replay':[]},{'rows':0},{'status':'won'},{'actions':[dict(action='reveal',row=4,col=0)]}, {'actions':[dict(action='reveal',row=0,col=0,seconds=float('nan'))]}):
            with self.assertRaises(ValueError): ReplayTimeline({**payload,**change})
        session.apply('reveal',(0,0))
        payload=build_feedback(session,seed=7)
        bad=copy.deepcopy(payload); bad['replay']['moves'][0]['row']=0
        with self.assertRaises(ValueError): ReplayTimeline(bad)
        bad=copy.deepcopy(payload); bad['actions'].append(bad['actions'][-1])
        with self.assertRaises(ValueError): ReplayTimeline(bad)

    def test_pause_excludes_replay_time_and_can_repeat(self):
        now=[10.0]
        session=LocalGameSession(MinesweeperGame(2,2,1,mine_positions={(0,0)}),clock=lambda:now[0])
        session.pause_clock(); now[0]=20; session.resume_clock()
        session.apply('reveal',(1,1)); now[0]=22; session.pause_clock()
        now[0]=32; session.pause_clock(); self.assertEqual(session.elapsed,2)
        session.resume_clock(); now[0]=33; self.assertEqual(session.elapsed,3)
        session.pause_clock(); now[0]=43; session.resume_clock()
        session.apply('reveal',(0,0)); now[0]=53; self.assertEqual(session.elapsed,3)
