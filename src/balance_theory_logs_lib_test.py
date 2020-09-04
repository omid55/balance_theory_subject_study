# Omid55
# Test module for group dynamics logs library.


from __future__ import division, print_function, absolute_import, unicode_literals

import os
import unittest
import numpy as np
import pandas as pd
from pandas import testing as pd_testing
from numpy import testing as np_testing
from parameterized import parameterized

import balance_theory_logs_lib as bt
import utils


class TestRmoveEarlyDuplicates(unittest.TestCase):
    # =========================================================================
    # ================ remove_early_duplicates ================================
    # =========================================================================
    def test_remove_early_duplicates(self):
        df = pd.DataFrame([
            ['person1', 'q1', '1', '2020-09-01 20:00:00'],
            ['person1', 'q1', '4', '2020-09-01 20:00:10'],
            ['person2', 'q1', '2', '2020-09-01 20:00:15'],
            ['person1', 'q1', '10', '2020-09-01 20:00:30'],
            ['person2', 'q2', '9', '2020-09-01 20:00:45'],
            ['person1', 'q2', '5', '2020-09-01 20:00:50'],
            ['person2', 'q1', '55', '2020-09-01 20:01:01'],
            ['person1', 'q1', '360', '2020-09-01 20:01:12']],
            columns=['sender', 'question', 'value', 'timestamp'])
        df = df.sample(frac=1)  # To shuffle all rows and to be out of order.
        expected_df = df = pd.DataFrame([
            ['person2', 'q2', '9', '2020-09-01 20:00:45'],
            ['person1', 'q2', '5', '2020-09-01 20:00:50'],
            ['person2', 'q1', '55', '2020-09-01 20:01:01'],
            ['person1', 'q1', '360', '2020-09-01 20:01:12']],
            columns=['sender', 'question', 'value', 'timestamp'])
        computed_df = bt.remove_early_duplicates(
            df=df, groupby_columns=['sender', 'question'])
        pd_testing.assert_frame_equal(expected_df, computed_df)

class TestTeamLogsLoader(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.loader = bt.TeamLogsLoader(
            directory=os.getcwd() + '/src/testing_log/synthetic1_raw_logs')

    @classmethod
    def tearDown(cls):
        del cls.loader

    # =========================================================================
    # ================================ _load ==================================
    # =========================================================================
    def test_load_answers_are_correct(self):
        dt = [['pogs01', 'GD_solo_disaster0', '$1000', '2020-09-01 20:32:39'],
              ['pogs02', 'GD_solo_disaster0', '$2000', '2020-09-01 20:32:42'],
              ['pogs03', 'GD_solo_disaster0', '$3000', '2020-09-01 20:33:37'],
              ['pogs01', 'GD_solo_disaster1', '1500', '2020-09-01 20:37:01'],
              ['pogs02', 'GD_solo_disaster1', '2000', '2020-09-01 20:37:03'],
              ['pogs03', 'GD_solo_disaster1', '$2500', '2020-09-01 20:37:08'],
              ['pogs01', 'GD_solo_disaster2', '$2000', '2020-09-01 20:43:12'],
              ['pogs02', 'GD_solo_disaster2', '$2200', '2020-09-01 20:43:21'],
              ['pogs03', 'GD_solo_disaster2', '$2100', '2020-09-01 20:43:26'],
              ['pogs01', 'GD_solo_disaster3', '2000', '2020-09-01 20:48:28'],
              ['pogs02', 'GD_solo_disaster3', '2000', '2020-09-01 20:48:31'],
              ['pogs03', 'GD_solo_disaster3', '$2000', '2020-09-01 20:48:34']]
        expected_answers = pd.DataFrame(
            dt, columns=['sender', 'question', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_answers, self.loader.answers)

    def test_load_messages_are_correct(self):
        dt = [['pogs01', 'GD_group_disaster1', 'Hey! my name is Sara.', '2020-09-01 20:34:32'],
              ['pogs02', 'GD_group_disaster1', 'Hi there', '2020-09-01 20:34:38'],
              ['pogs03', 'GD_group_disaster1', 'sup???', '2020-09-01 20:34:43'],
              ['pogs01', 'GD_group_disaster1', 'cool', '2020-09-01 20:35:07'],
              ['pogs01', 'GD_group_disaster2', 'heyyyy', '2020-09-01 20:42:08'],
              ['pogs03', 'GD_group_disaster2', 'good!', '2020-09-01 20:42:33'],
              ['pogs02', 'GD_group_disaster3', "let's go ...", '2020-09-01 20:46:25']]
        expected_messages = pd.DataFrame(dt,
            columns=['sender', 'question', 'text', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_messages, self.loader.messages)

    def test_load_members_are_correct(self):
        np_testing.assert_array_equal(
            self.loader.members, np.array(['pogs01', 'pogs02', 'pogs03']))

    def test_load_questions_order_are_correct(self):
        expected_questions_order = [
            'GD_solo_disaster_initial', 'GD_group_disaster1',
            'GD_solo_disaster1', 'GD_influence_disaster1',
            'GD_appraisal_disaster1', 'GD_group_disaster2',
            'GD_solo_disaster2', 'GD_influence_disaster2',
            'GD_appraisal_disaster2', 'GD_group_disaster3',
            'GD_solo_disaster3', 'GD_influence_disaster3',
            'GD_appraisal_disaster3']
        np_testing.assert_array_equal(
            self.loader.questions_order, expected_questions_order)

    def test_load_influences_are_correct(self):
        dt = [['pogs01', 'GD_influence_disaster1', [30, 30, 30], '2020-09-01 20:37:40'],
              ['pogs02', 'GD_influence_disaster1', [0, 100, 0], '2020-09-01 20:37:48'],
              ['pogs03', 'GD_influence_disaster1', [20, 20, 60], '2020-09-01 20:38:09'],
              ['pogs01', 'GD_influence_disaster2', [100, 0, 0], '2020-09-01 20:43:50'],
              ['pogs02', 'GD_influence_disaster2', [30, 30, 30], '2020-09-01 20:43:57'],
              ['pogs03', 'GD_influence_disaster2', [30, 30, 40], '2020-09-01 20:44:02'],
              ['pogs01', 'GD_influence_disaster3', [100, 0, 0], '2020-09-01 20:49:22'],
              ['pogs02', 'GD_influence_disaster3', [30, 40, 30], '2020-09-01 20:49:35'],
              ['pogs03', 'GD_influence_disaster3', [50, 50, 0], '2020-09-01 20:49:43']]
        expected_influences = pd.DataFrame(dt,
            columns=['sender', 'question', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_influences, self.loader.influences)

    def test_load_appraisals_are_correct(self):
        dt = [['pogs01', 'GD_appraisal_disaster1', [0, -1, 2], '2020-09-01 20:39:35'],
              ['pogs02', 'GD_appraisal_disaster1', [-10, 10, -10], '2020-09-01 20:40:23'],
              ['pogs03', 'GD_appraisal_disaster1', [-10, 0, 0], '2020-09-01 20:40:37'],
              ['pogs01', 'GD_appraisal_disaster2', [10, 10, 10], '2020-09-01 20:44:42'],
              ['pogs02', 'GD_appraisal_disaster2', [-3, -1, -2], '2020-09-01 20:44:52'],
              ['pogs03', 'GD_appraisal_disaster2', [10, -10, -1], '2020-09-01 20:45:20'],
              ['pogs01', 'GD_appraisal_disaster3', [10, 10, 10], '2020-09-01 20:50:35'],
              ['pogs02', 'GD_appraisal_disaster3', [1, 10, 1], '2020-09-01 20:50:56'],
              ['pogs03', 'GD_appraisal_disaster3', [-10, -10, 10], '2020-09-01 20:51:04']]
        expected_appraisals = pd.DataFrame(dt,
            columns=['sender', 'question', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_appraisals, self.loader.appraisals)

    # =========================================================================
    # =============== get_all_groups_info_in_one_dataframe ====================
    # =========================================================================
    def test_get_all_groups_info_in_one_dataframe(self):
        self.loader.team_id = 's10'
        teams_log_list = [self.loader]
        dt = [['s10', 'asbestos', 'pogs01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'asbestos', 'pogs02', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'asbestos', 'pogs03', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'disaster', 'pogs01', '$1000', '1500', '$2000', '2000', 30, 30, 30, 100, 0, 0, 100, 0, 0, 0, -1, 2, 10, 10, 10, 10, 10, 10],
              ['s10', 'disaster', 'pogs02', '$2000', '2000', '$2200', '2000', 0, 100, 0, 30, 30, 30, 30, 40, 30, -10, 10, -10, -3, -1, -2, 1, 10, 1],
              ['s10', 'disaster', 'pogs03', '$3000', '$2500', '$2100', '$2000', 20, 20, 60, 30, 30, 40, 50, 50, 0, -10, 0, 0, 10, -10, -1, -10, -10, 10],
              ['s10', 'sports', 'pogs01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'sports', 'pogs02', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'sports', 'pogs03', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'school', 'pogs01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'school', 'pogs02', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'school', 'pogs03', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'surgery', 'pogs01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'surgery', 'pogs02', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
              ['s10', 'surgery', 'pogs03', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']]
        expected = pd.DataFrame(dt, columns = [
            'Group', 'Issue', 'Person', 'Initial opinion',
            'Period1 opinion', 'Period2 opinion', 'Period3 opinion',
            'Period1 w1', 'Period1 w2', 'Period1 w3',
            'Period2 w1', 'Period2 w2', 'Period2 w3',
            'Period3 w1', 'Period3 w2', 'Period3 w3',
            'Period1 a1', 'Period1 a2', 'Period1 a3',
            'Period2 a1', 'Period2 a2', 'Period2 a3',
            'Period3 a1', 'Period3 a2', 'Period3 a3'])
        computed = bt.get_all_groups_info_in_one_dataframe(teams_log_list)
        pd_testing.assert_frame_equal(expected, computed, check_dtype=False)
