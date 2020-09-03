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

    # # =========================================================================
    # # ======================= get_influence_matrices ==========================
    # # =========================================================================
    # def test_get_influence_matrices2x2(self):
    #     expected_question_orders = ['surgery1', 'surgery2']
    #     expected_influence_matrices = [
    #         np.array([[0.9, 0.1],
    #                   [0.49, 0.51]]),
    #         np.array([[0.01, 0.99],
    #                   [0.0, 1.0]])]
    #     expected_influences_from_data = [
    #         np.array([[True, False], [True, True]]),
    #         np.array([[True, True], [True, True]])
    #     ]
    #     computed_questions_order, computed_influence_matrices, computed_influences_from_data = (
    #         self.loader.get_influence_matrices2x2(make_it_row_stochastic=True))
    #     np_testing.assert_array_equal(
    #         expected_question_orders, computed_questions_order)
    #     np_testing.assert_array_equal(
    #         expected_influence_matrices,
    #         computed_influence_matrices)
    #     np_testing.assert_array_equal(
    #         expected_influences_from_data,
    #         computed_influences_from_data)

    # @parameterized.expand([
    #     ['with_one_missing',
    #         pd.DataFrame({
    #             "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
    #             "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
    #             "input":{"0":"self","1":"self","2":"other"},"value":{"0":"90","1":"51","2":"49"},
    #             "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
    #     [np.array([[0.9, 0.1],
    #                [0.49, 0.51]])],
    #     ],
    #     ['with_one_missing_and_one_empty',
    #         pd.DataFrame({
    #             "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
    #             "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
    #             "input":{"0":"self","1":"self","2":"other"},"value":{"0":"","1":"51","2":"49"},
    #             "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
    #     [np.array([[0.5, 0.5],
    #                [0.49, 0.51]])],
    #     ],
    #     ['with_only_one',
    #         pd.DataFrame({
    #             "sender":{"0":"pogs10.1"},
    #             "question":{"0":"GD_influence_surgery1"},
    #             "input":{"0":"self"},"value":{"0":"50",},
    #             "timestamp":{"0":"2020-01-16 14:15:11"}}),
    #     [np.array([[0.50, 0.50],
    #                [0.50, 0.50]])],
    #     ],
    #     ['with_larger_values',
    #         pd.DataFrame({
    #             "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
    #             "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
    #             "input":{"0":"self","1":"self","2":"other"},"value":{"0":"","1":"60","2":"90"},
    #             "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
    #     [np.array([[0.5, 0.5],
    #                [0.6, 0.4]])],
    #     ],
    #     ['with_duplicate_due_to_change_of_value',
    #         pd.DataFrame({
    #             "sender":{"0":"pogs10.1","0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
    #             "question":{"0":"GD_influence_surgery1","0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
    #             "input":{"0":"self","0":"self","1":"self","2":"other"},
    #             "value":{"0":"55","0":"5","1":"51","2":"49"},
    #             "timestamp":{"0":"2020-01-16 14:15:11","0":"2020-01-16 14:15:12","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
    #     [np.array([[0.05, 0.95],
    #                [0.49, 0.51]])],
    #     ]])
    # def test_get_influence_matrices2x2_mocked(self, name, influences, expected_influence_matrices):
    #     self.loader.influences = influences
    #     _, computed_influence_matrices, _ = (
    #         self.loader.get_influence_matrices2x2(make_it_row_stochastic=True))
    #     np_testing.assert_array_equal(
    #         expected_influence_matrices,
    #         computed_influence_matrices)

    # # =========================================================================
    # # ============== get_frustrations_in_simple_format ========================
    # # =========================================================================
    # def test_get_frustrations_in_simple_format(self):
    #     expected = pd.DataFrame({
    #         "Question":{0: "surgery"},
    #         "pogs10.1's answer":{0: "0"},
    #         "pogs10.2's answer":{0: "5"}})
    #     computed = self.loader.get_frustrations_in_simple_format()
    #     pd_testing.assert_frame_equal(expected, computed)

    # # =========================================================================
    # # =============== get_all_groups_info_in_one_dataframe ====================
    # # =========================================================================
    # def test_get_all_groups_info_in_one_dataframe(self):
    #     teams_log_list = {'s10': self.loader}
    #     dt = [
    #         ['s10', '1', 'asbestos', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '1', 'disaster', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '1', 'sports', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '1', 'school', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '1', 'surgery', '0.7', '0.8', '0.9', '0.1', '', '0.01', '0.99', '0.85', '', ''],
    #         ['s10', '2', 'asbestos', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '2', 'disaster', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '2', 'sports', '0.1111', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '2', 'school', '', '', '', '', '', '', '', '', '', ''],
    #         ['s10', '2', 'surgery', '0.5', '0.6', '0.51', '0.49', '1', '1.0', '0.0', '0.8', '', '']]
    #     expected = pd.DataFrame(dt, columns = [
    #         'Group', 'Person', 'Issue', 'Initial opinion',
    #         'Period1 opinion', 'Period1 wii', 'Period1 wij',
    #         'Period2 opinion', 'Period2 wii', 'Period2 wij',
    #         'Period3 opinion', 'Period3 wii', 'Period3 wij'])
    #     computed = bt.get_all_groups_info_in_one_dataframe(
    #         teams_log_list)
    #     pd_testing.assert_frame_equal(expected, computed)

    # # =========================================================================
    # # ============== compute_attachment_to_initial_opinion ====================
    # # =========================================================================
    # def test_compute_attachment_raises_when_not_matching_opinions(self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7]
    #     w12 = [0.1, 0.0, 0.2]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_initial_opinion(x1, x2, w12)

    # def test_compute_attachment_raises_when_not_matching_opinions_influence(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_initial_opinion(x1, x2, w12)

    # def test_compute_attachment_raises_when_start_k_was_not_0_or_1(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_initial_opinion(x1, x2, w12, start_k=-1)

    # def test_compute_attachment_to_initial_op_when_denom_to_sum_almost_zero(
    #         self):
    #     x1 = [0.8, 0.85, 0.92, 0.92]
    #     x2 = [0.6, 0.6, 0.7, 0.7]
    #     w12 = [0.1, 0.2, 0.1]
    #     expected_a11 = [
    #         (0.85 - 0.8) / (0.1 * (0.6 - 0.8)),
    #         np.nan,  # (0.92 - 0.8) / (0.85 - 0.8 + 0.2 * (0.6 - 0.85)),
    #         (0.92 - 0.8) / (0.92 - 0.8 + 0.1 * (0.7 - 0.92))
    #         ]
    #     expected_details = [{}, {'n/0': 1}, {}]
    #     computed_a11, details = bt.compute_attachment_to_initial_opinion(
    #         xi=x1, xj=x2, wij=w12, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)
    #     utils.assert_dict_equals({'1': expected_details}, {'1': details})

    # def test_compute_attachment_to_initial_opinion(self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0, 0.2]
    #     expected_a11 = [0.1/0.08, 0.5/0.1, 0.3/0.52]
    #     expected_details = [{}, {}, {}]
    #     computed_a11, details = bt.compute_attachment_to_initial_opinion(
    #         xi=x1, xj=x2, wij=w12, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)
    #     utils.assert_dict_equals({'1': expected_details}, {'1': details})

    # def test_compute_attachment_to_initial_opinion_when_start_k_equals_1(self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0, 0.2]
    #     expected_a11 = [0.5/0.1, 0.3/0.52]
    #     computed_a11, _ = bt.compute_attachment_to_initial_opinion(
    #         xi=x1, xj=x2, wij=w12, start_k=1, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # def test_compute_attachment_to_initial_opinion_when_division_by_zero(self):
    #     x1 = [0.2, 0.2, 0.2]
    #     x2 = [0.4, 0.4, 0.4]
    #     w12 = [0.1, 0.0]
    #     expected_a11 = [0 / 0.02,
    #                     np.nan]
    #     computed_a11, _ = bt.compute_attachment_to_initial_opinion(
    #         xi=x1, xj=x2, wij=w12, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # def test_compute_attachment_to_initial_when_division_by_zero_with_eps(self):
    #     x1 = [0.2, 0.2, 0.2]
    #     x2 = [0.4, 0.4, 0.4]
    #     w12 = [0.1, 0.0]
    #     eps = 0.01
    #     expected_a11 = [(0 + eps) / (0.02 + eps),
    #                     (0 + eps) / (0 + eps)]
    #     computed_a11, _ = bt.compute_attachment_to_initial_opinion(
    #         xi=x1, xj=x2, wij=w12, eps=eps)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # # =========================================================================
    # # =========== compute_attachment_to_opinion_before_discussion =============
    # # =========================================================================
    # def test_compute_attachment_before_disc_raises_when_not_matching_opinions(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7]
    #     w12 = [0.1, 0.0, 0.2]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_opinion_before_discussion(x1, x2, w12)

    # def test_compute_attachment_bef_disc_raises_when_not_matching_op_influence(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_opinion_before_discussion(x1, x2, w12)

    # def test_compute_attachment_before_disc_raises_when_start_k_was_not_0_or_1(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0]
    #     with self.assertRaises(ValueError):
    #         bt.compute_attachment_to_opinion_before_discussion(
    #             x1, x2, w12, start_k=-1)

    # def test_compute_attachment_to_opinion_before_discussion(self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0, 0.2]
    #     expected_a11 = [
    #         0.1/0.08,
    #         (0.6-0.1)/(0.2-0.1),
    #         (0.4-0.2)/(0.6-0.2+0.2*(0.7-0.6))]
    #     computed_a11, _ = bt.compute_attachment_to_opinion_before_discussion(
    #         xi=x1, xj=x2, wij=w12, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # def test_compute_attachment_to_opinion_before_disc_when_start_k_equals_1(
    #         self):
    #     x1 = [0.1, 0.2, 0.6, 0.4]
    #     x2 = [0.9, 0.4, 0.7, 0.5]
    #     w12 = [0.1, 0.0, 0.2]
    #     expected_a11 = [(0.6-0.1)/(0.2-0.1),
    #                     (0.4-0.2)/(0.6-0.2+0.2*(0.7-0.6))]
    #     computed_a11, _ = bt.compute_attachment_to_opinion_before_discussion(
    #         xi=x1, xj=x2, wij=w12, start_k=1, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # def test_compute_attachment_to_opinion_before_discussion_when_div_by_zero(
    #         self):
    #     x1 = [0.2, 0.2, 0.2]
    #     x2 = [0.4, 0.4, 0.4]
    #     w12 = [0.1, 0.0]
    #     expected_a11 = [0, np.nan]
    #     computed_a11, _ = bt.compute_attachment_to_opinion_before_discussion(
    #         xi=x1, xj=x2, wij=w12, eps=0)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # def test_compute_attachment_to_opinion_bef_disc_when_div_by_zero_with_eps(
    #         self):
    #     x1 = [0.2, 0.2, 0.2]
    #     x2 = [0.4, 0.4, 0.4]
    #     w12 = [0.1, 0.0]
    #     eps = 0.01
    #     expected_a11 = [(0.2 - 0.2 + eps) / (0.1 * (0.4 - 0.2) + eps),
    #                     eps / eps]
    #     computed_a11, _ = bt.compute_attachment_to_opinion_before_discussion(
    #         xi=x1, xj=x2, wij=w12, eps=eps)
    #     np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # # =========================================================================
    # # ================== compute_all_teams_attachments ========================
    # # =========================================================================
    # def test_compute_all_teams_attachments(self):
    #     # Attachment to the initial opinion.
    #     teams_data = {
    #         55: {
    #             'asbestos': {
    #                 'w12': [0.1, 0.0, 0.2],
    #                 'w21': [0.0, 0.0, 0.0],
    #                 'x1': [0.1, 0.2, 0.6, 0.4],
    #                 'x2': [0.9, 0.4, 0.7, 0.5]},
    #             'surgery': {
    #                 'w12': [0.35, 0.4, 0.5],
    #                 'w21': [0.25, 0.3, 0.3],
    #                 'x1': [0.6, 0.65, 0.7, 0.7],
    #                 'x2': [0.75, 0.5, 0.6, 0.7]}}}
    #     expected_attachments = {
    #         55: {
    #             'asbestos': {
    #                 'a11': [0.1/0.08, 0.5/0.1, 0.3/0.52],
    #                 'a22': [np.nan, -0.2/-0.5, -0.4/-0.2],    # Nan was -0.5/0.
    #                 'a11_nan_details': [{}, {}, {}],
    #                 'a22_nan_details': [
    #                     {'n/0': 1, 'xi[k]-xi[0]==0': 1, 'wij[k]==0': 1}, {}, {}]
    #                 },
    #             'surgery': {
    #                 'a11': [0.05/(0.35*0.15),
    #                         0.1/(0.05+0.4*-0.15),
    #                         0.1/(0.1+0.5*-0.1)],
    #                 'a22': [-0.25/(0.25*-0.15),
    #                         -0.15/(-0.25+0.3*0.15),
    #                         -0.05/(-0.15+0.3*0.1)],
    #                 'a11_nan_details': [{}, {}, {}],
    #                 'a22_nan_details': [{}, {}, {}]
    #                 }}}
    #     computed_attachments = bt.compute_all_teams_attachments(
    #         teams_data=teams_data,
    #         start_k=0,
    #         use_attachment_to_initial_opinion=True,
    #         eps=0)
    #     utils.assert_dict_equals(
    #         d1=expected_attachments,
    #         d2=computed_attachments,
    #         almost_number_of_decimals=6)

    # def test_compute_all_teams_attachments_with_start_k_equals_1(self):
    #     teams_data = {
    #         55: {
    #             'asbestos': {
    #                 'w12': [0.1, 0.0, 0.2],
    #                 'w21': [0.0, 0.0, 0.0],
    #                 'x1': [0.1, 0.2, 0.6, 0.4],
    #                 'x2': [0.9, 0.4, 0.7, 0.5]},
    #             'surgery': {
    #                 'w12': [0.35, 0.4, 0.5],
    #                 'w21': [0.25, 0.3, 0.3],
    #                 'x1': [0.6, 0.65, 0.7, 0.7],
    #                 'x2': [0.75, 0.5, 0.6, 0.7]}}}
    #     expected_attachments = {
    #         55: {
    #             'asbestos': {
    #                 'a11': [0.5/0.1, 0.3/0.52],
    #                 'a22': [-0.2/-0.5, -0.4/-0.2],
    #                 'a11_nan_details': [{}, {}],
    #                 'a22_nan_details': [{}, {}]},
    #             'surgery': {
    #                 'a11': [0.1/(0.05+0.4*-0.15),
    #                         0.1/(0.1+0.5*-0.1)],
    #                 'a22': [-0.15/(-0.25+0.3*0.15),
    #                         -0.05/(-0.15+0.3*0.1)],
    #                         'a11_nan_details': [{}, {}],
    #                         'a22_nan_details': [{}, {}]}}}
    #     computed_attachments = bt.compute_all_teams_attachments(
    #         teams_data=teams_data, start_k=1, eps=0)
    #     utils.assert_dict_equals(
    #         d1=expected_attachments,
    #         d2=computed_attachments,
    #         almost_number_of_decimals=6)

    # def test_compute_all_teams_attachments_to_before_discussion_opinion(self):
    #     teams_data = {
    #         55: {
    #             'asbestos': {
    #                 'w12': [0.1, 0.0, 0.2],
    #                 'w21': [0.0, 0.0, 0.0],
    #                 'x1': [0.1, 0.2, 0.6, 0.4],
    #                 'x2': [0.9, 0.4, 0.7, 0.5]},
    #             'surgery': {
    #                 'w12': [0.35, 0.4, 0.5],
    #                 'w21': [0.25, 0.3, 0.3],
    #                 'x1': [0.6, 0.65, 0.7, 0.7],
    #                 'x2': [0.75, 0.5, 0.6, 0.7]}}}
    #     expected_attachments = {
    #         55: {
    #             'asbestos': {
    #                 'a11': [0.1/0.08, 0.5/0.1, (0.4-0.2)/(0.6-0.2+0.2*(0.7-0.6))],
    #                 'a22': [np.nan, -0.2/-0.5, (0.5-0.4)/(0.7-0.4)],  # Nan was -0.5/0.
    #                 'a11_nan_details': [{}, {}, {}],
    #                 'a22_nan_details': [{'n/0': 1, 'wij[k]==0': 1}, {}, {}]
    #                 },
    #             'surgery': {
    #                 'a11': [0.05/(0.35*0.15),
    #                         0.1/(0.05+0.4*-0.15),
    #                         (0.7-0.65)/(0.7-0.65+0.5*(0.6-0.7))],  # denominator is 0.
    #                 'a22': [-0.25/(0.25*-0.15),
    #                         -0.15/(-0.25+0.3*0.15),
    #                         (0.7-0.5)/(0.6-0.5+0.3*(0.7-0.6))],
    #                 'a11_nan_details': [{}, {}, {'n/0': 1}],
    #                 'a22_nan_details': [{}, {}, {}]
    #                 }}}
    #     computed_attachments = bt.compute_all_teams_attachments(
    #         teams_data=teams_data,
    #         start_k=0,
    #         use_attachment_to_initial_opinion=False,
    #         eps=0)
    #     utils.assert_dict_equals(
    #         d1=expected_attachments,
    #         d2=computed_attachments,
    #         almost_number_of_decimals=6)
