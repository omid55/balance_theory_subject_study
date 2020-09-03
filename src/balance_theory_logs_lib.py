###############################################################################
# Omid55
# Start date:     03 Sep 2020
# Modified date:  03 Sep 2020
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# Module to load balance theory human subject study logs for every team.
###############################################################################

from __future__ import division, print_function, absolute_import, unicode_literals

import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from os.path import expanduser
from typing import Dict, List, Text, Tuple

import utils


def remove_early_duplicates(
        df: pd.DataFrame,
        groupby_columns: List[Text]) -> pd.DataFrame:
    """Removes all early rows of duplicates and only keeps the last one."""
    new_df = df.sort_values(by='timestamp')
    indices = []
    for l in new_df.groupby(by=groupby_columns).indices.values():
        indices.append(l[-1])
    new_df = new_df.iloc[indices]
    new_df.sort_values(by='timestamp', inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return new_df


class TeamLogsLoader(object):
    """Processes the logs of one team who have done the balance thoery POGS.

        Usage:
            loader = TeamLogsLoader(
                directory='/home/omid/Datasets/HumanSubjectBalanceTheoryData/team1_logs')

        Properties:
            team_id: The id of the existing team in this object.
            messages: All messages broadcasted among teammates.
            answers: All answers individuals sent.
            influences: All influences individuals reported.
            appraisals: All appraisals w.r.t. others individuals reported.
            members: List of usernames in the team.
    """

    # Task id to task name map.
    # Task ids are ordered based on how protocol was executed.
    taskid2taskname = {
        52: 'GD_solo_asbestos_initial',
        53: 'GD_group_asbestos1',
        62: 'GD_solo_asbestos1',
        57: 'GD_influence_asbestos1',
        167: 'GD_appraisal_asbestos1',
        61: 'GD_group_asbestos2',
        67: 'GD_solo_asbestos2',
        63: 'GD_influence_asbestos2',
        170: 'GD_appraisal_asbestos2',
        68: 'GD_group_asbestos3',
        70: 'GD_solo_asbestos3',
        69: 'GD_influence_asbestos3',
        172: 'GD_appraisal_asbestos3',
        
        71: 'GD_solo_disaster_initial',
        72: 'GD_group_disaster1',
        73: 'GD_solo_disaster1',
        74: 'GD_influence_disaster1',
        193: 'GD_appraisal_disaster1',
        79: 'GD_group_disaster2',
        76: 'GD_solo_disaster2',
        78: 'GD_influence_disaster2',
        194: 'GD_appraisal_disaster2',
        80: 'GD_group_disaster3',
        75: 'GD_solo_disaster3',
        77: 'GD_influence_disaster3',
        195: 'GD_appraisal_disaster3',

        83: 'GD_solo_sports_initial',
        89: 'GD_group_sports1',
        84: 'GD_solo_sports1',
        92: 'GD_influence_sports1',
        196: 'GD_appraisal_sports1',
        87: 'GD_group_sports2',
        85: 'GD_solo_sports2',
        91: 'GD_influence_sports2',
        197: 'GD_appraisal_sports2',
        88: 'GD_group_sports3',
        86: 'GD_solo_sports3',
        90: 'GD_influence_sports3',
        198: 'GD_appraisal_sports3',    
        
        94: 'GD_solo_school_initial',
        100: 'GD_group_school1',
        95: 'GD_solo_school1',
        104: 'GD_influence_school1',
        199: 'GD_appraisal_school1',
        99: 'GD_group_school2',
        96: 'GD_solo_school2',
        103: 'GD_influence_school2',
        200: 'GD_appraisal_school2',
        98: 'GD_group_school3',
        97: 'GD_solo_school3',
        102: 'GD_influence_school3',
        201: 'GD_appraisal_school3',        
        
        105: 'GD_solo_surgery_initial',
        109: 'GD_group_surgery1',
        106: 'GD_solo_surgery1',
        112: 'GD_influence_surgery1',
        202: 'GD_appraisal_surgery1',
        110: 'GD_group_surgery2',
        107: 'GD_solo_surgery2',
        113: 'GD_influence_surgery2',
        203: 'GD_appraisal_surgery2',
        111: 'GD_group_surgery3',
        108: 'GD_solo_surgery3',
        114: 'GD_influence_surgery3',
        204: 'GD_appraisal_surgery3'}
    
    def __init__(self, directory: Text):
        self.team_id = -1
        self.messages = None
        self.answers = None
        self.influences = None
        self.appraisals = None
        self._load(directory=directory)

    def _load(self, directory: Text) -> None:
        """Loads all logs for one team in the given directory.
        """
        logs_filepath = '{}/EventLog_*.csv'.format(directory)
        log_filepath = glob.glob(expanduser(logs_filepath))[0]
        logs = pd.read_csv(log_filepath, sep=';', skiprows=1)
        logs.dropna(axis=1, how='all', inplace=True)
        logs = logs.sort_values(by='Timestamp')

        # Subjects and their order.
        subjects_filepath_exp = '{}/SubjectExport_session_*.csv'.format(directory)
        subjects_filepath = glob.glob(expanduser(subjects_filepath_exp))[0]
        subjects = pd.read_csv(subjects_filepath, sep=';', skiprows=1)
        subjects_order = list(subjects['SubjectExternalId'])
        self.members = np.sort(subjects_order)
        subjects_arg_sort = np.argsort(subjects_order)
        team_size = len(self.members)

        # Ignoring all check_in logs.
        logs = logs[~(logs['EventType'] == 'CHECK_IN')]

        # Loading the completed task logs for learning the order of executed tasks.
        task_orders_filepath = '{}/CompletedTask_*.csv'.format(directory)
        task_filepath = glob.glob(expanduser(task_orders_filepath))[0]
        task_orders = pd.read_csv(task_filepath, sep=';', skiprows=1)
        task_orders.dropna(axis=1, how='all', inplace=True)

        completed_taskid2taskname = {}
        for _, row in task_orders.iterrows():
            completed_taskid2taskname[row.Id] = self.taskid2taskname[row.TaskId]
            
        answers_dt = []
        messages_dt = []
        influences_dt = []
        appraisals_dt = []
        for _, row in logs.iterrows():
            content_file_id = row.EventContent[9:]
            if not np.isnan(row.CompletedTaskId):
                question_name = completed_taskid2taskname[row.CompletedTaskId]
                sender = row.Sender
                timestamp = row.Timestamp
                event_type = row.EventType    
                json_file_path = '{}/EventLog/{}_*.json'.format(
                    directory, content_file_id)
                json_file = glob.glob(expanduser(json_file_path))
                if len(json_file) != 1:
                    print(
                        'WARNING1: json file for id: {} was not found'
                        ' in the EventLog folder.\n'.format(
                            content_file_id))
                else:
                    with open(json_file[0], 'r') as f:
                        content = json.load(f)
                        if 'type' in content and content['type'] == 'JOINED':
                            continue
                        if event_type == 'TASK_ATTRIBUTE':
                            if str.startswith(question_name, 'GD_solo'):
                                if question_name.endswith('_initial'):
                                    question_name = question_name.split(
                                        '_initial')[0] + '0'
                                answers_dt.append(
                                    [sender, question_name,
                                    content['attributeStringValue'], timestamp])
                            elif str.startswith(question_name, 'GD_influence'):
                                radio_selection = content[
                                    'attributeStringValue']
                                radio_selection = radio_selection.replace(
                                    '"', '').replace('[', '').replace(']', '')
                                influence_values = np.zeros(team_size)
                                for index, selection in enumerate(
                                    radio_selection.split(',')):
                                    if str.isdigit(selection):
                                        influence_values[
                                            index // 11] = selection
                                # Fixing the order of subjects' influences alphabetically.
                                influence_values = influence_values[subjects_arg_sort]
                                influences_dt.append(
                                    [sender, question_name,
                                    influence_values, timestamp])
                            elif str.startswith(question_name, 'GD_appraisal'):
                                appraisals_dt.append(
                                    [sender, question_name,
                                    content['attributeName'],
                                    content['attributeStringValue'], timestamp])
                            else:
                                print('WARNING4: There was an unexpected '
                                'question: {}\n'.format(question_name))
                        elif event_type == 'COMMUNICATION_MESSAGE':
                            if len(content['message']) > 0:
                                messages_dt.append(
                                    [sender, question_name,
                                    content['message'], timestamp])
                        else:
                            print('WARNING5: There was an unexpected'
                            ' EventType: {}\n'.format(event_type))

        # Answers.
        self.answers = remove_early_duplicates(
            df=pd.DataFrame(
                answers_dt,
                columns=['sender', 'question', 'value', 'timestamp']),
            groupby_columns=['sender', 'question'])

        # Messages.
        self.messages = pd.DataFrame(messages_dt, columns = [
            'sender', 'question', 'text', 'timestamp'])

        # Influences.
        self.influences = remove_early_duplicates(
            df=pd.DataFrame(
                influences_dt, columns = [
                    'sender', 'question', 'value', 'timestamp']),
            groupby_columns=['sender', 'question'])

        # Appraisals.
        sep_appraisals = remove_early_duplicates(
            df=pd.DataFrame(
                appraisals_dt, columns = [
                    'sender', 'question', 'attribute', 'value', 'timestamp']),
            groupby_columns=['sender', 'question', 'attribute'])
        members = np.unique(sep_appraisals['sender'])
        questions = np.unique(sep_appraisals['question'])
        appraisals_dt = []
        for question in questions:
            for member in members:
                appraisal_values = np.zeros(team_size)
                df = sep_appraisals[(sep_appraisals['sender'] == member) & (sep_appraisals['question'] == question)]
                for _, row in df.iterrows():
                    appraisal_values[int(row.attribute[-1]) - 1] = row.value
                # Fixing the order of subjects' appraisals alphabetically.
                appraisal_values = appraisal_values[subjects_arg_sort]
                appraisals_dt.append([member, question, appraisal_values, df.timestamp.iloc[-1]])
        self.appraisals = pd.DataFrame(
            appraisals_dt, columns = [
                'sender', 'question', 'value', 'timestamp'])

        # Sorts all based on timestamp.
        self.answers.sort_values(by='timestamp', inplace=True)
        self.messages.sort_values(by='timestamp', inplace=True)
        self.influences.sort_values(by='timestamp', inplace=True)
        self.appraisals.sort_values(by='timestamp', inplace=True)
        if self.members.size > 0 and '.' in self.members[0]:
            self.team_id = self.members[0].split('.')[0]

    # def get_influence_matrices(
    #         self,
    #         make_it_row_stochastic: bool = True
    #         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """Gets n * n influence matrices where n is the number of individuals.
        
    #     If empty or missing string, it fills with 100 - other one. If both
    #     empty or missing it fills both with 50.
    #     """
    #     influence_matrices = []
    #     influences_from_data = []
    #     users = self.users
    #     questions = np.unique(self.influences.question)
    #     for question in questions:
    #         influences = []
    #         for user in users:
    #             for input in ['self', 'other']:
    #                 this_influence = self.influences[
    #                     (self.influences.question == question) & (
    #                         self.influences.sender == user) & (
    #                             self.influences.input == input)]
    #                 val = ''
    #                 if len(this_influence.value) > 0:
    #                     # Because if there might be multiple log entry for the
    #                     #  same text box, we take the last one.
    #                     val = list(this_influence.value)[-1]
    #                 val = str(val).split('%')[0]
    #                 influences.append(val)
    #         tmp = influences[2]
    #         influences[2] = influences[3]
    #         influences[3] = tmp
    #         influences = np.reshape(influences, (2, 2))
    #         empty_strings = np.where(influences == '')
    #         influence_from_data = np.ones((2, 2), dtype=np.bool)
    #         for l in range(len(empty_strings[0])):
    #             i = empty_strings[0][l]
    #             j = empty_strings[1][l]
    #             if influences[i, 1-j] == '':
    #                 influences[i, 1-j] = 50
    #                 influence_from_data[i, 1-j] = False
    #             influences[i, j] = 100 - float(influences[i, 1-j])
    #             influence_from_data[i, j] = False
    #         influences = np.array(influences, dtype=np.float)
    #         if make_it_row_stochastic:
    #             influences = utils.make_matrix_row_stochastic(influences)
    #         influence_matrices.append(influences)
    #         influences_from_data.append(influence_from_data)
    #     question_names = [
    #         question[len('GD_influence_'):] for question in questions]
    #     return question_names, np.array(influence_matrices), np.array(
    #         influences_from_data)

    # def get_combined_messages(self) -> pd.DataFrame:
    #     pass


# def get_all_groups_info_in_one_dataframe(
#         teams_log: Dict[Text, TeamLogsLoader]) -> pd.DataFrame:
#     """Gets all teams' logs in one dataframe.
#     """
#     dt = []
#     issues = ['asbestos', 'disaster', 'sports', 'school', 'surgery']
#     for team_log in teams_log.values():
#         answers = team_log.get_answers_in_simple_format()
#         q_order, inf_matrices , from_data = team_log.get_influence_matrices2x2(
#             make_it_row_stochastic=True)
#         for index, user in enumerate(team_log.users):
#             user_ans = answers[['Question', user + '\'s answer']]
#             # user_conf = answers[['Question', user + '\'s confidence']]
#             for issue in issues:
#                 vals = []
#                 for i in range(4):
#                     op = user_ans[user_ans['Question'] == issue + str(i)]
#                     # co = user_ans[user_conf['Question'] == issue + str(i)]

#                     op_v = ''
#                     if len(op.values) > 0:
#                         op_v = op.values[0][1]
#                     if issue == 'asbestos' or issue == 'disaster':
#                         if len(op_v) > 0:
#                             op_v = op_v.replace('$', '')
#                             op_v = op_v.replace('dollars', '')
#                             op_v = op_v.replace('per person', '')
#                             op_v = op_v.replace(',', '')
#                             op_v = op_v.replace('.000', '000')
#                             op_v = op_v.replace(' million', '000000')
#                             op_v = op_v.replace(' mil', '000000')
#                             op_v = op_v.replace(' M', '000000')
#                             op_v = op_v.replace('M', '000000')
#                             op_v = op_v.replace(' k', '000')
#                             op_v = op_v.replace('k', '000')
#                             op_v = op_v.replace(' K', '000')
#                             op_v = op_v.replace('K', '000')
#                             op_v = '$' + op_v
#                     else:
#                         if len(op_v) > 0:
#                             op_v = op_v.replace('%', '')
#                             if op_v.isdigit() and float(op_v) > 2:
#                                 op_v = str(float(op_v) / 100)
#                             if '/' in op_v:
#                                 nums = op_v.split('/')
#                                 op_v = str(int(nums[0]) / int(nums[1]))
#                     vals.append(op_v)

#                     if i > 0:
#                         wii = ''
#                         wij = ''
#                         v = np.where(np.array(q_order) == issue + str(i))[0]
#                         if len(v) > 0:
#                             if from_data[v[0]][index, index] or from_data[v[0]][index, 1 - index]:
#                                 wii = round(inf_matrices[v[0]][index, index], 2)
#                                 wij = round(inf_matrices[v[0]][index, 1 - index], 2)
#                                 wii = str(wii)
#                                 wij = str(wij)
#                         vals.extend([wii, wij])
#                 dt.append(
#                     [team_log.team_id[3:], user.split('.')[1], issue] + vals)
#     data = pd.DataFrame(dt, columns = [
#         'Group', 'Person', 'Issue', 'Initial opinion',
#         'Period1 opinion', 'Period1 wii', 'Period1 wij',
#         'Period2 opinion', 'Period2 wii', 'Period2 wij',
#         'Period3 opinion', 'Period3 wii', 'Period3 wij'])
#     return data

