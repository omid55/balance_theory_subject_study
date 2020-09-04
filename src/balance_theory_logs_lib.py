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
            questions_order: The order that questions were asked from subjects.
    """

    # Task id to task name map.
    # Task ids are ordered as follows in the protocol implemented in POGS.
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
            
        self.questions_order = []
        answers_dt = []
        messages_dt = []
        influences_dt = []
        appraisals_dt = []
        for _, row in logs.iterrows():
            content_file_id = row.EventContent[9:]
            if not np.isnan(row.CompletedTaskId):
                question_name = completed_taskid2taskname[row.CompletedTaskId]
                if not self.questions_order or (len(self.questions_order) > 0 and self.questions_order[-1] != question_name):
                    self.questions_order.append(question_name)
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

    def get_answers(self) -> pd.DataFrame:
        return self.answers.sort_values(['question', 'sender'])
    
    def get_messages(self) -> pd.DataFrame:
        return self.messages.sort_values(['question', 'sender'])

    def get_influences(self) -> pd.DataFrame:
        return self.influences.sort_values(['question', 'sender'])

    def get_appraisals(self) -> pd.DataFrame:
        return self.appraisals.sort_values(['question', 'sender'])

    def get_influence_matrices(
            self, make_it_row_stochastic: bool = True
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets n * n influence matrices where n is the number of individuals.
        
        If empty or missing string, it fills ...
        """
        pass

    def get_combined_messages(self) -> pd.DataFrame:
        pass


def get_all_groups_info_in_one_dataframe(
        teams_log: List[TeamLogsLoader],
        with_space: bool = False) -> pd.DataFrame:
    """Gets all teams' logs in one dataframe.

    Args:

    Returns:

    Raises:
        ValueError:
    """
    dt = []
    issues = ['asbestos', 'disaster', 'sports', 'school', 'surgery']
    for team_log in teams_log:
        team_id = team_log.team_id
        answers = team_log.answers
        influences = team_log.influences
        appraisals = team_log.appraisals
        team_size = len(team_log.members)
        for issue in issues:
            for _, member in enumerate(team_log.members):
                member_answers = []
                member_influences = []
                member_appraisals = []
                for i in range(4):
                    answer = answers[
                        (answers['sender'] == member) &
                        (answers['question'] == 'GD_solo_{}{}'.format(
                            issue, str(i)))]
                    ans = ''
                    if len(answer) > 0:
                        if len(answer) != 1:
                            raise ValueError(
                                'E1: There was a problem in'
                                ' answers dataframe. It has more than one'
                                ' row (it has {} rows) for {} and {}.'
                                .format(len(answer), issue, member))
                        else:
                            ans = answer.iloc[0]['value']
                    member_answers.append(ans)
                    
                    if i > 0:
                        influence = influences[
                            (influences['sender'] == member) &
                            (influences['question'] == 'GD_influence_{}{}'
                            .format(issue, str(i)))]
                        inf = ['' for _ in range(team_size)]
                        if len(influence) > 0:
                            if len(influence) != 1:
                                raise ValueError(
                                    'E1: There was a problem in influences'
                                    ' dataframe. It has more than one row'
                                    ' (it has {} rows) for {} and {}.'.format(
                                        len(influence), issue, member))
                            else:
                                inf = list(influence.iloc[0]['value'])
                        if with_space:
                            member_influences.extend(inf + [''])
                        else:
                            member_influences.extend(inf)

                        appraisal = appraisals[(
                            appraisals['sender'] == member) &
                            (appraisals['question'] == 'GD_appraisal_{}{}'
                            .format(issue, str(i)))]
                        appr = ['' for _ in range(team_size)]
                        if len(appraisal) > 0:
                            if len(appraisal) != 1:
                                raise ValueError(
                                    'E1: There was a problem in appraisal'
                                    ' dataframe. It has more than one row'
                                    ' (it has {} rows) for {} and {}.'.format(
                                        len(appraisal), issue, member))
                            else:
                                appr = list(appraisal.iloc[0]['value'])
                        if with_space and i != 3:
                            member_appraisals.extend(appr + [''])
                        else:
                            member_appraisals.extend(appr)
                if with_space:
                    dt.append([team_id, issue, member] + member_answers + [''] + member_influences + member_appraisals)
                else:
                    dt.append([team_id, issue, member] + member_answers + member_influences + member_appraisals)
        if with_space:
            dt.append(['' for _ in range(len(dt[0]))])
    if with_space:
        columns = [
            'Group', 'Issue', 'Person', 'Initial opinion',
            'Period1 opinion', 'Period2 opinion', 'Period3 opinion', '',
            'Period1 w1', 'Period1 w2', 'Period1 w3', '',
            'Period2 w1', 'Period2 w2', 'Period2 w3', '',
            'Period3 w1', 'Period3 w2', 'Period3 w3', '',
            'Period1 a1', 'Period1 a2', 'Period1 a3', '',
            'Period2 a1', 'Period2 a2', 'Period2 a3', '',
            'Period3 a1', 'Period3 a2', 'Period3 a3']
    else:
        columns = [
            'Group', 'Issue', 'Person', 'Initial opinion',
            'Period1 opinion', 'Period2 opinion', 'Period3 opinion',
            'Period1 w1', 'Period1 w2', 'Period1 w3',
            'Period2 w1', 'Period2 w2', 'Period2 w3',
            'Period3 w1', 'Period3 w2', 'Period3 w3',
            'Period1 a1', 'Period1 a2', 'Period1 a3',
            'Period2 a1', 'Period2 a2', 'Period2 a3',
            'Period3 a1', 'Period3 a2', 'Period3 a3']

    data = pd.DataFrame(dt, columns=columns)
    return data

