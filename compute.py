'''compute seq. similarity: uilog+time gap, uilog, high level uilog'''
import numpy as np
import pandas as pd
from datetime import datetime, timedelta 
from nltk.util import ngrams
import pickle
from collections import Counter, defaultdict
from functools import reduce
import itertools
from UserData import *


start_ref = ['', 'index.html', 'login.html', 
             'free-trial.html', 'freetrial.html','active-member.html',
             'member-task.html', 'register.html', 'member-management.html',
             'premium.html', 'subscription-management.html', 're-active.html',
             'forget-password.html', 'sso-mail.html', 'subscription-service-info.html',
             'reset-password.html', 'orderInvoice.html', '/', 
             'member-task.html#', 'freetrial.html#', 'order.html', 'contact-us.html', 
             'data-status.html', '###'
             'patent-vault-introduction.html', 'quality-insights-introduction.html',
             'design-search-introduction.html', 'project-list.html', 'sep']
start_uilog = ['Member0001', 'Member0002', 'Member0020']

def uilogCode_extract_advn(uilog):
    if uilog.count('Member') > 0:
        return 'Member'
    elif (uilog.count('Help') > 0) or (uilog.count('Contact') > 0) or ((uilog.count('Home') > 0)) \
    or (uilog.count('language') > 0):
        return 'Help'
    elif (uilog.count('EC') > 0) or (uilog.count('subscribe') > 0):
        return 'EC'
    elif (uilog.count('indexTour') > 0) or (uilog.count('projectOverviewTour') > 0) or \
        (uilog.count('patentInfoTour') > 0) or (uilog.count('searchResultTour')>0) or (uilog.count('onBoarding')>0):
        return 'onBoarding'
    elif uilog.count('Job') > 0:
        return 'Job'
    elif (uilog.count('VI') > 0) or (uilog.count('QI_') > 0) or \
        (uilog.count('Validity') > 0) or (uilog.count('validity') > 0):
        return 'QI'
    elif (uilog.count('PS_Save') > 0):
        return 'Project_Monitor'
    elif (uilog.count('nlq') > 0):
        return 'Search_nlq'
    elif (uilog.count('classification')>0) or (uilog.count('classkw_query')>0) or \
        (uilog.count('convert_to_query') > 0):
        return 'Search_number'
    elif (uilog.count('Search_hist') > 0):
        return 'Search_History'
    elif (uilog.count('Search_') > 0) or uilog.count('KeyExpand') or uilog.count('CorpAffli'): 
        return uilog.split('_')[0] + '_' + uilog.split('_')[1]
    elif (uilog.count('PS_Statistics') > 0) or (uilog.count('STS') > 0):
        return 'STS'
    elif (uilog.count('PageView') > 0): 
        if uilog.count('Export') > 0: 
            return uilog.split('_')[0] + '_Download'
        elif uilog.count('Show') > 0:
            return uilog.split('_')[0] + '_Bibl'
        elif uilog.count('Family') > 0:
            return uilog.split('_')[0] + '_Family'
        else:
            return uilog.split('_')[0] + '_' + uilog.split('_')[1]    
    
    elif (uilog.count('Project') > 0):
        if uilog.count('Monitor') > 0: 
            return uilog.split('_')[0] + '_Monitor'
        elif (uilog.count('Chart') > 0) or (uilog.count('Matrix') > 0) or (uilog.count('Analysis') > 0):
            return uilog.split('_')[0] + '_Analysis'
        elif (uilog.count('Collaborator')>0):
            return uilog.split('_')[0] + '_Collaborator'
        elif (uilog.count('Folder') > 0) or (uilog.count('Favorite') > 0) or \
            (uilog.count('View') > 0) or (uilog.count('Memo') > 0) or (uilog.count('Overview') > 0) or \
            (uilog.count('Sort') > 0):
            return uilog.split('_')[0] + '_Folder'
        else:
            return uilog
        
    elif (uilog.count('VS') > 0):
        return 'PageView_Compare'
    elif (uilog.count('List') > 0):
        return 'List'
    elif uilog.count('img') > 0:
        return 'DS'
    else: 
        return uilog

def connect_uilog(df, index):
    tmp = df.loc[index:(index+1),:].sort_values(by=['userId', 'first']).reset_index(drop=True)
    tmp = tmp.groupby(['userId']).agg(
                {'diff': sum, 'uilogCode': sum, 'uilogTime': sum, 'referer': sum, 
                 'click': sum, 'ref':sum, 'detail': sum,
                 'ref_first': 'first', 'uilog_first': 'first',
                 'ref_last': 'last', 'uilog_last': 'last'}).reset_index()
    df.at[(index+1),'diff'] = tmp['diff'][0]
    df.at[(index+1),'uilogCode'] = tmp['uilogCode'][0]
    df.at[(index+1),'uilogTime'] = tmp['uilogTime'][0]
    df.at[(index+1),'referer'] = tmp['referer'][0]
    df.at[(index+1),'click'] = tmp['click'][0]
    df.at[(index+1),'ref'] = tmp['ref'][0]
    df.at[(index+1),'detail'] = tmp['detail'][0]
    df.at[(index+1),'ref_first'] = tmp['ref_first'][0]
    df.at[(index+1),'ref_last'] = tmp['ref_last'][0]
    df.at[(index+1),'uilog_first'] = tmp['uilog_first'][0]
    df.at[(index+1),'uilog_last'] = tmp['uilog_last'][0]
    df.drop(index, inplace=True)
    return df

def Ngram_similarity(list1, list2, n_gram):
    subseq1 = list()
    subseq2 = list()

    gram1 = Counter(ngrams(list1, n_gram))
    gram2 = Counter(ngrams(list2, n_gram))
    target = gram1 + gram2
    for index, element in enumerate(target.keys()):
        subseq1.append(gram1.get(element, 0))
        subseq2.append(gram2.get(element, 0))

    n1 = max(sum(subseq1), 1)
    n2 = max(sum(subseq2), 1)
    tmp = [(element/n1 - subseq2[index]/n2)**2 for index, element in enumerate(subseq1)]
    return (1/(2**0.5)) * (sum(tmp))**0.5

def Pickle(file_name, output):
    file = open(str(file_name) + '.pickle', 'wb') 
    pickle.dump(output, file)
    file.close()

def get_pipeline(ids, start, end, delta):
    ids = uuid.UUID(ids)
    start = datetime.strptime(str(start), '%Y-%m-%d')
    end = datetime.strptime(str(end), '%Y-%m-%d') - timedelta(days=delta)
    pipeline = [{'$match': {'userId.id': {'$eq': ids}, 'startTime': {'$lte': end, '$gte': start}}},
                {'$project': {'_id': 0,'userId.id': 1, 'sessionId':1, 
                              'startTime':1, 'log.uiLog.uilogCode':1, 
                              'log.uiLog.uilogTime':1, 'request.header.Referer':1}},
                {'$group': {'_id': '$sessionId', 'userId': {'$addToSet':'$userId.id'}, 
                            'first': {'$first': '$startTime'}, 'last': {'$last': '$startTime'}, 
                            'uilogCode': {'$push': '$log.uiLog.uilogCode'},
                            'uilogTime': {'$push': '$log.uiLog.uilogTime'},
                            'referer': {'$push': '$request.header.Referer'}}},
                {'$project': {'sessionId': '$sessionId', 'userId': '$userId', 
                              'first':'$first', 'diff':{'$subtract':['$last','$first']}, 
                              'uilogCode': '$uilogCode', 'uilogTime': '$uilogTime',
                              'referer': '$referer'}}]
    data = a.UiLog(pipeline)
    if len(data) > 0:
        return data
    else:
        pass

def clickstream_prepocessing(id_list):
    clickstream = pd.DataFrame()
    for index, element in enumerate(id_list):
        start = data[data['id']==element]['auth_start_date'][index]
        end = data[data['id']==element]['auth_end_date'][index]
        clickstream = clickstream.append(get_pipeline(element, start, end, 0), ignore_index=True)
        
    clickstream['userId'] = clickstream['userId'].map(lambda x: x[0])
    clickstream['userId'] = clickstream['userId'].map(lambda x: str(x))
    clickstream['uilogCode'] = clickstream['uilogCode'].map(lambda x: list(itertools.chain(*x)))
    clickstream['uilogTime'] = clickstream['uilogTime'].map(lambda x: list(itertools.chain(*x)))
    clickstream['uilogTime'] = clickstream['uilogTime'].map(lambda x: 
            [datetime.strptime(xx.split('T')[0] + ' ' +(xx.split('T')[1]).split('Z')[0], 
                        '%Y-%m-%d %H:%M:%S.%f') for xx in x])
    clickstream['click'] = clickstream['uilogCode'].map(lambda x: len(x))
    clickstream['ref'] = clickstream['referer'].map(
        lambda x: [(xx.split('.com/')[1]).split('?')[0] for xx in x])
    clickstream['detail'] = clickstream['referer'].map(
        lambda x: [xx.split('?')[1] for xx in x if xx.count('?')])

    clickstream['ref_first'] = clickstream['ref'].map(lambda x: x[0])
    clickstream['ref_last'] = clickstream['ref'].map(lambda x: x[(len(x)-1)])
    clickstream['uilog_first'] = clickstream['uilogCode'].map(lambda x: x[0])
    clickstream['uilog_last'] = clickstream['uilogCode'].map(lambda x: x[(len(x)-1)])

    # pre-processing clickstream
    ## 1. ascending scanning: connect 'beforeClose' from different sessionId
    clickstream = clickstream.sort_values(by=['userId', 'first'], ascending=False).reset_index(drop=True)

    for index, row in clickstream[['ref_first', 'uilog_first']].iterrows():
        if (row['ref_first'] not in start_ref) & (row['uilog_first']=='beforeClose'):
            if clickstream.loc[index,'userId'] == clickstream.loc[(index+1),'userId']:
                clickstream = connect_uilog(clickstream, index)
            else:
                pass
        else:
            pass

    ## 2. check whether current referer exist in former referer or not
    clickstream = clickstream.sort_values(by=['userId', 'first'], ascending=False).reset_index(drop=True)
    upper = len(clickstream)

    for index, row in clickstream[['ref_first', 'referer', 'uilog_first']].iterrows():
        if (row['ref_first'] not in start_ref) & \
            (row['uilog_first'] not in start_uilog) & (index < upper):
            if clickstream.loc[index,'userId'] == clickstream.loc[(index+1),'userId']:
                tmp = set(clickstream.loc[(index+1),:]['detail'])
                if len(set(clickstream.loc[index, :]['detail']).intersection(tmp))>0:
                    clickstream = connect_uilog(clickstream, index)
                else:
                    pass
            else:
                pass
        else:
            pass

    clickstream = clickstream.sort_values(by=['userId', 'first']).reset_index(drop=True)
    clickstream['unique'] = clickstream['uilogCode'].map(lambda x: len(set(x)))
    clickstream['uilog_unique'] = clickstream['uilogCode'].map(lambda x: set(x))

    # sort
    for index in range(len(clickstream)):
        clickstream.at[index, 'uilogCode'] = [x for _,x in sorted(zip(clickstream['uilogTime'][index], clickstream['uilogCode'][index]))]
        clickstream.at[index, 'ref'] = [x for _,x in sorted(zip(clickstream['uilogTime'][index], clickstream['ref'][index]))]
        
    clickstream['uilogTime'] = clickstream['uilogTime'].map(lambda x: sorted(x))
    clickstream['gap'] = clickstream['uilogTime'].map(lambda x: [yy-xx for xx, yy in zip(x, x[1:])])
    clickstream['gap'] = clickstream['gap'].map(lambda x: [xx.seconds for xx in x])
    bins=[0, 1, 10, 100, 1000, np.inf]
    clickstream['gap'] = clickstream['gap'].map(lambda x: [np.digitize(xx,bins) for xx in x])
    clickstream['gap'] = clickstream['gap'].map(lambda x: [str(xx) for xx in x])
    clickstream['high_uilogCode'] = clickstream['uilogCode'].map(
        lambda x: [uilogCode_extract_advn(xx) for xx in x])

    # sequence
    clickstream.loc[:, 'sequence'] = [x[:] for x in clickstream['high_uilogCode']]
    for i in range(len(clickstream)):
        for index, element in enumerate(clickstream['gap'][i]):
            clickstream['sequence'][i].insert(2*index+1, element)

    return clickstream


# get data
start = '2019-10-01'
end = '2020-06-16'
startDate = datetime.strptime(start, '%Y-%m-%d')
endDate = datetime.strptime(end, '%Y-%m-%d')
a = UserBehaviorData(start, end)
a.format_date()

Query2 = '''SELECT member_id, product_code, auth_start_date, auth_end_date FROM model_privilege
         where (created_date_time > '{0}' and created_date_time < '{1}')
         and product_code = 'Lic_freetrial' '''.format(startDate, endDate)

privilege = a.Privilege(Query2)
len(privilege)
id_list = [str(x) for x in privilege['member_id']]
Query1 = '''SELECT id, source, email_domain, register_date_time FROM dbo.member
            where id in ''' + str(tuple(id_list))
member = a.MemberInfo(Query1, mode=1)

data = pd.merge(member, privilege, left_on='id', right_on='member_id')
data.drop(['index', 'member_id'], axis=1, inplace=True)

id_list = [str(x) for x in data['id']]
clickstream = clickstream_prepocessing(id_list)
print(len(clickstream))

# file = open('clickstream1217.pickle', 'wb')
# pickle.dump(clickstream, file)
# file.close()

# similarity
# n = len(clickstream)
# for gram in range(1, 6):
#     similarity = np.zeros((n, n))
#     for i in range(n):
#         similarity[i, i] = 0
#         for j in range(i+1, n):
#             similarity[i, j] = Ngram_similarity(
#                 clickstream['sequence'][i], clickstream['sequence'][j], gram)

#     X = similarity
#     a_triu = np.triu(X, k=0)
#     a_tril = np.tril(X, k=0)
#     a_diag = np.diag(np.diag(X))
#     X = a_triu + a_triu.T - a_diag

#     # output
#     Pickle(str(gram), X)


# # compute similarity for HC 
# with open("1.pickle", "rb") as file:
#     gram_1 = pickle.load(file)
# # with open("2.pickle", "rb") as file:
# #     gram_2 = pickle.load(file)
# # with open("3.pickle", "rb") as file:
# #     gram_3 = pickle.load(file)
# # with open("4.pickle", "rb") as file:
# #     gram_4 = pickle.load(file)
# with open("5.pickle", "rb") as file:
#     gram_5 = pickle.load(file)

# def HC_plot(X):
#     Z = sch.linkage(X, method = 'ward')
#     sch.dendrogram(Z)
#     c, coph_dists = cophenet(Z, pdist(X))
#     return {'HC_parms': Z, 'cophenet': c}

# Pickle('HC_1', HC_plot(gram_1))
# # Pickle('HC_2', HC_plot(gram_2))
# # Pickle('HC_3', HC_plot(gram_3))
# # Pickle('HC_4', HC_plot(gram_4))
# Pickle('HC_5', HC_plot(gram_5))
