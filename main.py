''''''
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime, timedelta
import pickle
import itertools
from matplotlib.cbook import flatten
from nltk.util import ngrams
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import cophenet, fcluster, sch
from scipy.spatial.distance import pdist
from FilterInner import *
from UserData import *

# import raw data
path = '.....'
with open(path + "clickstream1217.pickle", "rb") as file:
    raw_clickstream = pickle.load(file)
with open(path + "2.pickle", "rb") as file:
    XX = pickle.load(file)
with open(path + "clustering1217.pickle", "rb") as file:
    clustering = pickle.load(file)
with open(path + "gmm1217.pickle", "rb") as file:
    gmm = pickle.load(file)
with open(path + "Exp4.2_result.pickle", "rb") as file:
    raw_result = pickle.load(file)


# define func.
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

def get_pipeline(ids, start, end, delta):
    ids = uuid.UUID(ids)
    end = datetime.strptime(str(start), '%Y-%m-%d') + timedelta(days=delta)
    start = datetime.strptime(str(start), '%Y-%m-%d')

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
        clickstream = clickstream.append(get_pipeline(element, start, end, 5), ignore_index=True)
        
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

    # # pre-processing clickstream #!
    # ## 1. ascending scanning: connect 'beforeClose' from different sessionId
    # clickstream = clickstream.sort_values(by=['userId', 'first'], ascending=False).reset_index(drop=True)

    # for index, row in clickstream[['ref_first', 'uilog_first']].iterrows():
    #     if (row['ref_first'] not in start_ref) & (row['uilog_first']=='beforeClose'):
    #         if clickstream.loc[index,'userId'] == clickstream.loc[(index+1),'userId']:
    #             clickstream = connect_uilog(clickstream, index)
    #         else:
    #             pass
    #     else:
    #         pass

    # ## 2. check whether current referer exist in former referer or not
    # clickstream = clickstream.sort_values(by=['userId', 'first'], ascending=False).reset_index(drop=True)
    # upper = len(clickstream)

    # for index, row in clickstream[['ref_first', 'referer', 'uilog_first']].iterrows():
    #     if (row['ref_first'] not in start_ref) & \
    #         (row['uilog_first'] not in start_uilog) & (index < upper):
    #         if clickstream.loc[index,'userId'] == clickstream.loc[(index+1),'userId']:
    #             tmp = set(clickstream.loc[(index+1),:]['detail'])
    #             if len(set(clickstream.loc[index, :]['detail']).intersection(tmp))>0:
    #                 clickstream = connect_uilog(clickstream, index)
    #             else:
    #                 pass
    #         else:
    #             pass
    #     else:
    #         pass

    # clickstream = clickstream.sort_values(by=['userId', 'first']).reset_index(drop=True)
    # clickstream['unique'] = clickstream['uilogCode'].map(lambda x: len(set(x)))
    # clickstream['uilog_unique'] = clickstream['uilogCode'].map(lambda x: set(x))

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
    try:
        clickstream.loc[:, 'sequence'] = [x[:] for x in clickstream['high_uilogCode']]
    except:
        clickstream.loc[:, 'sequence'] = clickstream['high_uilogCode']

    for i in range(len(clickstream)):
        for index, element in enumerate(clickstream['gap'][i]):
            clickstream['sequence'][i].insert(2*index+1, element)

    return clickstream

def compute_similarity(list2, n_gram):
    similarity  = list()
    for i in range(len(raw_clickstream)):
        list1 = raw_clickstream['sequence'][i]
        similarity.append(Ngram_similarity(list1, list2, n_gram))
    return similarity 

def compute_similarity_main(raw_clickstream, validation_clickstream, n_gram):
    n = len(validation_clickstream)
    m = len(raw_clickstream)
    similarity = np.zeros((n, m))
    for i in range(n):
        similarity[i,:] = compute_similarity(validation_clickstream['sequence'][i], n_gram)
    len(similarity)
    return similarity

def validation_HC(HC_num, raw_similarity, raw_HC_parms, validation_similarity, validation_clickstream):
    X = validation_similarity
    XX = raw_similarity
    tmp = Counter(fcluster(raw_HC_parms, HC_num, criterion='maxclust'))
    tmp = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}
    tmp = [x for x in tmp.keys()]
    name = range(1, 10)
    mapping = dict(zip(tmp, name))
    
    YY = pd.Series(fcluster(raw_HC_parms, HC_num, criterion='maxclust'))
    YY = YY.map(mapping)
    
    X_train , X_test, y_train ,y_test = train_test_split(XX, YY, random_state=0, test_size=0.2)  

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("準確率: ", knn.score(X_test, y_test))

    validation_clickstream['clusters'] = knn.predict(X)
    print(validation_clickstream['clusters'].value_counts())
    
    tmp = validation_clickstream.groupby(['userId']).agg({'clusters': list}).reset_index()
    output = pd.merge(data, tmp, how='outer', left_on='id', right_on='userId')
    output = output[~(output['clusters'].isna())]
    output['clusters'] = output['clusters'].map(lambda x: Counter(x))

    output = pd.concat([output, output['clusters'].apply(pd.Series)], axis=1) 
    output.fillna(0, inplace=True)

    column_list = [x for x in range(1, HC_num+1)]
    for col in column_list:
        if col not in output.columns:
            output[col] = 0
    output['session_count'] = output[column_list].sum(axis=1)
    return output

def validation_GMM(HC_num, raw_similarity, raw_result, validation_clickstream):
    column_list = [x for x in range(1, HC_num+1)]
    Y = validation_clickstream[column_list].div(validation_clickstream['session_count'], axis=0)
    X =  raw_result[column_list].div(raw_result['session_count'], axis=0)
    
    X_train , X_test, y_train ,y_test = train_test_split(
        X, raw_result['GMM'], random_state=0, test_size=0.3)  

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    print("準確率: ", knn.score(X_test, y_test))

    validation_clickstream['GMM'] = knn.predict(Y)
    print(validation_clickstream['GMM'].value_counts())
    return validation_clickstream


# get data: user freetrial expired
end0 = str(datetime.today() + timedelta(days=9))[:10]
start = str(datetime.today() - timedelta(days=5))[:10]
end = str(datetime.today() + timedelta(days=2))[:10]
endDate0 = datetime.strptime(end0, '%Y-%m-%d')
startDate = datetime.strptime(start, '%Y-%m-%d')
endDate = datetime.strptime(end, '%Y-%m-%d')

Query2 = '''SELECT member_id, product_code, auth_start_date, auth_end_date FROM model_privilege
            where product_code = 'Lic_freetrial'
            and (auth_start_date = '{0}' and  auth_end_date = '{1}') 
            or (auth_start_date = '{0}' and  auth_end_date = '{2}')'''.format(startDate, endDate, endDate0)

a = UserBehaviorData(start, end)
a.format_date()
privilege = a.Privilege(Query2)
print(len(privilege))

if len(privilege) > 0:
    id_list = [str(x) for x in privilege['member_id']]
    Query1 = "SELECT id, source, email_domain, register_date_time FROM dbo.member where id in ('"
    Query1 += "','".join(id_list)
    Query1 += "')"
        
    member = a.MemberInfo(Query1, mode=1)
    print(len(member))

    data = pd.merge(member, privilege, left_on='id', right_on='member_id')
    data.drop(['index', 'member_id'], axis=1, inplace=True)

    # get every member session-based uilog in range and preprocessing
    id_list = [str(x) for x in data['id']]
    clickstream = clickstream_prepocessing(id_list)
    print(len(clickstream))

    # # compute similaruty for raw data and new data
    similarity = compute_similarity_main(raw_clickstream, clickstream, 2)

    # predict by KNN for HC.
    clickstream = validation_HC(5, XX, clustering['HC_parms'], similarity, clickstream)
    # print(clickstream.columns)
    # print(clickstream['clusters'].value_counts())

    # predict by KNN for GMM
    clickstream = validation_GMM(5, XX, raw_result, clickstream)

    # final tune
    clickstream['GMM'] = clickstream['GMM'].replace({6:4, 5:3, 4:2})
    clickstream.loc[((clickstream['GMM']==2)&(clickstream['session_count']>=5))|
                    (clickstream['session_count']>=5), 'GMM'] = 4
    
    
    # output and auto-email
    output = clickstream[['id', 'register_date', 'auth_start_date',
                 'auth_end_date', 1, 2, 3, 4, 5, 'session_count', 'GMM']]
    output.sort_values(by='GMM', ascending=False, inplace=True)

    AutoEmail(output, 'Potential Paying Users on {}','%Y-%m-%d','XXX@xxxxxx.com', '', 
            'Potential Paying Users on {}.xlsx', 1, 'Potential Paying Users ')
    
    output = output[['id', 'GMM']]
    output.columns = ['id', 'group']
