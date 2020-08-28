import pandas as pd
import recordlinkage
import recordlinkage.preprocessing
import numpy as np

class LocalityDedup:
    def __init__(self,locality_df, locality_df2=None,locality_col='locality',booth_num_col='booth_num',threshold=0.8):
        self.locality_df = locality_df
        self.locality_df2 = locality_df2
        self.locality_col = locality_col
        self.booth_num_col = booth_num_col
        self.threshold = threshold

        #Create in-class
        self.candidate_links = None
        self.compare_booths = None
        self.features = None
        self.master_names = None
        self.short_list_df = None


    def add_metaphones(self,locality_df):
        locality_df['metaphone'] = recordlinkage.preprocessing.phonetic(locality_df[self.locality_col],method="metaphone")

    def get_pairs(self):
        indexer = recordlinkage.Index()
        indexer.full()
        self.candidate_links = indexer.index(self.locality_df)

    def compare_builder(self):
        compare_booths = recordlinkage.Compare()

        # Get Levenshtein score booth name
        compare_booths.string(self.locality_col,self.locality_col, method='levenshtein', threshold=None, label='name_lv_score')

        # Get Jaro-Winkler Score for booth name
        compare_booths.string(self.locality_col, self.locality_col, method='jarowinkler', threshold=None, label='name_jw_score')

        # Get Jaro-winkler score for the way the name is pronounced
        compare_booths.string('metaphone', 'metaphone', method='levenshtein', threshold=None, label='metaphone')

        # Get score for how far apart the booth numbers are
        compare_booths.numeric(self.booth_num_col, self.booth_num_col, label='booth_number_score', method="gauss", offset=3, scale=5)

        self.compare_booths = compare_booths

    def compute(self):
        self.features = self.compare_booths.compute(self.candidate_links, self.locality_df)

    def consolidate_scores(self):
        match_index = self.features.index.values
        master_name_index = [x[0] for x in match_index]
        master_names = self.locality_df.iloc[master_name_index]

        duplicate_name_index = [x[1] for x in match_index]
        duplicate_names = self.locality_df.iloc[duplicate_name_index]

        master_names = master_names.rename(columns={self.locality_col: 'master_name', self.booth_num_col: "new_booth_num"})
        master_names['duplicate_name'] = duplicate_names[self.locality_col].tolist()
        master_names['duplicate_booth_num'] = duplicate_names[self.booth_num_col].tolist()
        master_names['name_jw_score'] = self.features.name_jw_score.tolist()
        master_names['name_lv_score'] = self.features.name_lv_score.tolist()
        master_names['booth_score'] = self.features.booth_number_score.tolist()
        master_names['metaphone'] = self.features.metaphone.tolist()
        # master_names['duplicate_candidate_name'] = duplicate_names.name2.tolist()

        master_names['average_score'] = master_names[['name_jw_score', 'name_lv_score', 'booth_score']] \
            .apply(lambda x: x.mean(), axis=1).round(2)

        master_names = master_names[['master_name', 'duplicate_name', 'new_booth_num', 'duplicate_booth_num',
                                     'name_jw_score', "name_lv_score", 'booth_score', 'average_score']]

        self.master_names = master_names


    def get_shortlist(self, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        self.short_list_df = self.master_names[self.master_names.average_score >= threshold] \
            .sort_values(by=['average_score'], ascending=False).reset_index(drop=True)

    def __call__(self, *args, **kwargs):
        self.add_metaphones(self.locality_df)
        self.get_pairs()
        self.compare_builder()
        self.compute()
        self.consolidate_scores()
        self.get_shortlist()



class LocalityMatch(LocalityDedup):

    def get_pairs(self):
        indexer = recordlinkage.Index()
        indexer.full()
        self.candidate_links = indexer.index(self.locality_df,self.locality_df2)

    def compute(self):
        self.features = self.compare_booths.compute(self.candidate_links, self.locality_df,self.locality_df2)

    def consolidate_scores(self):
        match_index = self.features.index.values
        master_name_index = [x[0] for x in match_index]
        new_names = self.locality_df.iloc[master_name_index].copy()

        old_name_index = [x[1] for x in match_index]
        old_names = self.locality_df2.iloc[old_name_index]

        new_names['old_name'] = old_names.master_name.tolist()
        new_names['old_booth_rank'] = old_names.booth_rank.tolist()
        new_names['name_jw_score'] = self.features.name_jw_score.tolist()
        new_names['name_lv_score'] = self.features.name_lv_score.tolist()
        new_names['booth_score'] = self.features.booth_number_score.tolist()
        new_names['metaphone'] = self.features.metaphone.tolist()
        new_names['average_score'] = new_names[['name_jw_score', 'name_lv_score', 'booth_score']] \
            .apply(lambda x: x.mean(), axis=1).round(2)
        new_names = new_names[['master_name', 'old_name', 'booth_rank', 'old_booth_rank',
                               'name_jw_score', "name_lv_score", 'booth_score', 'average_score']].reset_index(drop=True)
        self.new_names = new_names

    def get_matched_df(self):
        new_name_matched = self.new_names.iloc[self.new_names.groupby('master_name')['average_score'].idxmax()].copy()
        new_name_matched['matched_on'] = 'new_name'
        missing_old_localities = self.locality_df2.loc[~self.locality_df2.master_name.isin(new_name_matched.old_name)][
            'master_name']
        to_add_old = self.new_names.query('old_name in @missing_old_localities').reset_index(drop=True)
        to_add_old = to_add_old.iloc[to_add_old.groupby('old_name')['average_score'].idxmax()]
        to_add_old['matched_on'] = 'old_name'
        self.all_matched = new_name_matched.append(to_add_old)

    def get_connected_components(self):
        matched_index = pd.MultiIndex.from_frame(self.all_matched[['master_name', 'old_name']])
        cc = recordlinkage.ConnectedComponents()
        self.connected_components = cc.compute(matched_index)



    def __call__(self, *args, **kwargs):
        self.add_metaphones(self.locality_df)
        self.add_metaphones(self.locality_df2)
        self.get_pairs()
        self.compare_builder()
        self.compute()
        self.consolidate_scores()
        self.get_matched_df()
        self.get_connected_components()
#
#
# import pandas as pd
# #from src.booth_cleanup.locality_dedup import LocalityDedup
# #from src.booth_cleanup.locality_dedup import LocalityDedup
# new_df = pd.read_excel('new_booth_to_local_map.xlsx')
# old_df = pd.read_excel('old_booth_to_local_map.xlsx')
# new_localities = new_df.groupby('name2')['booth_num'].mean().reset_index()
# old_localities = old_df.groupby('name2')['booth_number'].mean().reset_index()
# new_locality_dedup = LocalityDedup(new_localities,locality_col='name2')
# new_locality_dedup()
# to_remove_new_index = [12,15]
# new_reviewed = new_locality_dedup.short_list_df.query('index not in @to_remove_new_index')
# new_reviewed
# old_locality_dedup = LocalityDedup(old_localities,locality_col='name2',booth_num_col='booth_number')
# old_locality_dedup()
# to_remove_old_index = [3,4,6,7,9,10]
# old_reviewed = old_locality_dedup.short_list_df.query('index not in @to_remove_old_index')
# old_reviewed
# new_booth_locality = new_df[['booth_num','new_names','name2']]\
#                         .join(new_reviewed[['duplicate_name','master_name']].set_index('duplicate_name'),on='name2')
# new_booth_locality.loc[new_booth_locality['master_name'].isnull(),'master_name'] = new_booth_locality\
#                                                                 .loc[new_booth_locality['master_name'].isnull(),'name2']
# old_booth_locality = old_df[['booth_number','booth_name','name2']]\
#                         .join(old_reviewed[['duplicate_name','master_name']].set_index('duplicate_name'),on='name2')
#
# old_booth_locality.loc[old_booth_locality['master_name'].isnull(),'master_name'] = old_booth_locality\
#                                                                 .loc[old_booth_locality['master_name'].isnull(),'name2']
# old_booth_locality
#
# old_booth_locality['booth_rank'] = old_booth_locality['booth_number']/old_booth_locality['booth_number'].max()
# new_booth_locality['booth_rank'] = new_booth_locality['booth_num']/new_booth_locality['booth_num'].max()
# old_localities = old_booth_locality.groupby('master_name')['booth_rank'].mean().reset_index()
# new_localities = new_booth_locality.groupby('master_name')['booth_rank'].mean().reset_index()
#
# locality_matcher = LocalityMatch(new_localities,old_localities,locality_col='master_name',booth_num_col='booth_rank')
# locality_matcher()
# locality_matcher.connected_components