import pandas as pd
import recordlinkage
import recordlinkage.preprocessing
import numpy as np

class LocalityDedup:
    def __init__(self,locality_df,locality_col='locality',booth_num_col='booth_num',threshold=0.8):
        self.locality_df = locality_df
        self.locality_col = locality_col
        self.booth_num_col = booth_num_col
        self.threshold = threshold

        #Create in-class
        self.candidate_links = None
        self.compare_booths = None
        self.features = None
        self.master_names = None
        self.short_list_df = None


    def get_metaphones(self):
        self.locality_df['metaphone'] = recordlinkage.preprocessing.phonetic(self.locality_df[self.locality_col],method="metaphone")

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
        self.get_metaphones()
        self.get_pairs()
        self.compare_builder()
        self.compute()
        self.consolidate_scores()
        self.get_shortlist()
