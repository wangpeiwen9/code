# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/25
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import numpy as np
import pandas as pd
from scipy import stats

from onekey_algo.custom.components.comp1 import select_feature_ttest


def test_ttest_sel():
    rng = np.random.default_rng()
    rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2), random_state=rng)
    data = pd.DataFrame(rvs, columns=['C1', 'C2'])
    sf = select_feature_ttest(data, popmean=5.0, threshold=.05)
    print(sf)


if __name__ == '__main__':
    test_ttest_sel()
