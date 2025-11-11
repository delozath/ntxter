import pytest


import pandas as pd


from ntxter.core.base.errors import RegistryError
from ntxter.adapters.mlops.metrics import SklearnMetricsReporting

def gen_data() -> pd.DataFrame:
    df = pd.DataFrame(
        {'identifier': {0: 'test_001', 1: 'test_002', 2: 'test_003'}, 
         'iteration': {0: 1, 1: 2, 2: 2}, 
         'accuracy': {0: 0.9, 1: 0.95, 2: 1.0}, 
         'f1_score': {0: 0.85, 1: 0.8, 2: 0.95}, 
         'precision': {0: 0.8, 1: 0.9, 2: 0.95}, 
         'auc': {0: 0.88, 1: 0.92, 2: 0.98}, 
         'repetition': {0: 3, 1: 3, 2: 5}
        }
     )
    
    return df


@pytest.fixture
def report_metrics() -> SklearnMetricsReporting:
    return SklearnMetricsReporting()

def test_SklearnMetricsReporting_to_reporting(report_metrics: SklearnMetricsReporting):
    test_cls = report_metrics.to_reporting(
        'test_001',
        iteration=1,
        performances={'accuracy': 0.9, 'f1_score': 0.85, 'precision': 0.8, 'auc': 0.88},
        optional={'repetition': 3}
    )
    
    assert test_cls.identifier == 'test_001'
    assert test_cls.iteration == 1
    assert test_cls.performances == {'accuracy': 0.9, 'f1_score': 0.85, 'precision': 0.8, 'auc': 0.88}
    assert test_cls.optional == {'repetition': 3}

    test_cls = report_metrics.to_reporting(
        'test_002',
        iteration=2,
        performances={'accuracy': 0.95, 'f1_score': 0.8, 'precision': 0.9, 'auc': 0.92},
        optional={'repetition': 3}
    )
    
    assert test_cls.identifier == 'test_002'
    assert test_cls.iteration == 2
    assert test_cls.performances == {'accuracy': 0.95, 'f1_score': 0.8, 'precision': 0.9, 'auc': 0.92}
    assert test_cls.optional == {'repetition': 3}


def add_reports(report_metrics: SklearnMetricsReporting) -> SklearnMetricsReporting:
    test_cls = report_metrics.to_reporting(
        'test_001',
        iteration=1,
        performances={'accuracy': 0.9, 'f1_score': 0.85, 'precision': 0.8, 'auc': 0.88},
        optional={'repetition': 3}
    )
    report_metrics.add(test_cls)

    test_cls = report_metrics.to_reporting(
        'test_002',
        iteration=2,
        performances={'accuracy': 0.95, 'f1_score': 0.8, 'precision': 0.9, 'auc': 0.92},
        optional={'repetition': 3}
    )
    report_metrics.add(test_cls)

    test_cls = report_metrics.to_reporting(
        'test_003',
        iteration=2,
        performances={'accuracy': 1, 'f1_score': 0.95, 'precision': 0.95, 'auc': 0.98},
        optional={'repetition': 5}
    )
    report_metrics.add(test_cls)
    return report_metrics


def test_SklearnMetricsReporting_build(report_metrics: SklearnMetricsReporting):
    report_metrics = add_reports(report_metrics)

    df = gen_data()
    df_built = report_metrics.build()
    
    assert df_built.equals(df)


def add_reports_diff_options(report_metrics: SklearnMetricsReporting) -> SklearnMetricsReporting:
    test_cls = report_metrics.to_reporting(
        'test_005',
        iteration=1,
        performances={'accuracy': 0.9, 'f1_score': 0.85, 'precision': 0.8, 'auc': 0.88},
        optional={'repetition': 3, 'note': 'first test'}
    )
    report_metrics.add(test_cls)

    test_cls = report_metrics.to_reporting(
        'test_006',
        iteration=2,
        performances={'accuracy': 0.95, 'f1_score': 0.8, 'precision': 0.9, 'auc': 0.92},
        optional={'repetition': 10, 'extra_info': -10}
    )
    report_metrics.add(test_cls)

    return report_metrics


def test_SklearnMetricsReporting_build_diff_options(report_metrics: SklearnMetricsReporting):
    report_metrics = add_reports(report_metrics)
    report_metrics = add_reports_diff_options(report_metrics)

    df_built = report_metrics.build()
    
    assert 'note' in df_built.columns
    assert 'extra_info' in df_built.columns
    assert df_built[df_built.identifier=='test_006'].extra_info.values[0] == -10
    assert df_built[df_built.identifier=='test_005'].note.values[0] == 'first test'

def test_SklearnMetricsReporting_add_raises(report_metrics: SklearnMetricsReporting):
    report_metrics = add_reports(report_metrics)

    test_cls = report_metrics.to_reporting(
        'test_003',
        iteration=0,
        performances={'accuracy': 1, 'f1_score': 1, 'precision': 1, 'auc': 1},
        optional={'repetition': 20}
    )
    
    with pytest.raises(RegistryError):
        assert report_metrics.add(test_cls)