import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest
from dask.distributed import Client, LocalCluster

from stumpy import stimp, stimped

T = [
    np.array([584, -11, 23, 79, 1001, 0, -19], dtype=np.float64),
    np.random.uniform(-1000, 1000, [64]).astype(np.float64),
]


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        dashboard_address=None,
        worker_dashboard_address=None,
    )
    yield cluster
    cluster.close()


@pytest.mark.parametrize("T", T)
def test_stimp_1_percent(T):
    threshold = 0.2
    percentage = 0.01
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(T, m, T, s=s, exclusion_zone=zone)
        ref_P, ref_I, _, _ = naive.scrump(T, m, T, percentage, zone, True, s)
        naive.merge_topk_PI(ref_P, tmp_P, ref_I, tmp_I)
        ref_PAN[pan._bfs_indices[idx], : ref_P.shape[0]] = ref_P

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.parametrize("T", T)
def test_stimp_max_m(T):
    threshold = 0.2
    percentage = 0.01
    min_m = 3
    max_m = 5
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = stimp(
        T,
        min_m=min_m,
        max_m=max_m,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(T, m, T, s=s, exclusion_zone=zone)
        ref_P, ref_I, _, _ = naive.scrump(T, m, T, percentage, zone, True, s)
        naive.merge_topk_PI(ref_P, tmp_P, ref_I, tmp_I)
        ref_PAN[pan._bfs_indices[idx], : ref_P.shape[0]] = ref_P

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.parametrize("T", T)
def test_stimp_100_percent(T):
    threshold = 0.2
    percentage = 1.0
    min_m = 3
    n = T.shape[0] - min_m + 1

    pan = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=False,
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_mp = naive.stump(T, m, T_B=None, exclusion_zone=zone)
        ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.parametrize("T", T)
def test_stimp_raw_mp(T):
    """
    Check pan.P_ attribute for raw matrix profile
    """
    percentage = 1.0
    min_m = 3
    n = 5

    pan = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=False,
        # normalize=True,
    )

    for i in range(n):
        pan.update()

    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        ref_P_ = naive.stump(T, m, T_B=None, exclusion_zone=zone)[:, 0]
        cmp_P_ = pan.P_[idx]

        naive.replace_inf(ref_P_)
        naive.replace_inf(cmp_P_)
        npt.assert_almost_equal(ref_P_, cmp_P_)


@pytest.mark.filterwarnings("ignore:numpy.dtype size changed")
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
@pytest.mark.parametrize("T", T)
def test_stimped(T, dask_cluster):
    with Client(dask_cluster) as dask_client:
        threshold = 0.2
        min_m = 3
        n = T.shape[0] - min_m + 1

        pan = stimped(
            dask_client,
            T,
            min_m=min_m,
            max_m=None,
            step=1,
            # normalize=True,
        )

        for i in range(n):
            pan.update()

        ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

        for idx, m in enumerate(pan.M_[:n]):
            zone = int(np.ceil(m / 4))
            ref_mp = naive.stump(T, m, T_B=None, exclusion_zone=zone)
            ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

        # Compare raw pan
        cmp_PAN = pan._PAN

        naive.replace_inf(ref_PAN)
        naive.replace_inf(cmp_PAN)

        npt.assert_almost_equal(ref_PAN, cmp_PAN)

        # Compare transformed pan
        cmp_pan = pan.PAN_
        ref_pan = naive.transform_pan(
            pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
        )

        naive.replace_inf(ref_pan)
        naive.replace_inf(cmp_pan)

        npt.assert_almost_equal(ref_pan, cmp_pan)


def test_stimp_1_percent_with_isconstant():
    T = np.random.uniform(-1, 1, [64])
    isconstant_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    threshold = 0.2
    percentage = 0.01
    min_m = 3
    n = T.shape[0] - min_m + 1

    seed = np.random.randint(100000)

    np.random.seed(seed)
    pan = stimp(
        T,
        min_m=min_m,
        max_m=None,
        step=1,
        percentage=percentage,
        pre_scrump=True,
        # normalize=True,
        T_subseq_isconstant_func=isconstant_func,
    )

    for i in range(n):
        pan.update()

    ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

    np.random.seed(seed)
    for idx, m in enumerate(pan.M_[:n]):
        zone = int(np.ceil(m / 4))
        s = zone
        tmp_P, tmp_I = naive.prescrump(
            T,
            m,
            T,
            s=s,
            exclusion_zone=zone,
            T_A_subseq_isconstant=isconstant_func,
            T_B_subseq_isconstant=isconstant_func,
        )
        ref_P, ref_I, _, _ = naive.scrump(
            T,
            m,
            T,
            percentage,
            zone,
            True,
            s,
            T_A_subseq_isconstant=isconstant_func,
            T_B_subseq_isconstant=isconstant_func,
        )
        naive.merge_topk_PI(ref_P, tmp_P, ref_I, tmp_I)
        ref_PAN[pan._bfs_indices[idx], : ref_P.shape[0]] = ref_P

    # Compare raw pan
    cmp_PAN = pan._PAN

    naive.replace_inf(ref_PAN)
    naive.replace_inf(cmp_PAN)

    npt.assert_almost_equal(ref_PAN, cmp_PAN)

    # Compare transformed pan
    cmp_pan = pan.PAN_
    ref_pan = naive.transform_pan(
        pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
    )

    naive.replace_inf(ref_pan)
    naive.replace_inf(cmp_pan)

    npt.assert_almost_equal(ref_pan, cmp_pan)


@pytest.mark.filterwarnings("ignore:\\s+Port 8787 is already in use:UserWarning")
def test_stimped_with_isconstant(dask_cluster):
    T = np.random.uniform(-1, 1, [64])
    isconstant_func = functools.partial(
        naive.isconstant_func_stddev_threshold, stddev_threshold=0.5
    )

    with Client(dask_cluster) as dask_client:
        threshold = 0.2
        min_m = 3
        n = T.shape[0] - min_m + 1

        pan = stimped(
            dask_client,
            T,
            min_m=min_m,
            max_m=None,
            step=1,
            # normalize=True,
            T_subseq_isconstant_func=isconstant_func,
        )

        for i in range(n):
            pan.update()

        ref_PAN = np.full((pan.M_.shape[0], T.shape[0]), fill_value=np.inf)

        for idx, m in enumerate(pan.M_[:n]):
            zone = int(np.ceil(m / 4))
            ref_mp = naive.stump(
                T,
                m,
                T_B=None,
                exclusion_zone=zone,
                T_A_subseq_isconstant=isconstant_func,
            )
            ref_PAN[pan._bfs_indices[idx], : ref_mp.shape[0]] = ref_mp[:, 0]

        # Compare raw pan
        cmp_PAN = pan._PAN

        naive.replace_inf(ref_PAN)
        naive.replace_inf(cmp_PAN)

        npt.assert_almost_equal(ref_PAN, cmp_PAN)

        # Compare transformed pan
        cmp_pan = pan.PAN_
        ref_pan = naive.transform_pan(
            pan._PAN, pan._M, threshold, pan._bfs_indices, pan._n_processed
        )

        naive.replace_inf(ref_pan)
        naive.replace_inf(cmp_pan)

        npt.assert_almost_equal(ref_pan, cmp_pan)
