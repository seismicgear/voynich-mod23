import math
from collections import Counter
import experiment.metrics as met
import experiment.stats as st

def test_gzip_size():
    text = "AAAAA"
    # Compression should be consistent
    s1 = met.gzip_size(text)
    s2 = met.gzip_size(text)
    assert s1 == s2
    assert s1 > 0

def test_ngram_counter():
    text = "ABC"
    c = met.ngram_counter(text, 2)
    assert c["AB"] == 1
    assert c["BC"] == 1
    assert len(c) == 2

def test_cosine_similarity_identical():
    c1 = Counter({"AB": 1, "CD": 1})
    c2 = Counter({"AB": 1, "CD": 1})
    sim = met.cosine_similarity(c1, c2)
    assert math.isclose(sim, 1.0)

def test_cosine_similarity_orthogonal():
    c1 = Counter({"AB": 1})
    c2 = Counter({"CD": 1})
    sim = met.cosine_similarity(c1, c2)
    assert math.isclose(sim, 0.0)

def test_shannon_entropy():
    # "AAAA" -> prob(A)=1 -> log(1)=0 -> entropy=0
    assert math.isclose(met.shannon_entropy("AAAA"), 0.0)
    # "AB" -> prob(A)=0.5, prob(B)=0.5 -> -2*(0.5*-1) = 1.0
    assert math.isclose(met.shannon_entropy("AB"), 1.0)

def test_index_of_coincidence():
    # "AAAA" -> N=4. sum(4*3) / (4*3) = 12/12 = 1.0
    assert math.isclose(met.index_of_coincidence("AAAA"), 1.0)
    # "ABCD" -> N=4. sum(1*0...) = 0.
    assert math.isclose(met.index_of_coincidence("ABCD"), 0.0)

def test_stats_z_score():
    obs = 10.0
    nulls = [8.0, 9.0, 11.0, 12.0] # mean=10, std=sqrt((4+1+1+4)/3) = sqrt(3.33) approx 1.82
    # But pstdev uses population std dev (divide by N)
    # mean=10. var=(4+1+1+4)/4 = 2.5. std=sqrt(2.5) approx 1.58
    res = st.calculate_stats(obs, nulls)
    assert res["mean"] == 10.0
    assert math.isclose(res["z_score"], 0.0)
