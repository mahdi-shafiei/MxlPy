# type: ignore

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelbase2.ode import Simulator
from modelbase2.sbml._import import Parser

try:
    import assimulo  # type: ignore

    ASSIMULO_FLAG = True
except ImportError:
    ASSIMULO_FLAG = False

ASSET_PATH = Path("tests") / "sbml" / "assets"


def get_simulation_settings(path, prefix):
    sim_settings = {}
    settings_file = path / f"{prefix}-settings.txt"
    sim_settings = {}
    with open(settings_file, "r") as f:
        for i in f.readlines():
            i = i.strip().split(": ")
            if i[0] == "absolute":
                sim_settings["atol"] = float(i[1])
            elif i[0] == "relative":
                sim_settings["rtol"] = float(i[1])
            elif i[0] == "amount":
                sim_settings["result-ids"] = [j.strip() for j in i[1].split(",")]
    return sim_settings


def add_constant_species_to_results(m, expected, result):
    """If a species is constant we don't include it in the results.
    Since SBML does, we need to manually add it.
    """
    for name in expected.columns.difference(result.columns):
        species = m.get_parameter(name)
        species = pd.Series(
            np.ones(len(expected.index)) * species,
            index=expected.index,
            name=name,
        )
        result = pd.concat([result, species], axis=1)
    return result


def get_files(test):
    prefix = f"{test:05d}"
    path = ASSET_PATH / prefix
    sim_settings = get_simulation_settings(path=path, prefix=prefix)
    expected = pd.read_csv(path / f"{prefix}-results.csv", index_col=0)
    parser = Parser(file=path / f"{prefix}-sbml-l3v2.xml")
    return parser, sim_settings, expected


def add_dummy_compound(m, y0):
    m.add_compound("dummy")
    m.add_reaction(
        "dummy",
        lambda: 0,
        {
            "dummy": 1,
        },
    )
    y0.update({"dummy": 0})


def routine(test):
    parser, sim_settings, expected = get_files(test=test)
    m, y0 = parser.build_model_from_sbml()
    if len(m.stoichiometries) == 0:
        add_dummy_compound(m, y0)

    # Make them a bit harder, such that we guarantee we are getting the required ones
    sim_kwargs = {
        "atol": sim_settings["atol"] / 100,
        "rtol": sim_settings["rtol"] / 100,
    }
    s = Simulator(m)
    s.initialise(y0)
    s.simulate(time_points=expected.index, **sim_kwargs)
    result = s.get_full_results()
    if result is None:
        return False
    result = add_constant_species_to_results(m, expected, result)
    # Sort results like expected
    result = result.loc[:, expected.columns]
    return np.isclose(
        result, expected, rtol=sim_settings["rtol"], atol=sim_settings["atol"]
    ).all()


def test_00001():
    assert routine(test=1)


def test_00002():
    assert routine(test=2)


def test_00003():
    assert routine(test=3)


def test_00004():
    assert routine(test=4)


def test_00005():
    assert routine(test=5)


def test_00006():
    assert routine(test=6)


def test_00007():
    assert routine(test=7)


def test_00008():
    assert routine(test=8)


def test_00009():
    assert routine(test=9)


def test_00010():
    assert routine(test=10)


def test_00011():
    assert routine(test=11)


def test_00012():
    assert routine(test=12)


def test_00013():
    assert routine(test=13)


def test_00014():
    assert routine(test=14)


def test_00015():
    assert routine(test=15)


def test_00016():
    assert routine(test=16)


def test_00017():
    assert routine(test=17)


def test_00018():
    assert routine(test=18)


def test_00019():
    assert routine(test=19)


def test_00020():
    assert routine(test=20)


def test_00021():
    assert routine(test=21)


def test_00022():
    assert routine(test=22)


def test_00023():
    assert routine(test=23)


def test_00024():
    assert routine(test=24)


def test_00025():
    assert routine(test=25)


def test_00026():
    with pytest.raises(NotImplementedError):
        routine(test=26)


def test_00027():
    assert routine(test=27)


def test_00028():
    if ASSIMULO_FLAG:
        assert not routine(test=28)
    else:
        assert routine(test=28)


def test_00029():
    assert routine(test=29)


def test_00030():
    assert routine(test=30)


def test_00031():
    assert routine(test=31)


def test_00032():
    assert routine(test=32)


def test_00033():
    with pytest.raises(NotImplementedError):
        routine(test=33)


def test_00034():
    assert routine(test=34)


def test_00035():
    assert routine(test=35)


def test_00036():
    assert routine(test=36)


def test_00037():
    assert routine(test=37)


def test_00038():
    assert routine(test=38)


def test_00039():
    assert routine(test=39)


def test_00040():
    assert routine(test=40)


def test_00041():
    with pytest.raises(NotImplementedError):
        routine(test=41)


def test_00042():
    assert routine(test=42)


def test_00043():
    assert routine(test=43)


def test_00044():
    assert routine(test=44)


def test_00045():
    assert routine(test=45)


def test_00046():
    assert routine(test=46)


def test_00047():
    assert routine(test=47)


def test_00048():
    assert routine(test=48)


def test_00049():
    assert routine(test=49)


def test_00050():
    assert routine(test=50)


def test_00051():
    with pytest.raises(NotImplementedError):
        routine(test=51)


def test_00052():
    with pytest.raises(NotImplementedError):
        routine(test=52)


def test_00053():
    with pytest.raises(NotImplementedError):
        routine(test=53)


def test_00054():
    assert routine(test=54)


def test_00055():
    assert routine(test=55)


def test_00056():
    assert routine(test=56)


def test_00057():
    assert routine(test=57)


def test_00058():
    assert routine(test=58)


# def test_00059():  # FIXME
#     assert routine(test=59)


def test_00060():
    assert routine(test=60)


def test_00061():
    assert routine(test=61)


def test_00062():
    assert routine(test=62)


def test_00063():
    assert routine(test=63)


def test_00064():
    assert routine(test=64)


def test_00065():
    assert routine(test=65)


def test_00066():
    with pytest.raises(NotImplementedError):
        routine(test=66)


def test_00067():
    with pytest.raises(NotImplementedError):
        routine(test=67)


# def test_00068():  # FIXME
#     assert routine(test=68)

# def test_00069():  # FIXME
#     assert routine(test=69)

# def test_00070():  # FIXME
#     assert routine(test=70)


def test_00071():
    with pytest.raises(NotImplementedError):
        routine(test=71)


def test_00072():
    with pytest.raises(NotImplementedError):
        routine(test=72)


def test_00073():
    with pytest.raises(NotImplementedError):
        routine(test=73)


def test_00074():
    with pytest.raises(NotImplementedError):
        routine(test=74)


def test_00075():
    assert routine(test=75)


def test_00076():
    assert routine(test=76)


def test_00077():
    assert routine(test=77)


def test_00078():
    assert routine(test=78)


def test_00079():
    assert routine(test=79)


def test_00080():
    assert routine(test=80)


def test_00081():
    assert routine(test=81)


def test_00082():
    assert routine(test=82)


def test_00083():
    assert routine(test=83)


def test_00084():
    assert routine(test=84)


def test_00085():
    assert routine(test=85)


def test_00086():
    assert routine(test=86)


def test_00087():
    assert routine(test=87)


def test_00088():
    assert routine(test=88)


def test_00089():
    assert routine(test=89)


def test_00090():
    assert routine(test=90)


def test_00091():
    assert routine(test=91)


def test_00092():
    assert routine(test=92)


def test_00093():
    assert routine(test=93)


def test_00094():
    assert routine(test=94)


def test_00095():
    assert routine(test=95)


def test_00096():
    assert routine(test=96)


def test_00097():
    assert routine(test=97)


def test_00098():
    assert routine(test=98)


def test_00099():
    assert routine(test=99)


def test_00100():
    assert routine(test=100)


def test_00101():
    assert routine(test=101)


def test_00102():
    assert routine(test=102)


def test_00103():
    assert routine(test=103)


def test_00104():
    with pytest.raises(NotImplementedError):
        routine(test=104)


def test_00105():
    with pytest.raises(NotImplementedError):
        routine(test=105)


def test_00106():
    with pytest.raises(NotImplementedError):
        routine(test=106)


def test_00107():
    assert routine(test=107)


def test_00108():
    assert routine(test=108)


def test_00109():
    assert routine(test=109)


def test_00110():
    assert routine(test=110)


def test_00111():
    assert routine(test=111)


def test_00112():
    assert routine(test=112)


def test_00113():
    assert routine(test=113)


def test_00114():
    assert routine(test=114)


def test_00115():
    assert routine(test=115)


def test_00116():
    assert routine(test=116)


def test_00117():
    assert routine(test=117)


def test_00118():
    assert routine(test=118)


def test_00119():
    assert routine(test=119)


def test_00120():
    assert routine(test=120)


def test_00121():
    assert routine(test=121)


def test_00122():
    with pytest.raises(NotImplementedError):
        routine(test=122)


def test_00123():
    with pytest.raises(NotImplementedError):
        routine(test=123)


def test_00124():
    with pytest.raises(NotImplementedError):
        routine(test=124)


def test_00125():
    assert routine(test=125)


def test_00126():
    assert routine(test=126)


def test_00127():
    assert routine(test=127)


def test_00128():
    assert routine(test=128)


# def test_00129():  # FIXME
#     assert routine(test=129)

# def test_00130():  # FIXME
#     assert routine(test=130)

# def test_00131():  # FIXME
#     assert routine(test=131)


def test_00132():
    assert routine(test=132)


def test_00133():
    assert routine(test=133)


# def test_00134():  # FIXME
#     assert routine(test=134)


def test_00135():
    assert routine(test=135)


def test_00136():
    assert routine(test=136)


def test_00137():
    assert routine(test=137)


def test_00138():
    assert routine(test=138)


def test_00139():
    assert routine(test=139)


def test_00140():
    assert routine(test=140)


def test_00141():
    assert routine(test=141)


def test_00142():
    assert routine(test=142)


def test_00143():
    assert routine(test=143)


def test_00144():
    assert routine(test=144)


def test_00145():
    assert routine(test=145)


def test_00146():
    assert routine(test=146)


def test_00147():
    assert routine(test=147)


def test_00148():
    assert routine(test=148)


def test_00149():
    assert routine(test=149)


def test_00150():
    assert routine(test=150)


def test_00151():
    assert routine(test=151)


def test_00152():
    assert routine(test=152)


def test_00153():
    assert routine(test=153)


def test_00154():
    assert routine(test=154)


def test_00155():
    assert routine(test=155)


def test_00156():
    assert routine(test=156)


def test_00157():
    assert routine(test=157)


def test_00158():
    assert routine(test=158)


def test_00159():
    assert routine(test=159)


def test_00160():
    assert routine(test=160)


def test_00161():
    with pytest.raises(NotImplementedError):
        routine(test=161)


def test_00162():
    with pytest.raises(NotImplementedError):
        routine(test=162)


def test_00163():
    with pytest.raises(NotImplementedError):
        routine(test=163)


def test_00164():
    with pytest.raises(NotImplementedError):
        routine(test=164)


def test_00165():
    with pytest.raises(NotImplementedError):
        routine(test=165)


def test_00166():
    with pytest.raises(NotImplementedError):
        routine(test=166)


def test_00167():
    with pytest.raises(NotImplementedError):
        routine(test=167)


def test_00168():
    with pytest.raises(NotImplementedError):
        routine(test=168)


def test_00169():
    with pytest.raises(NotImplementedError):
        routine(test=169)


def test_00170():
    with pytest.raises(NotImplementedError):
        routine(test=170)


def test_00171():
    with pytest.raises(NotImplementedError):
        routine(test=171)


def test_00172():
    with pytest.raises(NotImplementedError):
        routine(test=172)


def test_00173():
    with pytest.raises(NotImplementedError):
        routine(test=173)


def test_00174():
    assert routine(test=174)


def test_00175():
    with pytest.raises(NotImplementedError):
        routine(test=175)


def test_00176():
    with pytest.raises(NotImplementedError):
        routine(test=176)


def test_00177():
    with pytest.raises(NotImplementedError):
        routine(test=177)


def test_00178():
    with pytest.raises(NotImplementedError):
        routine(test=178)


def test_00179():
    with pytest.raises(NotImplementedError):
        routine(test=179)


def test_00180():
    with pytest.raises(NotImplementedError):
        routine(test=180)


def test_00181():
    with pytest.raises(NotImplementedError):
        routine(test=181)


def test_00182():
    with pytest.raises(NotImplementedError):
        routine(test=182)


def test_00183():
    with pytest.raises(NotImplementedError):
        routine(test=183)


def test_00184():
    with pytest.raises(NotImplementedError):
        routine(test=184)


def test_00185():
    with pytest.raises(NotImplementedError):
        routine(test=185)


def test_00186():
    assert routine(test=186)


def test_00187():
    assert routine(test=187)


def test_00188():
    assert routine(test=188)


def test_00189():
    assert routine(test=189)


def test_00190():
    assert routine(test=190)


def test_00191():
    assert routine(test=191)


def test_00192():
    assert routine(test=192)


def test_00193():
    assert routine(test=193)


def test_00194():
    assert routine(test=194)


def test_00195():
    assert routine(test=195)


def test_00196():
    if ASSIMULO_FLAG:
        assert not routine(test=196)
    else:
        assert routine(test=196)


def test_00197():
    if ASSIMULO_FLAG:
        assert not routine(test=197)
    else:
        assert routine(test=197)


def test_00198():
    assert routine(test=198)


def test_00199():
    assert routine(test=199)


def test_00200():
    assert routine(test=200)


def test_00201():
    assert routine(test=201)


def test_00202():
    assert routine(test=202)


def test_00203():
    assert routine(test=203)


def test_00204():
    assert routine(test=204)


def test_00205():
    assert routine(test=205)


def test_00206():
    assert routine(test=206)


def test_00207():
    assert routine(test=207)


def test_00208():
    assert routine(test=208)


def test_00209():
    assert routine(test=209)


def test_00210():
    assert routine(test=210)


def test_00211():
    assert routine(test=211)


def test_00212():
    assert routine(test=212)


def test_00213():
    assert routine(test=213)


def test_00214():
    assert routine(test=214)


def test_00215():
    assert routine(test=215)


def test_00216():
    assert routine(test=216)


def test_00217():
    assert routine(test=217)


def test_00218():
    assert routine(test=218)


def test_00219():
    assert routine(test=219)


def test_00220():
    assert routine(test=220)


def test_00221():
    assert routine(test=221)


def test_00222():
    assert routine(test=222)


def test_00223():
    assert routine(test=223)


def test_00224():
    assert routine(test=224)


def test_00225():
    assert routine(test=225)


def test_00226():
    assert routine(test=226)


def test_00227():
    assert routine(test=227)


def test_00228():
    assert routine(test=228)


def test_00229():
    assert routine(test=229)


def test_00230():
    assert routine(test=230)


def test_00231():
    assert routine(test=231)


def test_00232():
    assert routine(test=232)


def test_00233():
    assert routine(test=233)


def test_00234():
    assert routine(test=234)


def test_00235():
    assert routine(test=235)


def test_00236():
    assert routine(test=236)


def test_00237():
    assert routine(test=237)


def test_00238():
    assert routine(test=238)


def test_00239():
    assert routine(test=239)


def test_00240():
    assert routine(test=240)


def test_00241():
    assert routine(test=241)


def test_00242():
    assert routine(test=242)


def test_00243():
    assert routine(test=243)


def test_00244():
    assert routine(test=244)


def test_00245():
    assert routine(test=245)


def test_00246():
    assert routine(test=246)


def test_00247():
    assert routine(test=247)


def test_00248():
    assert routine(test=248)


def test_00249():
    assert routine(test=249)


def test_00250():
    assert routine(test=250)


def test_00251():
    assert routine(test=251)


def test_00252():
    assert routine(test=252)


def test_00253():
    assert routine(test=253)


def test_00254():
    assert routine(test=254)


def test_00255():
    assert routine(test=255)


def test_00256():
    assert routine(test=256)


def test_00257():
    assert routine(test=257)


def test_00258():
    assert routine(test=258)


def test_00259():
    assert routine(test=259)


def test_00260():
    assert routine(test=260)


def test_00261():
    assert routine(test=261)


def test_00262():
    assert routine(test=262)


def test_00263():
    assert routine(test=263)


def test_00264():
    assert routine(test=264)


def test_00265():
    assert routine(test=265)


def test_00266():
    assert routine(test=266)


def test_00267():
    assert routine(test=267)


def test_00268():
    assert routine(test=268)


def test_00269():
    if ASSIMULO_FLAG:
        assert not routine(test=269)
    else:
        assert routine(test=269)


def test_00270():
    assert routine(test=270)


def test_00271():
    with pytest.raises(TypeError):
        routine(test=271)


def test_00272():
    assert routine(test=272)


def test_00273():
    assert routine(test=273)


def test_00274():
    assert routine(test=274)


def test_00275():
    with pytest.raises(TypeError):
        routine(test=275)


def test_00276():
    assert routine(test=276)


def test_00277():
    assert routine(test=277)


def test_00278():
    assert routine(test=278)


def test_00279():
    assert routine(test=279)


def test_00280():
    assert routine(test=280)


def test_00281():
    assert routine(test=281)


def test_00282():
    assert routine(test=282)


def test_00283():
    assert routine(test=283)


def test_00284():
    assert routine(test=284)


def test_00285():
    assert routine(test=285)


def test_00286():
    assert routine(test=286)


def test_00287():
    assert routine(test=287)


def test_00288():
    assert routine(test=288)


def test_00289():
    assert routine(test=289)


def test_00290():
    assert routine(test=290)


def test_00291():
    assert routine(test=291)


def test_00292():
    assert routine(test=292)


def test_00293():
    assert routine(test=293)


def test_00294():
    assert routine(test=294)


def test_00295():
    assert routine(test=295)


def test_00296():
    assert routine(test=296)


def test_00297():
    assert routine(test=297)


def test_00298():
    assert routine(test=298)


def test_00299():
    assert routine(test=299)


def test_00300():
    assert routine(test=300)


def test_00301():
    assert routine(test=301)


def test_00302():
    assert routine(test=302)


def test_00303():
    assert routine(test=303)


def test_00304():
    assert routine(test=304)


def test_00305():
    assert routine(test=305)


def test_00306():
    assert routine(test=306)


def test_00307():
    assert routine(test=307)


def test_00308():
    assert routine(test=308)


def test_00309():
    assert routine(test=309)


def test_00310():
    with pytest.raises(NotImplementedError):
        routine(test=310)


def test_00311():
    with pytest.raises(NotImplementedError):
        routine(test=311)


def test_00312():
    with pytest.raises(NotImplementedError):
        routine(test=312)


def test_00313():
    with pytest.raises(NotImplementedError):
        routine(test=313)


def test_00314():
    with pytest.raises(NotImplementedError):
        routine(test=314)


def test_00315():
    with pytest.raises(NotImplementedError):
        routine(test=315)


def test_00316():
    with pytest.raises(NotImplementedError):
        routine(test=316)


def test_00317():
    with pytest.raises(NotImplementedError):
        routine(test=317)


def test_00318():
    with pytest.raises(NotImplementedError):
        routine(test=318)


def test_00319():
    assert routine(test=319)


def test_00320():
    assert routine(test=320)


def test_00321():
    assert routine(test=321)


def test_00322():
    assert routine(test=322)


def test_00323():
    assert routine(test=323)


def test_00324():
    assert routine(test=324)


def test_00325():
    assert routine(test=325)


def test_00326():
    assert routine(test=326)


def test_00327():
    assert routine(test=327)


def test_00328():
    assert routine(test=328)


def test_00329():
    assert routine(test=329)


def test_00330():
    assert routine(test=330)


def test_00331():
    assert routine(test=331)


def test_00332():
    assert routine(test=332)


def test_00333():
    assert routine(test=333)


def test_00334():
    assert routine(test=334)


def test_00335():
    assert routine(test=335)


def test_00336():
    assert routine(test=336)


def test_00337():
    assert routine(test=337)


def test_00338():
    assert routine(test=338)


def test_00339():
    assert routine(test=339)


def test_00340():
    assert routine(test=340)


def test_00341():
    assert routine(test=341)


def test_00342():
    assert routine(test=342)


def test_00343():
    assert routine(test=343)


def test_00344():
    assert routine(test=344)


def test_00345():
    assert routine(test=345)


def test_00346():
    assert routine(test=346)


def test_00347():
    assert routine(test=347)


def test_00348():
    with pytest.raises(NotImplementedError):
        routine(test=348)


def test_00349():
    with pytest.raises(NotImplementedError):
        routine(test=349)


def test_00350():
    with pytest.raises(NotImplementedError):
        routine(test=350)


def test_00351():
    with pytest.raises(NotImplementedError):
        routine(test=351)


def test_00352():
    with pytest.raises(NotImplementedError):
        routine(test=352)


def test_00353():
    with pytest.raises(NotImplementedError):
        routine(test=353)


def test_00354():
    with pytest.raises(NotImplementedError):
        routine(test=354)


def test_00355():
    with pytest.raises(NotImplementedError):
        routine(test=355)


def test_00356():
    with pytest.raises(NotImplementedError):
        routine(test=356)


def test_00357():
    with pytest.raises(NotImplementedError):
        routine(test=357)


def test_00358():
    with pytest.raises(NotImplementedError):
        routine(test=358)


def test_00359():
    with pytest.raises(NotImplementedError):
        routine(test=359)


def test_00360():
    with pytest.raises(NotImplementedError):
        routine(test=360)


def test_00361():
    with pytest.raises(NotImplementedError):
        routine(test=361)


def test_00362():
    with pytest.raises(NotImplementedError):
        routine(test=362)


def test_00363():
    with pytest.raises(NotImplementedError):
        routine(test=363)


def test_00364():
    with pytest.raises(NotImplementedError):
        routine(test=364)


def test_00365():
    with pytest.raises(NotImplementedError):
        routine(test=365)


def test_00366():
    with pytest.raises(NotImplementedError):
        routine(test=366)


def test_00367():
    with pytest.raises(NotImplementedError):
        routine(test=367)


def test_00368():
    with pytest.raises(NotImplementedError):
        routine(test=368)


def test_00369():
    with pytest.raises(NotImplementedError):
        routine(test=369)


def test_00370():
    with pytest.raises(NotImplementedError):
        routine(test=370)


def test_00371():
    with pytest.raises(NotImplementedError):
        routine(test=371)


def test_00372():
    with pytest.raises(NotImplementedError):
        routine(test=372)


def test_00373():
    with pytest.raises(NotImplementedError):
        routine(test=373)


def test_00374():
    with pytest.raises(NotImplementedError):
        routine(test=374)


def test_00375():
    with pytest.raises(NotImplementedError):
        routine(test=375)


def test_00376():
    with pytest.raises(NotImplementedError):
        routine(test=376)


def test_00377():
    with pytest.raises(NotImplementedError):
        routine(test=377)


def test_00378():
    with pytest.raises(NotImplementedError):
        routine(test=378)


def test_00379():
    with pytest.raises(NotImplementedError):
        routine(test=379)


def test_00380():
    with pytest.raises(NotImplementedError):
        routine(test=380)


def test_00381():
    with pytest.raises(NotImplementedError):
        routine(test=381)


def test_00382():
    with pytest.raises(NotImplementedError):
        routine(test=382)


def test_00383():
    with pytest.raises(NotImplementedError):
        routine(test=383)


def test_00384():
    with pytest.raises(NotImplementedError):
        routine(test=384)


def test_00385():
    with pytest.raises(NotImplementedError):
        routine(test=385)


def test_00386():
    with pytest.raises(NotImplementedError):
        routine(test=386)


def test_00387():
    with pytest.raises(NotImplementedError):
        routine(test=387)


# def test_00388():  # FIXME
#     assert routine(test=388)


def test_00389():
    with pytest.raises(NotImplementedError):
        routine(test=389)


def test_00390():
    with pytest.raises(NotImplementedError):
        routine(test=390)


# def test_00391():  # FIXME
#     assert routine(test=391)


def test_00392():
    with pytest.raises(NotImplementedError):
        routine(test=392)


def test_00393():
    with pytest.raises(NotImplementedError):
        routine(test=393)


# def test_00394():  # FIXME
#     assert routine(test=394)


def test_00395():
    with pytest.raises(NotImplementedError):
        routine(test=395)


def test_00396():
    with pytest.raises(NotImplementedError):
        routine(test=396)


def test_00397():
    with pytest.raises(NotImplementedError):
        routine(test=397)


def test_00398():
    with pytest.raises(NotImplementedError):
        routine(test=398)


def test_00399():
    with pytest.raises(NotImplementedError):
        routine(test=399)


def test_00400():
    with pytest.raises(NotImplementedError):
        routine(test=400)


def test_00401():
    with pytest.raises(NotImplementedError):
        routine(test=401)


def test_00402():
    with pytest.raises(NotImplementedError):
        routine(test=402)


def test_00403():
    with pytest.raises(NotImplementedError):
        routine(test=403)


def test_00404():
    with pytest.raises(NotImplementedError):
        routine(test=404)


def test_00405():
    with pytest.raises(NotImplementedError):
        routine(test=405)


def test_00406():
    with pytest.raises(NotImplementedError):
        routine(test=406)


def test_00407():
    with pytest.raises(NotImplementedError):
        routine(test=407)


def test_00408():
    with pytest.raises(NotImplementedError):
        routine(test=408)


def test_00409():
    with pytest.raises(NotImplementedError):
        routine(test=409)


def test_00410():
    with pytest.raises(NotImplementedError):
        routine(test=410)


def test_00411():
    with pytest.raises(NotImplementedError):
        routine(test=411)


def test_00412():
    with pytest.raises(NotImplementedError):
        routine(test=412)


def test_00413():
    with pytest.raises(NotImplementedError):
        routine(test=413)


def test_00414():
    with pytest.raises(NotImplementedError):
        routine(test=414)


def test_00415():
    with pytest.raises(NotImplementedError):
        routine(test=415)


def test_00416():
    with pytest.raises(NotImplementedError):
        routine(test=416)


def test_00417():
    with pytest.raises(NotImplementedError):
        routine(test=417)


def test_00418():
    with pytest.raises(NotImplementedError):
        routine(test=418)


def test_00419():
    with pytest.raises(NotImplementedError):
        routine(test=419)


def test_00420():
    with pytest.raises(NotImplementedError):
        routine(test=420)


def test_00421():
    with pytest.raises(NotImplementedError):
        routine(test=421)


def test_00422():
    with pytest.raises(NotImplementedError):
        routine(test=422)


def test_00423():
    with pytest.raises(NotImplementedError):
        routine(test=423)


def test_00424():
    with pytest.raises(NotImplementedError):
        routine(test=424)


def test_00425():
    with pytest.raises(NotImplementedError):
        routine(test=425)


def test_00426():
    with pytest.raises(NotImplementedError):
        routine(test=426)


def test_00427():
    with pytest.raises(NotImplementedError):
        routine(test=427)


def test_00428():
    with pytest.raises(NotImplementedError):
        routine(test=428)


def test_00429():
    with pytest.raises(NotImplementedError):
        routine(test=429)


def test_00430():
    with pytest.raises(NotImplementedError):
        routine(test=430)


def test_00431():
    with pytest.raises(NotImplementedError):
        routine(test=431)


def test_00432():
    with pytest.raises(NotImplementedError):
        routine(test=432)


def test_00433():
    with pytest.raises(NotImplementedError):
        routine(test=433)


def test_00434():
    with pytest.raises(NotImplementedError):
        routine(test=434)


def test_00435():
    with pytest.raises(NotImplementedError):
        routine(test=435)


def test_00436():
    with pytest.raises(NotImplementedError):
        routine(test=436)


def test_00437():
    with pytest.raises(NotImplementedError):
        routine(test=437)


def test_00438():
    with pytest.raises(NotImplementedError):
        routine(test=438)


def test_00439():
    with pytest.raises(NotImplementedError):
        routine(test=439)


def test_00440():
    with pytest.raises(NotImplementedError):
        routine(test=440)


def test_00441():
    with pytest.raises(NotImplementedError):
        routine(test=441)


def test_00442():
    with pytest.raises(NotImplementedError):
        routine(test=442)


def test_00443():
    with pytest.raises(NotImplementedError):
        routine(test=443)


def test_00444():
    with pytest.raises(NotImplementedError):
        routine(test=444)


# def test_00445():  # FIXME
#     assert routine(test=445)


def test_00446():
    with pytest.raises(NotImplementedError):
        routine(test=446)


def test_00447():
    with pytest.raises(NotImplementedError):
        routine(test=447)


# def test_00448():  # FIXME
#     assert routine(test=448)


def test_00449():
    with pytest.raises(NotImplementedError):
        routine(test=449)


def test_00450():
    with pytest.raises(NotImplementedError):
        routine(test=450)


# def test_00451():  # FIXME
#     assert routine(test=451)


def test_00452():
    with pytest.raises(NotImplementedError):
        routine(test=452)


def test_00453():
    with pytest.raises(NotImplementedError):
        routine(test=453)


def test_00454():
    with pytest.raises(NotImplementedError):
        routine(test=454)


def test_00455():
    with pytest.raises(NotImplementedError):
        routine(test=455)


def test_00456():
    with pytest.raises(NotImplementedError):
        routine(test=456)


def test_00457():
    with pytest.raises(NotImplementedError):
        routine(test=457)


def test_00458():
    with pytest.raises(NotImplementedError):
        routine(test=458)


def test_00459():
    with pytest.raises(NotImplementedError):
        routine(test=459)


def test_00460():
    with pytest.raises(NotImplementedError):
        routine(test=460)


def test_00461():
    with pytest.raises(NotImplementedError):
        routine(test=461)


def test_00462():
    assert routine(test=462)


def test_00463():
    assert routine(test=463)


def test_00464():
    assert routine(test=464)


def test_00465():
    assert routine(test=465)


def test_00466():
    assert routine(test=466)


def test_00467():
    assert routine(test=467)


def test_00468():
    assert routine(test=468)


def test_00469():
    assert routine(test=469)


def test_00470():
    assert routine(test=470)


def test_00471():
    assert routine(test=471)


def test_00472():
    assert routine(test=472)


def test_00473():
    assert routine(test=473)


def test_00474():
    assert routine(test=474)


def test_00475():
    assert routine(test=475)


def test_00476():
    assert routine(test=476)


def test_00477():
    assert routine(test=477)


def test_00478():
    assert routine(test=478)


def test_00479():
    assert routine(test=479)


def test_00480():
    assert routine(test=480)


def test_00481():
    assert routine(test=481)


def test_00482():
    assert routine(test=482)


def test_00483():
    assert routine(test=483)


def test_00484():
    assert routine(test=484)


def test_00485():
    assert routine(test=485)


def test_00486():
    assert routine(test=486)


def test_00487():
    assert routine(test=487)


def test_00488():
    assert routine(test=488)


def test_00489():
    assert routine(test=489)


def test_00490():
    assert routine(test=490)


def test_00491():
    assert routine(test=491)


def test_00492():
    assert routine(test=492)


def test_00493():
    assert routine(test=493)


def test_00494():
    assert routine(test=494)


def test_00495():
    assert routine(test=495)


def test_00496():
    assert routine(test=496)


def test_00497():
    assert routine(test=497)


def test_00498():
    assert routine(test=498)


def test_00499():
    assert routine(test=499)


def test_00500():
    assert routine(test=500)


def test_00501():
    assert routine(test=501)


def test_00502():
    assert routine(test=502)


def test_00503():
    assert routine(test=503)


# def test_00504():  # FIXME
#     assert routine(test=504)


# def test_00505():  # FIXME
#     assert routine(test=505)


# def test_00506():  # FIXME
#     assert routine(test=506)


# def test_00507():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=507)


# def test_00508():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=508)


# def test_00509():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=509)


def test_00510():
    assert routine(test=510)


def test_00511():
    assert routine(test=511)


def test_00512():
    assert routine(test=512)


def test_00513():
    assert routine(test=513)


def test_00514():
    assert routine(test=514)


def test_00515():
    assert routine(test=515)


# def test_00516():  # FIXME
#     assert routine(test=516)

# def test_00517():  # FIXME
#     assert routine(test=517)

# def test_00518():  # FIXME
#     assert routine(test=518)

# def test_00519():  # FIXME
#     assert routine(test=519)

# def test_00520():  # FIXME
#     assert routine(test=520)

# def test_00521():  # FIXME
#     assert routine(test=521)


def test_00522():
    assert routine(test=522)


def test_00523():
    assert routine(test=523)


def test_00524():
    assert routine(test=524)


def test_00525():
    assert routine(test=525)


def test_00526():
    assert routine(test=526)


def test_00527():
    assert routine(test=527)


def test_00528():
    assert routine(test=528)


def test_00529():
    assert routine(test=529)


def test_00530():
    assert routine(test=530)


def test_00531():
    assert routine(test=531)


def test_00532():
    assert routine(test=532)


def test_00533():
    assert routine(test=533)


def test_00534():
    assert routine(test=534)


def test_00535():
    assert routine(test=535)


def test_00536():
    assert routine(test=536)


def test_00537():
    assert routine(test=537)


def test_00538():
    assert routine(test=538)


def test_00539():
    assert routine(test=539)


def test_00540():
    assert routine(test=540)


def test_00541():
    assert routine(test=541)


def test_00542():
    assert routine(test=542)


def test_00543():
    assert routine(test=543)


def test_00544():
    assert routine(test=544)


def test_00545():
    assert routine(test=545)


def test_00546():
    assert routine(test=546)


def test_00547():
    assert routine(test=547)


def test_00548():
    assert routine(test=548)


def test_00549():
    assert routine(test=549)


def test_00550():
    assert routine(test=550)


def test_00551():
    assert routine(test=551)


def test_00552():
    assert routine(test=552)


def test_00553():
    assert routine(test=553)


def test_00554():
    assert routine(test=554)


def test_00555():
    assert routine(test=555)


def test_00556():
    assert routine(test=556)


def test_00557():
    assert routine(test=557)


def test_00558():
    assert routine(test=558)


def test_00559():
    assert routine(test=559)


def test_00560():
    assert routine(test=560)


# def test_00561():  # FIXME
#     assert routine(test=561)

# def test_00562():  # FIXME
#     assert routine(test=562)

# def test_00563():  # FIXME
#     assert routine(test=563)

# def test_00564():  # FIXME
#     assert routine(test=564)


def test_00565():
    assert routine(test=565)


def test_00566():
    assert routine(test=566)


def test_00567():
    assert routine(test=567)


def test_00568():
    assert routine(test=568)


def test_00569():
    assert routine(test=569)


def test_00570():
    assert routine(test=570)


def test_00571():
    with pytest.raises(NameError):
        routine(test=571)


def test_00572():
    with pytest.raises(TypeError):
        routine(test=572)


def test_00573():
    with pytest.raises(NotImplementedError):
        routine(test=573)


def test_00574():
    with pytest.raises(NotImplementedError):
        routine(test=574)


def test_00575():
    with pytest.raises(NotImplementedError):
        routine(test=575)


def test_00576():
    with pytest.raises(NotImplementedError):
        routine(test=576)


def test_00577():
    assert routine(test=577)


def test_00578():
    assert routine(test=578)


def test_00579():
    assert routine(test=579)


def test_00580():
    assert routine(test=580)


def test_00581():
    assert routine(test=581)


def test_00582():
    assert routine(test=582)


def test_00583():
    assert routine(test=583)


def test_00584():
    assert routine(test=584)


def test_00585():
    assert routine(test=585)


def test_00586():
    assert routine(test=586)


def test_00587():
    assert routine(test=587)


def test_00588():
    assert routine(test=588)


def test_00589():
    assert routine(test=589)


def test_00590():
    assert routine(test=590)


def test_00591():
    assert routine(test=591)


def test_00592():
    assert routine(test=592)


def test_00593():
    assert routine(test=593)


def test_00594():
    assert routine(test=594)


def test_00595():
    assert routine(test=595)


def test_00596():
    assert routine(test=596)


# def test_00597():  # FIXME
#     assert routine(test=597)


def test_00598():
    assert routine(test=598)


def test_00599():
    assert routine(test=599)


def test_00600():
    assert routine(test=600)


def test_00601():
    assert routine(test=601)


def test_00602():
    assert routine(test=602)


def test_00603():
    assert routine(test=603)


def test_00604():
    assert routine(test=604)


def test_00605():
    assert routine(test=605)


def test_00606():
    assert routine(test=606)


def test_00607():
    assert routine(test=607)


def test_00608():
    assert routine(test=608)


# def test_00609():  # FIXME
#     assert routine(test=609)

# def test_00610():  # FIXME
#     assert routine(test=610)


def test_00611():
    assert routine(test=611)


def test_00612():
    assert routine(test=612)


def test_00613():
    assert routine(test=613)


def test_00614():
    assert routine(test=614)


def test_00615():
    assert routine(test=615)


def test_00616():
    assert routine(test=616)


def test_00617():
    assert routine(test=617)


def test_00618():
    assert routine(test=618)


def test_00619():
    with pytest.raises(NotImplementedError):
        routine(test=619)


def test_00620():
    with pytest.raises(NotImplementedError):
        routine(test=620)


def test_00621():
    with pytest.raises(NotImplementedError):
        routine(test=621)


def test_00622():
    with pytest.raises(NotImplementedError):
        routine(test=622)


def test_00623():
    with pytest.raises(NotImplementedError):
        routine(test=623)


def test_00624():
    with pytest.raises(NotImplementedError):
        routine(test=624)


def test_00625():
    assert routine(test=625)


def test_00626():
    assert routine(test=626)


def test_00627():
    assert routine(test=627)


def test_00628():
    assert routine(test=628)


def test_00629():
    assert routine(test=629)


def test_00630():
    assert routine(test=630)


def test_00631():
    assert routine(test=631)


def test_00632():
    assert routine(test=632)


def test_00633():
    assert routine(test=633)


def test_00634():
    with pytest.raises(NotImplementedError):
        routine(test=634)


def test_00635():
    with pytest.raises(NotImplementedError):
        routine(test=635)


def test_00636():
    with pytest.raises(NotImplementedError):
        routine(test=636)


def test_00637():
    with pytest.raises(NotImplementedError):
        routine(test=637)


def test_00638():
    with pytest.raises(NotImplementedError):
        routine(test=638)


def test_00639():
    with pytest.raises(NotImplementedError):
        routine(test=639)


def test_00640():
    assert routine(test=640)


def test_00641():
    assert routine(test=641)


def test_00642():
    assert routine(test=642)


def test_00643():
    assert routine(test=643)


def test_00644():
    assert routine(test=644)


def test_00645():
    assert routine(test=645)


def test_00646():
    with pytest.raises(NotImplementedError):
        routine(test=646)


def test_00647():
    with pytest.raises(NotImplementedError):
        routine(test=647)


def test_00648():
    with pytest.raises(NotImplementedError):
        routine(test=648)


def test_00649():
    with pytest.raises(NotImplementedError):
        routine(test=649)


def test_00650():
    with pytest.raises(NotImplementedError):
        routine(test=650)


def test_00651():
    with pytest.raises(NotImplementedError):
        routine(test=651)


def test_00652():
    with pytest.raises(NotImplementedError):
        routine(test=652)


def test_00653():
    with pytest.raises(NotImplementedError):
        routine(test=653)


def test_00654():
    with pytest.raises(NotImplementedError):
        routine(test=654)


def test_00655():
    with pytest.raises(NotImplementedError):
        routine(test=655)


def test_00656():
    with pytest.raises(NotImplementedError):
        routine(test=656)


def test_00657():
    with pytest.raises(NotImplementedError):
        routine(test=657)


def test_00658():
    assert routine(test=658)


def test_00659():
    assert routine(test=659)


def test_00660():
    assert routine(test=660)


def test_00661():
    with pytest.raises(NotImplementedError):
        routine(test=661)


def test_00662():
    with pytest.raises(NotImplementedError):
        routine(test=662)


def test_00663():
    with pytest.raises(NotImplementedError):
        routine(test=663)


def test_00664():
    with pytest.raises(NotImplementedError):
        routine(test=664)


def test_00665():
    with pytest.raises(NotImplementedError):
        routine(test=665)


def test_00666():
    with pytest.raises(NotImplementedError):
        routine(test=666)


def test_00667():
    assert routine(test=667)


def test_00668():
    assert routine(test=668)


def test_00669():
    assert routine(test=669)


def test_00670():
    assert routine(test=670)


def test_00671():
    assert routine(test=671)


def test_00672():
    assert routine(test=672)


def test_00673():
    assert routine(test=673)


def test_00674():
    assert routine(test=674)


def test_00675():
    assert routine(test=675)


def test_00676():
    assert routine(test=676)


def test_00677():
    assert routine(test=677)


def test_00678():
    assert routine(test=678)


def test_00679():
    with pytest.raises(NotImplementedError):
        routine(test=679)


def test_00680():
    with pytest.raises(NotImplementedError):
        routine(test=680)


def test_00681():
    with pytest.raises(NotImplementedError):
        routine(test=681)


def test_00682():
    with pytest.raises(NotImplementedError):
        routine(test=682)


def test_00683():
    with pytest.raises(NotImplementedError):
        routine(test=683)


def test_00684():
    with pytest.raises(NotImplementedError):
        routine(test=684)


def test_00685():
    assert routine(test=685)


def test_00686():
    assert routine(test=686)


def test_00687():
    assert routine(test=687)


# def test_00688():  # FIXME
#     assert routine(test=688)


def test_00689():
    with pytest.raises(NotImplementedError):
        routine(test=689)


def test_00690():
    with pytest.raises(NotImplementedError):
        routine(test=690)


def test_00691():
    assert routine(test=691)


def test_00692():
    assert routine(test=692)


def test_00693():
    assert routine(test=693)


def test_00694():
    assert routine(test=694)


def test_00695():
    assert routine(test=695)


def test_00696():
    assert routine(test=696)


def test_00697():
    assert routine(test=697)


def test_00698():
    assert routine(test=698)


def test_00699():
    with pytest.raises(NotImplementedError):
        routine(test=699)


def test_00700():
    with pytest.raises(NotImplementedError):
        routine(test=700)


def test_00701():
    with pytest.raises(NotImplementedError):
        routine(test=701)


def test_00702():
    with pytest.raises(NotImplementedError):
        routine(test=702)


def test_00703():
    assert routine(test=703)


def test_00704():
    assert routine(test=704)


def test_00705():
    assert routine(test=705)


def test_00706():
    assert routine(test=706)


def test_00707():
    with pytest.raises(NotImplementedError):
        routine(test=707)


def test_00708():
    with pytest.raises(NotImplementedError):
        routine(test=708)


def test_00709():
    assert routine(test=709)


def test_00710():
    assert routine(test=710)


def test_00711():
    assert routine(test=711)


def test_00712():
    with pytest.raises(NotImplementedError):
        routine(test=712)


def test_00713():
    with pytest.raises(NotImplementedError):
        routine(test=713)


def test_00714():
    with pytest.raises(NotImplementedError):
        routine(test=714)


def test_00715():
    assert routine(test=715)


def test_00716():
    assert routine(test=716)


def test_00717():
    assert routine(test=717)


def test_00718():
    assert routine(test=718)


def test_00719():
    assert routine(test=719)


def test_00720():
    assert routine(test=720)


def test_00721():
    assert routine(test=721)


def test_00722():
    assert routine(test=722)


def test_00723():
    with pytest.raises(NotImplementedError):
        routine(test=723)


def test_00724():
    with pytest.raises(NotImplementedError):
        routine(test=724)


# def test_00725():  # FIXME
#     assert routine(test=725)

# def test_00726():  # FIXME
#     assert routine(test=726)

# def test_00727():  # FIXME
#     assert routine(test=727)

# def test_00728():  # FIXME
#     assert routine(test=728)

# def test_00729():  # FIXME
#     assert routine(test=729)

# def test_00730():  # FIXME
#     assert routine(test=730)

# def test_00731():  # FIXME
#     assert routine(test=731)


def test_00732():
    assert routine(test=732)


def test_00733():
    assert routine(test=733)


def test_00734():
    assert routine(test=734)


def test_00735():
    assert routine(test=735)


def test_00736():
    with pytest.raises(NotImplementedError):
        routine(test=736)


def test_00737():
    with pytest.raises(NotImplementedError):
        routine(test=737)


def test_00738():
    assert routine(test=738)


def test_00739():
    assert routine(test=739)


def test_00740():
    with pytest.raises(NotImplementedError):
        routine(test=740)


def test_00741():
    with pytest.raises(NotImplementedError):
        routine(test=741)


def test_00742():
    with pytest.raises(NotImplementedError):
        routine(test=742)


def test_00743():
    with pytest.raises(NotImplementedError):
        routine(test=743)


def test_00744():
    with pytest.raises(NotImplementedError):
        routine(test=744)


def test_00745():
    with pytest.raises(NotImplementedError):
        routine(test=745)


def test_00746():
    with pytest.raises(NotImplementedError):
        routine(test=746)


def test_00747():
    with pytest.raises(NotImplementedError):
        routine(test=747)


def test_00748():
    with pytest.raises(NotImplementedError):
        routine(test=748)


def test_00749():
    with pytest.raises(NotImplementedError):
        routine(test=749)


def test_00750():
    with pytest.raises(NotImplementedError):
        routine(test=750)


def test_00751():
    with pytest.raises(NotImplementedError):
        routine(test=751)


def test_00752():
    with pytest.raises(NotImplementedError):
        routine(test=752)


def test_00753():
    with pytest.raises(NotImplementedError):
        routine(test=753)


def test_00754():
    with pytest.raises(NotImplementedError):
        routine(test=754)


def test_00755():
    with pytest.raises(NotImplementedError):
        routine(test=755)


def test_00756():
    with pytest.raises(NotImplementedError):
        routine(test=756)


def test_00757():
    with pytest.raises(NotImplementedError):
        routine(test=757)


def test_00758():
    with pytest.raises(NotImplementedError):
        routine(test=758)


def test_00759():
    with pytest.raises(NotImplementedError):
        routine(test=759)


def test_00760():
    with pytest.raises(NotImplementedError):
        routine(test=760)


def test_00761():
    with pytest.raises(NotImplementedError):
        routine(test=761)


def test_00762():
    with pytest.raises(NotImplementedError):
        routine(test=762)


def test_00763():
    with pytest.raises(NotImplementedError):
        routine(test=763)


def test_00764():
    with pytest.raises(NotImplementedError):
        routine(test=764)


def test_00765():
    with pytest.raises(NotImplementedError):
        routine(test=765)


def test_00766():
    with pytest.raises(NotImplementedError):
        routine(test=766)


def test_00767():
    with pytest.raises(NotImplementedError):
        routine(test=767)


def test_00768():
    with pytest.raises(NotImplementedError):
        routine(test=768)


def test_00769():
    with pytest.raises(NotImplementedError):
        routine(test=769)


def test_00770():
    with pytest.raises(NotImplementedError):
        routine(test=770)


def test_00771():
    with pytest.raises(NotImplementedError):
        routine(test=771)


def test_00772():
    with pytest.raises(NotImplementedError):
        routine(test=772)


def test_00773():
    with pytest.raises(NotImplementedError):
        routine(test=773)


def test_00774():
    with pytest.raises(NotImplementedError):
        routine(test=774)


def test_00775():
    with pytest.raises(NotImplementedError):
        routine(test=775)


def test_00776():
    with pytest.raises(NotImplementedError):
        routine(test=776)


def test_00777():
    with pytest.raises(NotImplementedError):
        routine(test=777)


def test_00778():
    with pytest.raises(NotImplementedError):
        routine(test=778)


def test_00779():
    with pytest.raises(NotImplementedError):
        routine(test=779)


def test_00780():
    with pytest.raises(NotImplementedError):
        routine(test=780)


def test_00781():
    assert routine(test=781)


def test_00782():
    assert routine(test=782)


def test_00783():
    assert routine(test=783)


def test_00784():
    assert routine(test=784)


def test_00785():
    assert routine(test=785)


def test_00786():
    assert routine(test=786)


def test_00787():
    assert routine(test=787)


def test_00788():
    assert routine(test=788)


def test_00789():
    with pytest.raises(NotImplementedError):
        routine(test=789)


def test_00790():
    with pytest.raises(NotImplementedError):
        routine(test=790)


def test_00791():
    with pytest.raises(NotImplementedError):
        routine(test=791)


def test_00792():
    assert routine(test=792)


def test_00793():
    assert routine(test=793)


def test_00794():
    assert routine(test=794)


def test_00795():
    assert routine(test=795)


def test_00796():
    assert routine(test=796)


def test_00797():
    assert routine(test=797)


def test_00798():
    assert routine(test=798)


def test_00799():
    assert routine(test=799)


def test_00800():
    assert routine(test=800)


def test_00801():
    assert routine(test=801)


def test_00802():
    assert routine(test=802)


def test_00803():
    assert routine(test=803)


def test_00804():
    assert routine(test=804)


def test_00805():
    assert routine(test=805)


def test_00806():
    assert routine(test=806)


def test_00807():
    assert routine(test=807)


def test_00808():
    assert routine(test=808)


def test_00809():
    assert routine(test=809)


def test_00810():
    assert routine(test=810)


def test_00811():
    assert routine(test=811)


def test_00812():
    assert routine(test=812)


def test_00813():
    assert routine(test=813)


def test_00814():
    assert routine(test=814)


def test_00815():
    assert routine(test=815)


def test_00816():
    assert routine(test=816)


def test_00817():
    assert routine(test=817)


def test_00818():
    assert routine(test=818)


def test_00819():
    assert routine(test=819)


def test_00820():
    assert routine(test=820)


def test_00821():
    assert routine(test=821)


def test_00822():
    assert routine(test=822)


def test_00823():
    assert routine(test=823)


def test_00824():
    assert routine(test=824)


def test_00825():
    assert routine(test=825)


def test_00826():
    assert routine(test=826)


# def test_00827():  # FIXME
#     assert routine(test=827)

# def test_00828():  # FIXME
#     assert routine(test=828)

# def test_00829():  # FIXME
#     assert routine(test=829)


def test_00830():
    assert routine(test=830)


def test_00831():
    assert routine(test=831)


def test_00832():
    assert routine(test=832)


def test_00833():
    assert routine(test=833)


def test_00834():
    assert routine(test=834)


def test_00835():
    assert routine(test=835)


def test_00836():
    assert routine(test=836)


def test_00837():
    assert routine(test=837)


def test_00838():
    assert routine(test=838)


def test_00839():
    assert routine(test=839)


def test_00840():
    assert routine(test=840)


def test_00841():
    assert routine(test=841)


def test_00842():
    assert routine(test=842)


def test_00843():
    assert routine(test=843)


def test_00844():
    assert routine(test=844)


def test_00845():
    with pytest.raises(NotImplementedError):
        routine(test=845)


def test_00846():
    with pytest.raises(NotImplementedError):
        routine(test=846)


def test_00847():
    with pytest.raises(NotImplementedError):
        routine(test=847)


def test_00848():
    with pytest.raises(NotImplementedError):
        routine(test=848)


def test_00849():
    with pytest.raises(NotImplementedError):
        routine(test=849)


def test_00850():
    with pytest.raises(NotImplementedError):
        routine(test=850)


def test_00851():
    assert routine(test=851)


def test_00852():
    assert routine(test=852)


def test_00853():
    assert routine(test=853)


def test_00854():
    assert routine(test=854)


def test_00855():
    assert routine(test=855)


def test_00856():
    assert routine(test=856)


def test_00857():
    assert routine(test=857)


def test_00858():
    assert routine(test=858)


def test_00859():
    assert routine(test=859)


def test_00860():
    assert routine(test=860)


def test_00861():
    assert routine(test=861)


def test_00862():
    assert routine(test=862)


def test_00863():
    assert routine(test=863)


def test_00864():
    assert routine(test=864)


def test_00865():
    assert routine(test=865)


def test_00866():
    assert routine(test=866)


def test_00867():
    assert routine(test=867)


def test_00868():
    assert routine(test=868)


def test_00869():
    assert routine(test=869)


# def test_00870():  # FIXME
#     assert routine(test=870)

# def test_00871():  # FIXME
#     assert routine(test=871)

# def test_00872():  # FIXME
#     assert routine(test=872)

# def test_00873():  # FIXME
#     assert routine(test=873)

# def test_00874():  # FIXME
#     assert routine(test=874)

# def test_00875():  # FIXME
#     assert routine(test=875)


def test_00876():
    assert routine(test=876)


def test_00877():
    assert routine(test=877)


def test_00878():
    assert routine(test=878)


# def test_00879():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=879)


def test_00880():
    assert routine(test=880)


def test_00881():
    assert routine(test=881)


def test_00882():
    assert routine(test=882)


def test_00883():
    with pytest.raises(NotImplementedError):
        routine(test=883)


def test_00884():
    with pytest.raises(NotImplementedError):
        routine(test=884)


def test_00885():
    with pytest.raises(NotImplementedError):
        routine(test=885)


def test_00886():
    with pytest.raises(NotImplementedError):
        routine(test=886)


def test_00887():
    with pytest.raises(NotImplementedError):
        routine(test=887)


def test_00888():
    assert routine(test=888)


def test_00889():
    assert routine(test=889)


def test_00890():
    assert routine(test=890)


def test_00891():
    with pytest.raises(NotImplementedError):
        routine(test=891)


def test_00892():
    with pytest.raises(NotImplementedError):
        routine(test=892)


def test_00893():
    with pytest.raises(NotImplementedError):
        routine(test=893)


def test_00894():
    assert routine(test=894)


def test_00895():
    assert routine(test=895)


def test_00896():
    assert routine(test=896)


def test_00897():
    assert routine(test=897)


# def test_00898():  # FIXME
#     assert routine(test=898)

# def test_00899():  # FIXME
#     assert routine(test=899)

# def test_00900():  # FIXME
#     assert routine(test=900)


def test_00901():
    with pytest.raises(NotImplementedError):
        routine(test=901)


def test_00902():
    with pytest.raises(NotImplementedError):
        routine(test=902)


def test_00903():
    with pytest.raises(NotImplementedError):
        routine(test=903)


def test_00904():
    with pytest.raises(NotImplementedError):
        routine(test=904)


def test_00905():
    with pytest.raises(NotImplementedError):
        routine(test=905)


def test_00906():
    with pytest.raises(NotImplementedError):
        routine(test=906)


def test_00907():
    with pytest.raises(NotImplementedError):
        routine(test=907)


def test_00908():
    with pytest.raises(NotImplementedError):
        routine(test=908)


def test_00909():
    with pytest.raises(NotImplementedError):
        routine(test=909)


def test_00910():
    with pytest.raises(NotImplementedError):
        routine(test=910)


def test_00911():
    with pytest.raises(NotImplementedError):
        routine(test=911)


def test_00912():
    with pytest.raises(NotImplementedError):
        routine(test=912)


def test_00913():
    with pytest.raises(NotImplementedError):
        routine(test=913)


def test_00914():
    with pytest.raises(NotImplementedError):
        routine(test=914)


def test_00915():
    with pytest.raises(NotImplementedError):
        routine(test=915)


def test_00916():
    with pytest.raises(NotImplementedError):
        routine(test=916)


def test_00917():
    with pytest.raises(NotImplementedError):
        routine(test=917)


def test_00918():
    with pytest.raises(NotImplementedError):
        routine(test=918)


def test_00919():
    with pytest.raises(NotImplementedError):
        routine(test=919)


# def test_00920():  # FIXME
#     assert routine(test=920)


# def test_00921():  # FIXME
#     assert routine(test=921)


# def test_00922():  # FIXME
#     assert routine(test=922)


def test_00923():
    assert routine(test=923)


def test_00924():
    assert routine(test=924)


def test_00925():
    assert routine(test=925)


def test_00926():
    with pytest.raises(NotImplementedError):
        routine(test=926)


def test_00927():
    with pytest.raises(NotImplementedError):
        routine(test=927)


def test_00928():
    with pytest.raises(NotImplementedError):
        routine(test=928)


def test_00929():
    with pytest.raises(NotImplementedError):
        routine(test=929)


def test_00930():
    with pytest.raises(NotImplementedError):
        routine(test=930)


def test_00931():
    with pytest.raises(NotImplementedError):
        routine(test=931)


def test_00932():
    with pytest.raises(NotImplementedError):
        routine(test=932)


def test_00933():
    with pytest.raises(NotImplementedError):
        routine(test=933)


def test_00934():
    with pytest.raises(NotImplementedError):
        routine(test=934)


def test_00935():
    with pytest.raises(NotImplementedError):
        routine(test=935)


def test_00936():
    with pytest.raises(NotImplementedError):
        routine(test=936)


def test_00937():
    with pytest.raises(NotImplementedError):
        routine(test=937)


def test_00938():
    with pytest.raises(NotImplementedError):
        routine(test=938)


def test_00939():
    with pytest.raises(NotImplementedError):
        routine(test=939)


def test_00940():
    with pytest.raises(NotImplementedError):
        routine(test=940)


def test_00941():
    with pytest.raises(NotImplementedError):
        routine(test=941)


def test_00942():
    with pytest.raises(NotImplementedError):
        routine(test=942)


def test_00943():
    with pytest.raises(NotImplementedError):
        routine(test=943)


def test_00944():
    with pytest.raises(NotImplementedError):
        routine(test=944)


def test_00945():
    with pytest.raises(NotImplementedError):
        routine(test=945)


def test_00946():
    with pytest.raises(NotImplementedError):
        routine(test=946)


def test_00947():
    with pytest.raises(NotImplementedError):
        routine(test=947)


def test_00948():
    with pytest.raises(NotImplementedError):
        routine(test=948)


def test_00949():
    assert routine(test=949)


# def test_00950():  # FIXME
#     assert routine(test=950)

# def test_00951():  # FIXME
#     assert routine(test=951)


def test_00952():
    with pytest.raises(NotImplementedError):
        routine(test=952)


def test_00953():
    with pytest.raises(NotImplementedError):
        routine(test=953)


# def test_00954():  # FIXME
#     assert routine(test=954)

# def test_00955():  # FIXME
#     assert routine(test=955)

# def test_00956():  # FIXME
#     assert routine(test=956)


def test_00957():
    assert routine(test=957)


# def test_00958():  # FIXME
#     assert routine(test=958)


def test_00959():
    with pytest.raises(ZeroDivisionError):
        routine(test=959)


# def test_00960():  # FIXME
#     assert routine(test=960)


def test_00961():
    assert routine(test=961)


def test_00962():
    with pytest.raises(NotImplementedError):
        routine(test=962)


def test_00963():
    with pytest.raises(NotImplementedError):
        routine(test=963)


def test_00964():
    with pytest.raises(NotImplementedError):
        routine(test=964)


def test_00965():
    with pytest.raises(NotImplementedError):
        routine(test=965)


def test_00966():
    with pytest.raises(NotImplementedError):
        routine(test=966)


def test_00967():
    with pytest.raises(NotImplementedError):
        routine(test=967)


# def test_00968():  # FIXME
#     assert routine(test=968)


def test_00969():
    with pytest.raises(NotImplementedError):
        routine(test=969)


def test_00970():
    with pytest.raises(NotImplementedError):
        routine(test=970)


def test_00971():
    with pytest.raises(NotImplementedError):
        routine(test=971)


def test_00972():
    with pytest.raises(NotImplementedError):
        routine(test=972)


# def test_00973():  # FIXME
#     assert routine(test=973)


def test_00974():
    with pytest.raises(NotImplementedError):
        routine(test=974)


def test_00975():
    with pytest.raises(NotImplementedError):
        routine(test=975)


def test_00976():
    with pytest.raises(NotImplementedError):
        routine(test=976)


def test_00977():
    with pytest.raises(NotImplementedError):
        routine(test=977)


def test_00978():
    with pytest.raises(NotImplementedError):
        routine(test=978)


def test_00979():
    with pytest.raises(NotImplementedError):
        routine(test=979)


def test_00980():
    with pytest.raises(NotImplementedError):
        routine(test=980)


def test_00981():
    with pytest.raises(NotImplementedError):
        routine(test=981)


def test_00982():
    with pytest.raises(NotImplementedError):
        routine(test=982)


def test_00983():
    with pytest.raises(NotImplementedError):
        routine(test=983)


def test_00984():
    with pytest.raises(NotImplementedError):
        routine(test=984)


def test_00985():
    with pytest.raises(NotImplementedError):
        routine(test=985)


# def test_00986():  # FIXME
#     assert routine(test=986)

# def test_00987():  # FIXME
#     assert routine(test=987)

# def test_00988():  # FIXME
#     assert routine(test=988)

# def test_00989():  # FIXME
#     assert routine(test=989)

# def test_00990():  # FIXME
#     assert routine(test=990)

# def test_00991():  # FIXME
#     assert routine(test=991)

# def test_00992():  # FIXME
#     assert routine(test=992)

# def test_00993():  # FIXME
#     assert routine(test=993)

# def test_00994():  # FIXME
#     assert routine(test=994)


def test_00995():
    with pytest.raises(NotImplementedError):
        routine(test=995)


def test_00996():
    with pytest.raises(NotImplementedError):
        routine(test=996)


def test_00997():
    with pytest.raises(NotImplementedError):
        routine(test=997)


# def test_00998():  # FIXME
#     assert routine(test=998)


def test_00999():
    with pytest.raises(NotImplementedError):
        routine(test=999)


def test_01000():
    with pytest.raises(NotImplementedError):
        routine(test=1000)


def test_01001():
    assert routine(test=1001)


def test_01002():
    assert routine(test=1002)


def test_01003():
    assert routine(test=1003)


def test_01004():
    assert routine(test=1004)


def test_01005():
    assert routine(test=1005)


def test_01006():
    assert routine(test=1006)


def test_01007():
    assert routine(test=1007)


def test_01008():
    assert routine(test=1008)


def test_01009():
    assert routine(test=1009)


def test_01010():
    assert routine(test=1010)


def test_01011():
    assert routine(test=1011)


def test_01012():
    assert routine(test=1012)


def test_01013():
    assert routine(test=1013)


def test_01014():
    assert routine(test=1014)


def test_01015():
    assert routine(test=1015)


def test_01016():
    assert routine(test=1016)


def test_01017():
    assert routine(test=1017)


def test_01018():
    assert routine(test=1018)


def test_01019():
    assert routine(test=1019)


def test_01020():
    assert routine(test=1020)


def test_01021():
    assert routine(test=1021)


def test_01022():
    assert routine(test=1022)


def test_01023():
    assert routine(test=1023)


def test_01024():
    assert routine(test=1024)


def test_01025():
    assert routine(test=1025)


def test_01026():
    assert routine(test=1026)


# def test_01027():  # FIXME
#     assert routine(test=1027)

# def test_01028():  # FIXME
#     assert routine(test=1028)

# def test_01029():  # FIXME
#     assert routine(test=1029)


def test_01030():
    assert routine(test=1030)


def test_01031():
    assert routine(test=1031)


def test_01032():
    assert routine(test=1032)


def test_01033():
    assert routine(test=1033)


def test_01034():
    assert routine(test=1034)


def test_01035():
    assert routine(test=1035)


def test_01036():
    assert routine(test=1036)


def test_01037():
    assert routine(test=1037)


def test_01038():
    assert routine(test=1038)


def test_01039():
    assert routine(test=1039)


def test_01040():
    assert routine(test=1040)


def test_01041():
    assert routine(test=1041)


def test_01042():
    assert routine(test=1042)


def test_01043():
    assert routine(test=1043)


def test_01044():
    assert routine(test=1044)


def test_01045():
    with pytest.raises(NotImplementedError):
        routine(test=1045)


def test_01046():
    with pytest.raises(NotImplementedError):
        routine(test=1046)


def test_01047():
    with pytest.raises(NotImplementedError):
        routine(test=1047)


def test_01048():
    with pytest.raises(NotImplementedError):
        routine(test=1048)


def test_01049():
    with pytest.raises(NotImplementedError):
        routine(test=1049)


def test_01050():
    with pytest.raises(NotImplementedError):
        routine(test=1050)


# def test_01051():  # FIXME
#     assert routine(test=1051)

# def test_01052():  # FIXME
#     assert routine(test=1052)

# def test_01053():  # FIXME
#     assert routine(test=1053)


def test_01054():
    assert routine(test=1054)


def test_01055():
    assert routine(test=1055)


def test_01056():
    assert routine(test=1056)


def test_01057():
    assert routine(test=1057)


def test_01058():
    assert routine(test=1058)


def test_01059():
    assert routine(test=1059)


def test_01060():
    assert routine(test=1060)


def test_01061():
    assert routine(test=1061)


def test_01062():
    assert routine(test=1062)


def test_01063():
    assert routine(test=1063)


def test_01064():
    with pytest.raises(NotImplementedError):
        routine(test=1064)


def test_01065():
    with pytest.raises(NotImplementedError):
        routine(test=1065)


def test_01066():
    with pytest.raises(NotImplementedError):
        routine(test=1066)


def test_01067():
    with pytest.raises(NotImplementedError):
        routine(test=1067)


def test_01068():
    with pytest.raises(NotImplementedError):
        routine(test=1068)


def test_01069():
    with pytest.raises(NotImplementedError):
        routine(test=1069)


def test_01070():
    with pytest.raises(NotImplementedError):
        routine(test=1070)


def test_01071():
    with pytest.raises(NotImplementedError):
        routine(test=1071)


def test_01072():
    with pytest.raises(NotImplementedError):
        routine(test=1072)


def test_01073():
    with pytest.raises(NotImplementedError):
        routine(test=1073)


def test_01074():
    with pytest.raises(NotImplementedError):
        routine(test=1074)


def test_01075():
    with pytest.raises(NotImplementedError):
        routine(test=1075)


def test_01076():
    with pytest.raises(NotImplementedError):
        routine(test=1076)


def test_01077():
    with pytest.raises(NotImplementedError):
        routine(test=1077)


def test_01078():
    with pytest.raises(NotImplementedError):
        routine(test=1078)


def test_01079():
    with pytest.raises(NotImplementedError):
        routine(test=1079)


def test_01080():
    with pytest.raises(NotImplementedError):
        routine(test=1080)


def test_01081():
    with pytest.raises(NotImplementedError):
        routine(test=1081)


def test_01082():
    with pytest.raises(NotImplementedError):
        routine(test=1082)


def test_01083():
    with pytest.raises(NotImplementedError):
        routine(test=1083)


def test_01084():
    with pytest.raises(NotImplementedError):
        routine(test=1084)


def test_01085():
    with pytest.raises(NotImplementedError):
        routine(test=1085)


def test_01086():
    with pytest.raises(NotImplementedError):
        routine(test=1086)


def test_01087():
    with pytest.raises(NotImplementedError):
        routine(test=1087)


def test_01088():
    with pytest.raises(NotImplementedError):
        routine(test=1088)


def test_01089():
    with pytest.raises(NotImplementedError):
        routine(test=1089)


def test_01090():
    with pytest.raises(NotImplementedError):
        routine(test=1090)


def test_01091():
    with pytest.raises(NotImplementedError):
        routine(test=1091)


def test_01092():
    with pytest.raises(NotImplementedError):
        routine(test=1092)


def test_01093():
    with pytest.raises(NotImplementedError):
        routine(test=1093)


def test_01094():
    with pytest.raises(NotImplementedError):
        routine(test=1094)


def test_01095():
    with pytest.raises(NotImplementedError):
        routine(test=1095)


def test_01096():
    with pytest.raises(NotImplementedError):
        routine(test=1096)


def test_01097():
    with pytest.raises(NotImplementedError):
        routine(test=1097)


def test_01098():
    with pytest.raises(NotImplementedError):
        routine(test=1098)


def test_01099():
    with pytest.raises(NotImplementedError):
        routine(test=1099)


def test_01100():
    with pytest.raises(NotImplementedError):
        routine(test=1100)


def test_01101():
    with pytest.raises(NotImplementedError):
        routine(test=1101)


def test_01102():
    with pytest.raises(NotImplementedError):
        routine(test=1102)


def test_01103():
    with pytest.raises(NotImplementedError):
        routine(test=1103)


def test_01104():
    with pytest.raises(NotImplementedError):
        routine(test=1104)


def test_01105():
    with pytest.raises(NotImplementedError):
        routine(test=1105)


def test_01106():
    with pytest.raises(NotImplementedError):
        routine(test=1106)


def test_01107():
    with pytest.raises(NotImplementedError):
        routine(test=1107)


def test_01108():
    with pytest.raises(NotImplementedError):
        routine(test=1108)


def test_01109():
    with pytest.raises(NotImplementedError):
        routine(test=1109)


def test_01110():
    with pytest.raises(NotImplementedError):
        routine(test=1110)


def test_01111():
    with pytest.raises(NotImplementedError):
        routine(test=1111)


def test_01112():
    with pytest.raises(SyntaxError):
        routine(test=1112)


def test_01113():
    with pytest.raises(SyntaxError):
        routine(test=1113)


def test_01114():
    with pytest.raises(SyntaxError):
        routine(test=1114)


def test_01115():
    with pytest.raises(SyntaxError):
        routine(test=1115)


def test_01116():
    with pytest.raises(SyntaxError):
        routine(test=1116)


def test_01117():
    with pytest.raises(NotImplementedError):
        routine(test=1117)


def test_01118():
    with pytest.raises(NotImplementedError):
        routine(test=1118)


def test_01119():
    with pytest.raises(NotImplementedError):
        routine(test=1119)


def test_01120():
    with pytest.raises(NotImplementedError):
        routine(test=1120)


def test_01121():
    with pytest.raises(NotImplementedError):
        routine(test=1121)


def test_01122():
    with pytest.raises(NotImplementedError):
        routine(test=1122)


def test_01123():
    with pytest.raises(NotImplementedError):
        routine(test=1123)


def test_01124():
    with pytest.raises(NotImplementedError):
        routine(test=1124)


def test_01125():
    with pytest.raises(NotImplementedError):
        routine(test=1125)


def test_01126():
    with pytest.raises(NotImplementedError):
        routine(test=1126)


def test_01127():
    with pytest.raises(NotImplementedError):
        routine(test=1127)


def test_01128():
    with pytest.raises(NotImplementedError):
        routine(test=1128)


def test_01129():
    with pytest.raises(NotImplementedError):
        routine(test=1129)


def test_01130():
    with pytest.raises(NotImplementedError):
        routine(test=1130)


def test_01131():
    with pytest.raises(NotImplementedError):
        routine(test=1131)


def test_01132():
    with pytest.raises(NotImplementedError):
        routine(test=1132)


def test_01133():
    with pytest.raises(NotImplementedError):
        routine(test=1133)


def test_01134():
    with pytest.raises(NotImplementedError):
        routine(test=1134)


def test_01135():
    with pytest.raises(NotImplementedError):
        routine(test=1135)


def test_01136():
    with pytest.raises(NotImplementedError):
        routine(test=1136)


def test_01137():
    with pytest.raises(NotImplementedError):
        routine(test=1137)


def test_01138():
    with pytest.raises(NotImplementedError):
        routine(test=1138)


def test_01139():
    with pytest.raises(NotImplementedError):
        routine(test=1139)


def test_01140():
    with pytest.raises(NotImplementedError):
        routine(test=1140)


def test_01141():
    with pytest.raises(NotImplementedError):
        routine(test=1141)


def test_01142():
    with pytest.raises(NotImplementedError):
        routine(test=1142)


def test_01143():
    with pytest.raises(NotImplementedError):
        routine(test=1143)


def test_01144():
    with pytest.raises(NotImplementedError):
        routine(test=1144)


def test_01145():
    with pytest.raises(NotImplementedError):
        routine(test=1145)


def test_01146():
    with pytest.raises(NotImplementedError):
        routine(test=1146)


def test_01147():
    with pytest.raises(NotImplementedError):
        routine(test=1147)


def test_01148():
    with pytest.raises(NotImplementedError):
        routine(test=1148)


def test_01149():
    with pytest.raises(NotImplementedError):
        routine(test=1149)


def test_01150():
    with pytest.raises(NotImplementedError):
        routine(test=1150)


def test_01151():
    with pytest.raises(NotImplementedError):
        routine(test=1151)


def test_01152():
    with pytest.raises(NotImplementedError):
        routine(test=1152)


def test_01153():
    with pytest.raises(NotImplementedError):
        routine(test=1153)


def test_01154():
    with pytest.raises(NotImplementedError):
        routine(test=1154)


def test_01155():
    with pytest.raises(NotImplementedError):
        routine(test=1155)


def test_01156():
    with pytest.raises(NotImplementedError):
        routine(test=1156)


def test_01157():
    with pytest.raises(NotImplementedError):
        routine(test=1157)


def test_01158():
    with pytest.raises(NotImplementedError):
        routine(test=1158)


def test_01159():
    with pytest.raises(NotImplementedError):
        routine(test=1159)


def test_01160():
    with pytest.raises(NotImplementedError):
        routine(test=1160)


def test_01161():
    with pytest.raises(NotImplementedError):
        routine(test=1161)


def test_01162():
    with pytest.raises(NotImplementedError):
        routine(test=1162)


def test_01163():
    with pytest.raises(NotImplementedError):
        routine(test=1163)


def test_01164():
    with pytest.raises(NotImplementedError):
        routine(test=1164)


def test_01165():
    with pytest.raises(NotImplementedError):
        routine(test=1165)


def test_01166():
    with pytest.raises(NotImplementedError):
        routine(test=1166)


def test_01167():
    with pytest.raises(NotImplementedError):
        routine(test=1167)


def test_01168():
    with pytest.raises(NotImplementedError):
        routine(test=1168)


def test_01169():
    with pytest.raises(NotImplementedError):
        routine(test=1169)


def test_01170():
    with pytest.raises(NotImplementedError):
        routine(test=1170)


def test_01171():
    with pytest.raises(NotImplementedError):
        routine(test=1171)


def test_01172():
    with pytest.raises(NotImplementedError):
        routine(test=1172)


def test_01173():
    with pytest.raises(NotImplementedError):
        routine(test=1173)


def test_01174():
    with pytest.raises(NotImplementedError):
        routine(test=1174)


def test_01175():
    with pytest.raises(NotImplementedError):
        routine(test=1175)


def test_01176():
    with pytest.raises(NotImplementedError):
        routine(test=1176)


def test_01177():
    with pytest.raises(NotImplementedError):
        routine(test=1177)


def test_01178():
    with pytest.raises(NotImplementedError):
        routine(test=1178)


def test_01179():
    with pytest.raises(NotImplementedError):
        routine(test=1179)


def test_01180():
    with pytest.raises(NotImplementedError):
        routine(test=1180)


def test_01181():
    with pytest.raises(NotImplementedError):
        routine(test=1181)


def test_01182():
    with pytest.raises(NotImplementedError):
        routine(test=1182)


def test_01183():
    with pytest.raises(NotImplementedError):
        routine(test=1183)


# def test_01184():  # FIXME
#     assert routine(test=1184)


# def test_01185():  # FIXME
#     assert routine(test=1185)


# def test_01186():  # FIXME
#     assert routine(test=1186)

# def test_01187():  # FIXME
#     assert routine(test=1187)

# def test_01188():  # FIXME
#     assert routine(test=1188)

# def test_01189():  # FIXME
#     assert routine(test=1189)

# def test_01190():  # FIXME
#     assert routine(test=1190)

# def test_01191():  # FIXME
#     assert routine(test=1191)

# def test_01192():  # FIXME
#     assert routine(test=1192)

# def test_01193():  # FIXME
#     assert routine(test=1193)

# def test_01194():  # FIXME
#     assert routine(test=1194)

# def test_01195():  # FIXME
#     assert routine(test=1195)

# def test_01196():  # FIXME
#     assert routine(test=1196)


def test_01197():
    assert routine(test=1197)


def test_01198():
    with pytest.raises(NotImplementedError):
        routine(test=1198)


def test_01199():
    with pytest.raises(NotImplementedError):
        routine(test=1199)


def test_01200():
    with pytest.raises(NotImplementedError):
        routine(test=1200)


def test_01201():
    with pytest.raises(NotImplementedError):
        routine(test=1201)


def test_01202():
    with pytest.raises(NotImplementedError):
        routine(test=1202)


def test_01203():
    with pytest.raises(NotImplementedError):
        routine(test=1203)


def test_01204():
    with pytest.raises(NotImplementedError):
        routine(test=1204)


# def test_01205():  # FIXME
#     assert routine(test=1205)


def test_01206():
    with pytest.raises(NotImplementedError):
        routine(test=1206)


def test_01207():
    with pytest.raises(NotImplementedError):
        routine(test=1207)


def test_01208():
    with pytest.raises(NotImplementedError):
        routine(test=1208)


def test_01209():
    with pytest.raises(NotImplementedError):
        routine(test=1209)


def test_01210():
    with pytest.raises(NotImplementedError):
        routine(test=1210)


def test_01211():
    with pytest.raises(NotImplementedError):
        routine(test=1211)


def test_01212():
    with pytest.raises(NotImplementedError):
        routine(test=1212)


def test_01213():
    with pytest.raises(NotImplementedError):
        routine(test=1213)


def test_01214():
    with pytest.raises(NotImplementedError):
        routine(test=1214)


def test_01215():
    with pytest.raises(NotImplementedError):
        routine(test=1215)


def test_01216():
    with pytest.raises(NotImplementedError):
        routine(test=1216)


# def test_01217():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1217)


# def test_01218():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1218)


def test_01219():
    assert routine(test=1219)


def test_01220():
    assert routine(test=1220)


def test_01221():
    with pytest.raises(NotImplementedError):
        routine(test=1221)


def test_01222():
    with pytest.raises(NotImplementedError):
        routine(test=1222)


def test_01223():
    assert routine(test=1223)


def test_01224():
    with pytest.raises(KeyError):
        routine(test=1224)


def test_01225():
    with pytest.raises(KeyError):
        routine(test=1225)


def test_01226():
    with pytest.raises(NotImplementedError):
        routine(test=1226)


def test_01227():
    with pytest.raises(NotImplementedError):
        routine(test=1227)


def test_01228():
    with pytest.raises(NotImplementedError):
        routine(test=1228)


def test_01229():
    with pytest.raises(NotImplementedError):
        routine(test=1229)


def test_01230():
    with pytest.raises(NotImplementedError):
        routine(test=1230)


def test_01231():
    with pytest.raises(KeyError):
        routine(test=1231)


def test_01232():
    assert routine(test=1232)


def test_01233():
    with pytest.raises(KeyError):
        routine(test=1233)


def test_01234():
    assert routine(test=1234)


def test_01235():
    assert routine(test=1235)


def test_01236():
    assert routine(test=1236)


def test_01237():
    with pytest.raises(NotImplementedError):
        routine(test=1237)


def test_01238():
    with pytest.raises(NotImplementedError):
        routine(test=1238)


def test_01239():
    with pytest.raises(NotImplementedError):
        routine(test=1239)


def test_01240():
    with pytest.raises(NotImplementedError):
        routine(test=1240)


def test_01241():
    with pytest.raises(NotImplementedError):
        routine(test=1241)


def test_01242():
    with pytest.raises(NotImplementedError):
        routine(test=1242)


def test_01243():
    with pytest.raises(NotImplementedError):
        routine(test=1243)


def test_01244():
    assert routine(test=1244)


def test_01245():
    assert routine(test=1245)


def test_01246():
    assert routine(test=1246)


def test_01247():
    assert routine(test=1247)


def test_01248():
    with pytest.raises(NotImplementedError):
        routine(test=1248)


def test_01249():
    with pytest.raises(NotImplementedError):
        routine(test=1249)


def test_01250():
    with pytest.raises(NotImplementedError):
        routine(test=1250)


def test_01251():
    with pytest.raises(NotImplementedError):
        routine(test=1251)


def test_01252():
    with pytest.raises(NotImplementedError):
        routine(test=1252)


def test_01253():
    with pytest.raises(NotImplementedError):
        routine(test=1253)


def test_01254():
    with pytest.raises(NotImplementedError):
        routine(test=1254)


def test_01255():
    with pytest.raises(NotImplementedError):
        routine(test=1255)


def test_01256():
    with pytest.raises(NotImplementedError):
        routine(test=1256)


def test_01257():
    with pytest.raises(NotImplementedError):
        routine(test=1257)


def test_01258():
    with pytest.raises(NotImplementedError):
        routine(test=1258)


def test_01259():
    with pytest.raises(NotImplementedError):
        routine(test=1259)


def test_01260():
    with pytest.raises(NotImplementedError):
        routine(test=1260)


def test_01261():
    with pytest.raises(NotImplementedError):
        routine(test=1261)


def test_01262():
    with pytest.raises(NotImplementedError):
        routine(test=1262)


def test_01263():
    with pytest.raises(NotImplementedError):
        routine(test=1263)


def test_01264():
    with pytest.raises(NotImplementedError):
        routine(test=1264)


def test_01265():
    with pytest.raises(NotImplementedError):
        routine(test=1265)


def test_01266():
    with pytest.raises(NotImplementedError):
        routine(test=1266)


def test_01267():
    with pytest.raises(NotImplementedError):
        routine(test=1267)


def test_01268():
    with pytest.raises(NotImplementedError):
        routine(test=1268)


def test_01269():
    with pytest.raises(NotImplementedError):
        routine(test=1269)


def test_01270():
    with pytest.raises(NotImplementedError):
        routine(test=1270)


def test_01271():
    assert routine(test=1271)


# def test_01272():  # FIXME
#     assert routine(test=1272)


# def test_01273():  # FIXME
#     assert routine(test=1273)


def test_01274():
    with pytest.raises(NotImplementedError):
        routine(test=1274)


# def test_01275():  # FIXME
#     assert routine(test=1275)


# def test_01276():
#     with pytest.raises(np.AxisError):
#         assert routine(test=1276)


def test_01277():
    with pytest.raises(NameError):
        routine(test=1277)


def test_01278():
    assert routine(test=1278)


def test_01279():
    with pytest.raises(NotImplementedError):
        routine(test=1279)


def test_01280():
    with pytest.raises(ValueError):
        routine(test=1280)


def test_01281():
    with pytest.raises(TypeError):
        routine(test=1281)


# def test_01282():  # FIXME
#     assert routine(test=1282)


# def test_01283():  # FIXME
#     assert routine(test=1283)


def test_01284():
    with pytest.raises(NotImplementedError):
        routine(test=1284)


def test_01285():
    with pytest.raises(NotImplementedError):
        routine(test=1285)


def test_01286():
    with pytest.raises(NotImplementedError):
        routine(test=1286)


def test_01287():
    with pytest.raises(NotImplementedError):
        routine(test=1287)


def test_01288():
    assert routine(test=1288)


# def test_01289():  # FIXME
#     assert routine(test=1289)


def test_01290():
    with pytest.raises(NotImplementedError):
        routine(test=1290)


def test_01291():
    assert routine(test=1291)


def test_01292():
    with pytest.raises(NotImplementedError):
        routine(test=1292)


def test_01293():
    with pytest.raises(NotImplementedError):
        routine(test=1293)


def test_01294():
    with pytest.raises(NotImplementedError):
        routine(test=1294)


def test_01295():
    with pytest.raises(NotImplementedError):
        routine(test=1295)


def test_01296():
    with pytest.raises(NotImplementedError):
        routine(test=1296)


def test_01297():
    with pytest.raises(NotImplementedError):
        routine(test=1297)


def test_01298():
    with pytest.raises(NotImplementedError):
        routine(test=1298)


def test_01299():
    with pytest.raises(NotImplementedError):
        routine(test=1299)


def test_01300():
    with pytest.raises(KeyError):
        routine(test=1300)


def test_01301():
    with pytest.raises(KeyError):
        routine(test=1301)


def test_01302():
    with pytest.raises(NotImplementedError):
        routine(test=1302)


def test_01303():
    with pytest.raises(NotImplementedError):
        routine(test=1303)


def test_01304():
    with pytest.raises(NotImplementedError):
        routine(test=1304)


def test_01305():
    with pytest.raises(NotImplementedError):
        routine(test=1305)


def test_01306():
    with pytest.raises(KeyError):
        routine(test=1306)


# def test_01307():  # FIXME
#     assert routine(test=1307)

# def test_01308():  # FIXME
#     assert routine(test=1308)


def test_01309():
    with pytest.raises(NotImplementedError):
        routine(test=1309)


def test_01310():
    assert routine(test=1310)


# def test_01311():  # FIXME
#     assert routine(test=1311)


# def test_01312():  # FIXME
#     assert routine(test=1312)


# def test_01313():  # FIXME
#     assert routine(test=1313)


# def test_01314():  # FIXME
#     assert routine(test=1314)


# def test_01315():  # FIXME
#     assert routine(test=1315)


# def test_01316():  # FIXME
#     assert routine(test=1316)


def test_01317():
    assert routine(test=1317)


def test_01318():
    with pytest.raises(NotImplementedError):
        routine(test=1318)


def test_01319():
    with pytest.raises(NotImplementedError):
        routine(test=1319)


def test_01320():
    with pytest.raises(NotImplementedError):
        routine(test=1320)


def test_01321():
    with pytest.raises(NotImplementedError):
        routine(test=1321)


def test_01322():
    with pytest.raises(NotImplementedError):
        routine(test=1322)


# def test_01323():  # FIXME
#     assert routine(test=1323)


def test_01324():
    with pytest.raises(NotImplementedError):
        routine(test=1324)


def test_01325():
    with pytest.raises(NotImplementedError):
        routine(test=1325)


def test_01326():
    with pytest.raises(NotImplementedError):
        routine(test=1326)


def test_01327():
    with pytest.raises(NotImplementedError):
        routine(test=1327)


def test_01328():
    with pytest.raises(NotImplementedError):
        routine(test=1328)


def test_01329():
    with pytest.raises(NotImplementedError):
        routine(test=1329)


def test_01330():
    with pytest.raises(NotImplementedError):
        routine(test=1330)


def test_01331():
    with pytest.raises(NotImplementedError):
        routine(test=1331)


def test_01332():
    with pytest.raises(NotImplementedError):
        routine(test=1332)


def test_01333():
    with pytest.raises(NotImplementedError):
        routine(test=1333)


def test_01334():
    with pytest.raises(NotImplementedError):
        routine(test=1334)


def test_01335():
    with pytest.raises(NotImplementedError):
        routine(test=1335)


def test_01336():
    with pytest.raises(NotImplementedError):
        routine(test=1336)


def test_01337():
    with pytest.raises(NotImplementedError):
        routine(test=1337)


def test_01338():
    with pytest.raises(NotImplementedError):
        routine(test=1338)


def test_01339():
    with pytest.raises(NotImplementedError):
        routine(test=1339)


def test_01340():
    with pytest.raises(NotImplementedError):
        routine(test=1340)


def test_01341():
    assert routine(test=1341)


def test_01342():
    assert routine(test=1342)


# def test_01343():  # FIXME
#     assert routine(test=1343)


def test_01344():
    with pytest.raises(NotImplementedError):
        routine(test=1344)


def test_01345():
    with pytest.raises(NotImplementedError):
        routine(test=1345)


def test_01346():
    with pytest.raises(NotImplementedError):
        routine(test=1346)


def test_01347():
    with pytest.raises(NotImplementedError):
        routine(test=1347)


def test_01348():
    with pytest.raises(NotImplementedError):
        routine(test=1348)


def test_01349():
    with pytest.raises(NotImplementedError):
        routine(test=1349)


def test_01350():
    with pytest.raises(NotImplementedError):
        routine(test=1350)


def test_01351():
    with pytest.raises(NotImplementedError):
        routine(test=1351)


def test_01352():
    with pytest.raises(NotImplementedError):
        routine(test=1352)


def test_01353():
    with pytest.raises(NotImplementedError):
        routine(test=1353)


def test_01354():
    with pytest.raises(NotImplementedError):
        routine(test=1354)


def test_01355():
    with pytest.raises(NotImplementedError):
        routine(test=1355)


def test_01356():
    with pytest.raises(NotImplementedError):
        routine(test=1356)


def test_01357():
    with pytest.raises(NotImplementedError):
        routine(test=1357)


def test_01358():
    with pytest.raises(NotImplementedError):
        routine(test=1358)


def test_01359():
    with pytest.raises(NotImplementedError):
        routine(test=1359)


def test_01360():
    with pytest.raises(NotImplementedError):
        routine(test=1360)


def test_01361():
    with pytest.raises(NotImplementedError):
        routine(test=1361)


def test_01362():
    with pytest.raises(NotImplementedError):
        routine(test=1362)


def test_01363():
    with pytest.raises(NotImplementedError):
        routine(test=1363)


def test_01364():
    with pytest.raises(NotImplementedError):
        routine(test=1364)


def test_01365():
    with pytest.raises(NotImplementedError):
        routine(test=1365)


def test_01366():
    with pytest.raises(NotImplementedError):
        routine(test=1366)


def test_01367():
    with pytest.raises(NotImplementedError):
        routine(test=1367)


def test_01368():
    with pytest.raises(NotImplementedError):
        routine(test=1368)


def test_01369():
    with pytest.raises(NotImplementedError):
        routine(test=1369)


def test_01370():
    with pytest.raises(NotImplementedError):
        routine(test=1370)


def test_01371():
    with pytest.raises(NotImplementedError):
        routine(test=1371)


def test_01372():
    with pytest.raises(NotImplementedError):
        routine(test=1372)


def test_01373():
    with pytest.raises(NotImplementedError):
        routine(test=1373)


def test_01374():
    with pytest.raises(NotImplementedError):
        routine(test=1374)


def test_01375():
    with pytest.raises(NotImplementedError):
        routine(test=1375)


def test_01376():
    with pytest.raises(NotImplementedError):
        routine(test=1376)


def test_01377():
    with pytest.raises(NotImplementedError):
        routine(test=1377)


def test_01378():
    with pytest.raises(NotImplementedError):
        routine(test=1378)


def test_01379():
    with pytest.raises(NotImplementedError):
        routine(test=1379)


def test_01380():
    with pytest.raises(NotImplementedError):
        routine(test=1380)


def test_01381():
    with pytest.raises(NotImplementedError):
        routine(test=1381)


def test_01382():
    with pytest.raises(NotImplementedError):
        routine(test=1382)


def test_01383():
    with pytest.raises(NotImplementedError):
        routine(test=1383)


def test_01384():
    with pytest.raises(NotImplementedError):
        routine(test=1384)


def test_01385():
    with pytest.raises(NotImplementedError):
        routine(test=1385)


def test_01386():
    with pytest.raises(NotImplementedError):
        routine(test=1386)


def test_01387():
    with pytest.raises(NotImplementedError):
        routine(test=1387)


def test_01388():
    with pytest.raises(NotImplementedError):
        routine(test=1388)


def test_01389():
    with pytest.raises(NotImplementedError):
        routine(test=1389)


def test_01390():
    with pytest.raises(NotImplementedError):
        routine(test=1390)


def test_01391():
    with pytest.raises(NotImplementedError):
        routine(test=1391)


def test_01392():
    with pytest.raises(NotImplementedError):
        routine(test=1392)


def test_01393():
    with pytest.raises(NotImplementedError):
        routine(test=1393)


def test_01394():
    with pytest.raises(NotImplementedError):
        routine(test=1394)


# def test_01395():  # FIXME
#     with pytest.raises(KeyError):
#         routine(test=1395)


# def test_01396():  # FIXME
#     assert routine(test=1396)

# def test_01397():  # FIXME
#     assert routine(test=1397)

# def test_01398():  # FIXME
#     assert routine(test=1398)

# def test_01399():  # FIXME
#     assert routine(test=1399)


def test_01400():
    with pytest.raises(NotImplementedError):
        routine(test=1400)


def test_01401():
    with pytest.raises(NotImplementedError):
        routine(test=1401)


def test_01402():
    with pytest.raises(NotImplementedError):
        routine(test=1402)


def test_01403():
    with pytest.raises(NotImplementedError):
        routine(test=1403)


def test_01404():
    with pytest.raises(NotImplementedError):
        routine(test=1404)


def test_01405():
    with pytest.raises(NotImplementedError):
        routine(test=1405)


def test_01406():
    with pytest.raises(NotImplementedError):
        routine(test=1406)


def test_01407():
    with pytest.raises(NotImplementedError):
        routine(test=1407)


def test_01408():
    with pytest.raises(NotImplementedError):
        routine(test=1408)


def test_01409():
    with pytest.raises(NotImplementedError):
        routine(test=1409)


def test_01410():
    with pytest.raises(NotImplementedError):
        routine(test=1410)


def test_01411():
    with pytest.raises(NotImplementedError):
        routine(test=1411)


def test_01412():
    with pytest.raises(NotImplementedError):
        routine(test=1412)


def test_01413():
    with pytest.raises(NotImplementedError):
        routine(test=1413)


def test_01414():
    with pytest.raises(NotImplementedError):
        routine(test=1414)


def test_01415():
    with pytest.raises(NotImplementedError):
        routine(test=1415)


def test_01416():
    with pytest.raises(NotImplementedError):
        routine(test=1416)


def test_01417():
    with pytest.raises(NotImplementedError):
        routine(test=1417)


def test_01418():
    with pytest.raises(NotImplementedError):
        routine(test=1418)


def test_01419():
    with pytest.raises(NotImplementedError):
        routine(test=1419)


def test_01420():
    assert routine(test=1420)


def test_01421():
    assert routine(test=1421)


def test_01422():
    assert routine(test=1422)


def test_01423():
    assert routine(test=1423)


def test_01424():
    assert routine(test=1424)


def test_01425():
    assert routine(test=1425)


def test_01426():
    assert routine(test=1426)


def test_01427():
    assert routine(test=1427)


def test_01428():
    assert routine(test=1428)


def test_01429():
    assert routine(test=1429)


def test_01430():
    assert routine(test=1430)


def test_01431():
    assert routine(test=1431)


def test_01432():
    assert routine(test=1432)


def test_01433():
    assert routine(test=1433)


def test_01434():
    with pytest.raises(NotImplementedError):
        routine(test=1434)


# def test_01435():  # FIXME
#     assert routine(test=1435)


def test_01436():
    with pytest.raises(NotImplementedError):
        routine(test=1436)


# def test_01437():  # FIXME
#     assert routine(test=1437)


def test_01438():
    with pytest.raises(NotImplementedError):
        routine(test=1438)


# def test_01439():  # FIXME
#     assert routine(test=1439)


def test_01440():
    with pytest.raises(NotImplementedError):
        routine(test=1440)


# def test_01441():  # FIXME
#     assert routine(test=1441)


def test_01442():
    with pytest.raises(NotImplementedError):
        routine(test=1442)


# def test_01443():  # FIXME
#     assert routine(test=1443)


def test_01444():
    with pytest.raises(NotImplementedError):
        routine(test=1444)


def test_01445():
    with pytest.raises(NotImplementedError):
        routine(test=1445)


def test_01446():
    with pytest.raises(NotImplementedError):
        routine(test=1446)


def test_01447():
    with pytest.raises(NotImplementedError):
        routine(test=1447)


def test_01448():
    with pytest.raises(NotImplementedError):
        routine(test=1448)


def test_01449():
    with pytest.raises(NotImplementedError):
        routine(test=1449)


def test_01450():
    with pytest.raises(NotImplementedError):
        routine(test=1450)


def test_01451():
    with pytest.raises(NotImplementedError):
        routine(test=1451)


def test_01452():
    with pytest.raises(NotImplementedError):
        routine(test=1452)


def test_01453():
    with pytest.raises(NotImplementedError):
        routine(test=1453)


def test_01454():
    with pytest.raises(NotImplementedError):
        routine(test=1454)


def test_01455():
    with pytest.raises(NotImplementedError):
        routine(test=1455)


def test_01456():
    with pytest.raises(NotImplementedError):
        routine(test=1456)


def test_01457():
    with pytest.raises(NotImplementedError):
        routine(test=1457)


def test_01458():
    with pytest.raises(NotImplementedError):
        routine(test=1458)


def test_01459():
    with pytest.raises(NotImplementedError):
        routine(test=1459)


def test_01460():
    with pytest.raises(NotImplementedError):
        routine(test=1460)


def test_01461():
    with pytest.raises(NotImplementedError):
        routine(test=1461)


def test_01462():
    with pytest.raises(NotImplementedError):
        routine(test=1462)


def test_01463():
    with pytest.raises(NotImplementedError):
        routine(test=1463)


def test_01464():
    assert routine(test=1464)


def test_01465():
    assert routine(test=1465)


def test_01466():
    with pytest.raises(NotImplementedError):
        routine(test=1466)


def test_01467():
    with pytest.raises(NotImplementedError):
        routine(test=1467)


def test_01468():
    with pytest.raises(NotImplementedError):
        routine(test=1468)


def test_01469():
    with pytest.raises(NotImplementedError):
        routine(test=1469)


def test_01470():
    with pytest.raises(NotImplementedError):
        routine(test=1470)


def test_01471():
    with pytest.raises(NotImplementedError):
        routine(test=1471)


def test_01472():
    with pytest.raises(NotImplementedError):
        routine(test=1472)


def test_01473():
    with pytest.raises(NotImplementedError):
        routine(test=1473)


def test_01474():
    with pytest.raises(NotImplementedError):
        routine(test=1474)


def test_01475():
    with pytest.raises(NotImplementedError):
        routine(test=1475)


def test_01476():
    with pytest.raises(NotImplementedError):
        routine(test=1476)


def test_01477():
    with pytest.raises(NotImplementedError):
        routine(test=1477)


def test_01478():
    with pytest.raises(NotImplementedError):
        routine(test=1478)


def test_01479():
    assert routine(test=1479)


def test_01480():
    with pytest.raises(NotImplementedError):
        routine(test=1480)


# def test_01481():  # FIXME
#     assert routine(test=1481)


def test_01482():
    with pytest.raises(NotImplementedError):
        routine(test=1482)


def test_01483():
    with pytest.raises(NotImplementedError):
        routine(test=1483)


def test_01484():
    with pytest.raises(NotImplementedError):
        routine(test=1484)


def test_01485():
    with pytest.raises(TypeError):
        routine(test=1485)


# def test_01486():  # FIXME
#     assert routine(test=1486)


def test_01487():
    with pytest.raises(TypeError):
        routine(test=1487)


def test_01488():
    with pytest.raises(ZeroDivisionError):
        routine(test=1488)


def test_01489():
    with pytest.raises(SyntaxError):
        routine(test=1489)


def test_01490():
    with pytest.raises(SyntaxError):
        routine(test=1490)


def test_01491():
    with pytest.raises(SyntaxError):
        routine(test=1491)


def test_01492():
    with pytest.raises(NotImplementedError):
        routine(test=1492)


def test_01493():
    with pytest.raises(NotImplementedError):
        routine(test=1493)


def test_01494():
    with pytest.raises(NotImplementedError):
        routine(test=1494)


def test_01495():
    with pytest.raises(NameError):
        routine(test=1495)


def test_01496():
    assert routine(test=1496)


def test_01497():
    with pytest.raises(NotImplementedError):
        routine(test=1497)


def test_01498():
    with pytest.raises(NotImplementedError):
        routine(test=1498)


def test_01499():
    with pytest.raises(NotImplementedError):
        routine(test=1499)


def test_01500():
    with pytest.raises(NotImplementedError):
        routine(test=1500)


def test_01501():
    with pytest.raises(NotImplementedError):
        routine(test=1501)


def test_01502():
    with pytest.raises(NotImplementedError):
        routine(test=1502)


def test_01503():
    with pytest.raises(SyntaxError):
        routine(test=1503)


def test_01504():
    with pytest.raises(NotImplementedError):
        routine(test=1504)


def test_01505():
    with pytest.raises(NotImplementedError):
        routine(test=1505)


def test_01506():
    with pytest.raises(NotImplementedError):
        routine(test=1506)


def test_01507():
    with pytest.raises(NotImplementedError):
        routine(test=1507)


def test_01508():
    with pytest.raises(NotImplementedError):
        routine(test=1508)


def test_01509():
    with pytest.raises(NotImplementedError):
        routine(test=1509)


def test_01510():
    with pytest.raises(NotImplementedError):
        routine(test=1510)


def test_01511():
    with pytest.raises(NotImplementedError):
        routine(test=1511)


def test_01512():
    with pytest.raises(NotImplementedError):
        routine(test=1512)


def test_01513():
    with pytest.raises(NotImplementedError):
        routine(test=1513)


def test_01514():
    with pytest.raises(NotImplementedError):
        routine(test=1514)


def test_01515():
    with pytest.raises(NotImplementedError):
        routine(test=1515)


def test_01516():
    with pytest.raises(NotImplementedError):
        routine(test=1516)


# def test_01517():  # FIXME
#     assert routine(test=1517)


def test_01518():
    with pytest.raises(NotImplementedError):
        routine(test=1518)


def test_01519():
    with pytest.raises(NotImplementedError):
        routine(test=1519)


def test_01520():
    with pytest.raises(NotImplementedError):
        routine(test=1520)


def test_01521():
    with pytest.raises(NotImplementedError):
        routine(test=1521)


def test_01522():
    with pytest.raises(NotImplementedError):
        routine(test=1522)


def test_01523():
    with pytest.raises(NotImplementedError):
        routine(test=1523)


def test_01524():
    with pytest.raises(NotImplementedError):
        routine(test=1524)


def test_01525():
    with pytest.raises(NotImplementedError):
        routine(test=1525)


def test_01526():
    with pytest.raises(NotImplementedError):
        routine(test=1526)


def test_01527():
    with pytest.raises(NotImplementedError):
        routine(test=1527)


def test_01528():
    with pytest.raises(NotImplementedError):
        routine(test=1528)


def test_01529():
    with pytest.raises(NotImplementedError):
        routine(test=1529)


def test_01530():
    with pytest.raises(NotImplementedError):
        routine(test=1530)


def test_01531():
    with pytest.raises(NotImplementedError):
        routine(test=1531)


def test_01532():
    with pytest.raises(NotImplementedError):
        routine(test=1532)


def test_01533():
    with pytest.raises(NotImplementedError):
        routine(test=1533)


def test_01534():
    with pytest.raises(NotImplementedError):
        routine(test=1534)


def test_01535():
    with pytest.raises(NotImplementedError):
        routine(test=1535)


def test_01536():
    with pytest.raises(NotImplementedError):
        routine(test=1536)


def test_01537():
    with pytest.raises(NotImplementedError):
        routine(test=1537)


def test_01538():
    with pytest.raises(NotImplementedError):
        routine(test=1538)


# def test_01539():  # FIXME
#     assert routine(test=1539)


def test_01540():
    with pytest.raises(NotImplementedError):
        routine(test=1540)


def test_01541():
    with pytest.raises(NotImplementedError):
        routine(test=1541)


def test_01542():
    with pytest.raises(NotImplementedError):
        routine(test=1542)


def test_01543():
    with pytest.raises(NotImplementedError):
        routine(test=1543)


# def test_01544():  # FIXME
#     assert routine(test=1544)

# def test_01545():  # FIXME
#     assert routine(test=1545)

# def test_01546():  # FIXME
#     assert routine(test=1546)

# def test_01547():  # FIXME
#     assert routine(test=1547)

# def test_01548():  # FIXME
#     assert routine(test=1548)

# def test_01549():  # FIXME
#     assert routine(test=1549)

# def test_01550():  # FIXME
#     assert routine(test=1550)

# def test_01551():  # FIXME
#     assert routine(test=1551)


def test_01552():
    assert routine(test=1552)


def test_01553():
    assert routine(test=1553)


def test_01554():
    assert routine(test=1554)


def test_01555():
    assert routine(test=1555)


def test_01556():
    assert routine(test=1556)


def test_01557():
    assert routine(test=1557)


# def test_01558():  # FIXME
#     assert routine(test=1558)

# def test_01559():  # FIXME
#     assert routine(test=1559)

# def test_01560():  # FIXME
#     assert routine(test=1560)


def test_01561():
    with pytest.raises(NotImplementedError):
        routine(test=1561)


# def test_01562():  # FIXME
#     assert routine(test=1562)


def test_01563():
    with pytest.raises(NotImplementedError):
        routine(test=1563)


def test_01564():
    with pytest.raises(SyntaxError):
        routine(test=1564)


# def test_01565():  # FIXME
#     assert routine(test=1565)


# def test_01566():  # FIXME
#     with pytest.raises(KeyError):
#         routine(test=1566)


# def test_01567():  # FIXME
#     assert routine(test=1567)

# def test_01568():  # FIXME
#     assert routine(test=1568)

# def test_01569():  # FIXME
#     assert routine(test=1569)

# def test_01570():  # FIXME
#     assert routine(test=1570)

# def test_01571():  # FIXME
#     assert routine(test=1571)

# def test_01572():  # FIXME
#     assert routine(test=1572)

# def test_01573():  # FIXME
#     assert routine(test=1573)


def test_01574():
    assert routine(test=1574)


def test_01575():
    with pytest.raises(NotImplementedError):
        routine(test=1575)


def test_01576():
    with pytest.raises(NotImplementedError):
        routine(test=1576)


def test_01577():
    with pytest.raises(NotImplementedError):
        routine(test=1577)


def test_01578():
    with pytest.raises(NotImplementedError):
        routine(test=1578)


def test_01579():
    with pytest.raises(NotImplementedError):
        routine(test=1579)


def test_01580():
    with pytest.raises(NotImplementedError):
        routine(test=1580)


# def test_01581():  # FIXME
#     assert routine(test=1581)


def test_01582():
    with pytest.raises(NotImplementedError):
        routine(test=1582)


def test_01583():
    with pytest.raises(NotImplementedError):
        routine(test=1583)


def test_01584():
    with pytest.raises(NotImplementedError):
        routine(test=1584)


# def test_01585():  # FIXME
#     assert routine(test=1585)


def test_01586():
    with pytest.raises(NotImplementedError):
        routine(test=1586)


# def test_01587():  # FIXME
#     assert routine(test=1587)


def test_01588():
    with pytest.raises(NotImplementedError):
        routine(test=1588)


def test_01589():
    with pytest.raises(NotImplementedError):
        routine(test=1589)


def test_01590():
    with pytest.raises(NotImplementedError):
        routine(test=1590)


def test_01591():
    with pytest.raises(NotImplementedError):
        routine(test=1591)


def test_01592():
    with pytest.raises(NotImplementedError):
        routine(test=1592)


def test_01593():
    with pytest.raises(NotImplementedError):
        routine(test=1593)


def test_01594():
    with pytest.raises(NotImplementedError):
        routine(test=1594)


def test_01595():
    with pytest.raises(NotImplementedError):
        routine(test=1595)


def test_01596():
    with pytest.raises(NotImplementedError):
        routine(test=1596)


def test_01597():
    with pytest.raises(NotImplementedError):
        routine(test=1597)


def test_01598():
    with pytest.raises(NotImplementedError):
        routine(test=1598)


def test_01599():
    with pytest.raises(NotImplementedError):
        routine(test=1599)


def test_01600():
    with pytest.raises(NotImplementedError):
        routine(test=1600)


def test_01601():
    with pytest.raises(NotImplementedError):
        routine(test=1601)


def test_01602():
    with pytest.raises(NotImplementedError):
        routine(test=1602)


def test_01603():
    with pytest.raises(NotImplementedError):
        routine(test=1603)


def test_01604():
    with pytest.raises(NotImplementedError):
        routine(test=1604)


def test_01605():
    with pytest.raises(NotImplementedError):
        routine(test=1605)


# def test_01606():  # FIXME
#     assert routine(test=1606)

# def test_01607():  # FIXME
#     assert routine(test=1607)

# def test_01608():  # FIXME
#     assert routine(test=1608)

# def test_01609():  # FIXME
#     assert routine(test=1609)

# def test_01610():  # FIXME
#     assert routine(test=1610)

# def test_01611():  # FIXME
#     assert routine(test=1611)

# def test_01612():  # FIXME
#     assert routine(test=1612)

# def test_01613():  # FIXME
#     assert routine(test=1613)

# def test_01614():  # FIXME
#     assert routine(test=1614)

# def test_01615():  # FIXME
#     assert routine(test=1615)

# def test_01616():  # FIXME
#     assert routine(test=1616)

# def test_01617():  # FIXME
#     assert routine(test=1617)

# def test_01618():  # FIXME
#     assert routine(test=1618)

# def test_01619():  # FIXME
#     assert routine(test=1619)

# def test_01620():  # FIXME
#     assert routine(test=1620)

# def test_01621():  # FIXME
#     assert routine(test=1621)

# def test_01622():  # FIXME
#     assert routine(test=1622)

# def test_01623():  # FIXME
#     assert routine(test=1623)

# def test_01624():  # FIXME
#     assert routine(test=1624)

# def test_01625():  # FIXME
#     assert routine(test=1625)


def test_01626():
    with pytest.raises(NotImplementedError):
        routine(test=1626)


def test_01627():
    with pytest.raises(NotImplementedError):
        routine(test=1627)


# def test_01628():  # FIXME
#     assert routine(test=1628)

# def test_01629():  # FIXME
#     assert routine(test=1629)

# def test_01630():  # FIXME
#     assert routine(test=1630)


def test_01631():
    with pytest.raises(NotImplementedError):
        routine(test=1631)


# def test_01632():  # FIXME
#     assert routine(test=1632)


def test_01633():
    with pytest.raises(NotImplementedError):
        routine(test=1633)


# def test_01634():  # FIXME
#     assert routine(test=1634)


def test_01635():
    with pytest.raises(NotImplementedError):
        routine(test=1635)


# def test_01636():  # FIXME
#     assert routine(test=1636)

# def test_01637():  # FIXME
#     assert routine(test=1637)

# def test_01638():  # FIXME
#     assert routine(test=1638)

# def test_01639():  # FIXME
#     assert routine(test=1639)

# def test_01640():  # FIXME
#     assert routine(test=1640)


# def test_01641():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1641)


def test_01642():
    with pytest.raises(NotImplementedError):
        routine(test=1642)


def test_01643():
    with pytest.raises(NotImplementedError):
        routine(test=1643)


def test_01644():
    with pytest.raises(NotImplementedError):
        routine(test=1644)


def test_01645():
    with pytest.raises(NotImplementedError):
        routine(test=1645)


def test_01646():
    with pytest.raises(NotImplementedError):
        routine(test=1646)


def test_01647():
    with pytest.raises(NotImplementedError):
        routine(test=1647)


def test_01648():
    with pytest.raises(NotImplementedError):
        routine(test=1648)


def test_01649():
    with pytest.raises(NotImplementedError):
        routine(test=1649)


def test_01650():
    with pytest.raises(NotImplementedError):
        routine(test=1650)


def test_01651():
    with pytest.raises(NotImplementedError):
        routine(test=1651)


def test_01652():
    with pytest.raises(NotImplementedError):
        routine(test=1652)


def test_01653():
    with pytest.raises(NotImplementedError):
        routine(test=1653)


# def test_01654():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1654)


# def test_01655():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1655)


# def test_01656():  # FIXME
#     assert routine(test=1656)


# def test_01657():  # FIXME
#     with pytest.raises(NotImplementedError):
#         routine(test=1657)


def test_01658():
    with pytest.raises(NotImplementedError):
        routine(test=1658)


def test_01659():
    with pytest.raises(NotImplementedError):
        routine(test=1659)


def test_01660():
    with pytest.raises(NotImplementedError):
        routine(test=1660)


def test_01661():
    with pytest.raises(NotImplementedError):
        routine(test=1661)


def test_01662():
    with pytest.raises(NotImplementedError):
        routine(test=1662)


def test_01663():
    with pytest.raises(NotImplementedError):
        routine(test=1663)


def test_01664():
    with pytest.raises(NotImplementedError):
        routine(test=1664)


def test_01665():
    with pytest.raises(NotImplementedError):
        routine(test=1665)


def test_01666():
    with pytest.raises(NotImplementedError):
        routine(test=1666)


def test_01667():
    with pytest.raises(NotImplementedError):
        routine(test=1667)


def test_01668():
    with pytest.raises(NotImplementedError):
        routine(test=1668)


def test_01669():
    with pytest.raises(NotImplementedError):
        routine(test=1669)


def test_01670():
    with pytest.raises(NotImplementedError):
        routine(test=1670)


def test_01671():
    with pytest.raises(NotImplementedError):
        routine(test=1671)


def test_01672():
    with pytest.raises(NotImplementedError):
        routine(test=1672)


def test_01673():
    with pytest.raises(NotImplementedError):
        routine(test=1673)


def test_01674():
    with pytest.raises(NotImplementedError):
        routine(test=1674)


def test_01675():
    with pytest.raises(NotImplementedError):
        routine(test=1675)


def test_01676():
    with pytest.raises(NotImplementedError):
        routine(test=1676)


def test_01677():
    with pytest.raises(NotImplementedError):
        routine(test=1677)


def test_01678():
    with pytest.raises(NotImplementedError):
        routine(test=1678)


def test_01679():
    with pytest.raises(NotImplementedError):
        routine(test=1679)


def test_01680():
    with pytest.raises(NotImplementedError):
        routine(test=1680)


def test_01681():
    with pytest.raises(NotImplementedError):
        routine(test=1681)


def test_01682():
    with pytest.raises(NotImplementedError):
        routine(test=1682)


def test_01683():
    with pytest.raises(NotImplementedError):
        routine(test=1683)


def test_01684():
    with pytest.raises(NotImplementedError):
        routine(test=1684)


def test_01685():
    with pytest.raises(NotImplementedError):
        routine(test=1685)


def test_01686():
    with pytest.raises(NotImplementedError):
        routine(test=1686)


def test_01687():
    with pytest.raises(NotImplementedError):
        routine(test=1687)


def test_01688():
    with pytest.raises(NotImplementedError):
        routine(test=1688)


def test_01689():
    with pytest.raises(NotImplementedError):
        routine(test=1689)


def test_01690():
    with pytest.raises(NotImplementedError):
        routine(test=1690)


def test_01691():
    with pytest.raises(NotImplementedError):
        routine(test=1691)


def test_01692():
    with pytest.raises(NotImplementedError):
        routine(test=1692)


def test_01693():
    with pytest.raises(NotImplementedError):
        routine(test=1693)


def test_01694():
    with pytest.raises(NotImplementedError):
        routine(test=1694)


def test_01695():
    with pytest.raises(NotImplementedError):
        routine(test=1695)


def test_01696():
    with pytest.raises(NotImplementedError):
        routine(test=1696)


def test_01697():
    with pytest.raises(NotImplementedError):
        routine(test=1697)


def test_01698():
    with pytest.raises(NotImplementedError):
        routine(test=1698)


def test_01699():
    with pytest.raises(NotImplementedError):
        routine(test=1699)


def test_01700():
    with pytest.raises(NotImplementedError):
        routine(test=1700)


def test_01701():
    with pytest.raises(NotImplementedError):
        routine(test=1701)


def test_01702():
    with pytest.raises(NotImplementedError):
        routine(test=1702)


def test_01703():
    with pytest.raises(NotImplementedError):
        routine(test=1703)


def test_01704():
    with pytest.raises(NotImplementedError):
        routine(test=1704)


def test_01705():
    with pytest.raises(NotImplementedError):
        routine(test=1705)


def test_01706():
    with pytest.raises(NotImplementedError):
        routine(test=1706)


def test_01707():
    with pytest.raises(NotImplementedError):
        routine(test=1707)


def test_01708():
    with pytest.raises(NotImplementedError):
        routine(test=1708)


def test_01709():
    with pytest.raises(NotImplementedError):
        routine(test=1709)


def test_01710():
    with pytest.raises(NotImplementedError):
        routine(test=1710)


def test_01711():
    with pytest.raises(NotImplementedError):
        routine(test=1711)


def test_01712():
    with pytest.raises(NotImplementedError):
        routine(test=1712)


def test_01713():
    with pytest.raises(NotImplementedError):
        routine(test=1713)


def test_01714():
    with pytest.raises(NotImplementedError):
        routine(test=1714)


def test_01715():
    with pytest.raises(NotImplementedError):
        routine(test=1715)


def test_01716():
    with pytest.raises(NotImplementedError):
        routine(test=1716)


def test_01717():
    with pytest.raises(NotImplementedError):
        routine(test=1717)


def test_01718():
    with pytest.raises(NotImplementedError):
        routine(test=1718)


def test_01719():
    with pytest.raises(NotImplementedError):
        routine(test=1719)


def test_01720():
    with pytest.raises(NotImplementedError):
        routine(test=1720)


def test_01721():
    with pytest.raises(NotImplementedError):
        routine(test=1721)


def test_01722():
    with pytest.raises(NotImplementedError):
        routine(test=1722)


def test_01723():
    with pytest.raises(NotImplementedError):
        routine(test=1723)


def test_01724():
    with pytest.raises(NotImplementedError):
        routine(test=1724)


def test_01725():
    with pytest.raises(NotImplementedError):
        routine(test=1725)


def test_01726():
    with pytest.raises(NotImplementedError):
        routine(test=1726)


def test_01727():
    with pytest.raises(NotImplementedError):
        routine(test=1727)


def test_01728():
    with pytest.raises(NotImplementedError):
        routine(test=1728)


def test_01729():
    with pytest.raises(NotImplementedError):
        routine(test=1729)


def test_01730():
    with pytest.raises(NotImplementedError):
        routine(test=1730)


def test_01731():
    with pytest.raises(NotImplementedError):
        routine(test=1731)


def test_01732():
    with pytest.raises(NotImplementedError):
        routine(test=1732)


def test_01733():
    with pytest.raises(NotImplementedError):
        routine(test=1733)


def test_01734():
    with pytest.raises(NotImplementedError):
        routine(test=1734)


def test_01735():
    with pytest.raises(NotImplementedError):
        routine(test=1735)


def test_01736():
    with pytest.raises(NotImplementedError):
        routine(test=1736)


def test_01737():
    with pytest.raises(NotImplementedError):
        routine(test=1737)


def test_01738():
    with pytest.raises(NotImplementedError):
        routine(test=1738)


def test_01739():
    with pytest.raises(NotImplementedError):
        routine(test=1739)


def test_01740():
    with pytest.raises(NotImplementedError):
        routine(test=1740)


def test_01741():
    with pytest.raises(NotImplementedError):
        routine(test=1741)


def test_01742():
    with pytest.raises(NotImplementedError):
        routine(test=1742)


# def test_01743():  # FIXME
#     assert routine(test=1743)

# def test_01744():  # FIXME
#     assert routine(test=1744)

# def test_01745():  # FIXME
#     assert routine(test=1745)


def test_01746():
    with pytest.raises(NotImplementedError):
        routine(test=1746)


# def test_01747():  # FIXME
#     assert routine(test=1747)

# def test_01748():  # FIXME
#     assert routine(test=1748)

# def test_01749():  # FIXME
#     assert routine(test=1749)

# def test_01750():  # FIXME
#     assert routine(test=1750)

# def test_01751():  # FIXME
#     assert routine(test=1751)

# def test_01752():  # FIXME
#     assert routine(test=1752)

# def test_01753():  # FIXME
#     assert routine(test=1753)


def test_01754():
    with pytest.raises(NotImplementedError):
        routine(test=1754)


def test_01755():
    with pytest.raises(NotImplementedError):
        routine(test=1755)


def test_01756():
    with pytest.raises(NotImplementedError):
        routine(test=1756)


def test_01757():
    with pytest.raises(NotImplementedError):
        routine(test=1757)


def test_01758():
    with pytest.raises(NotImplementedError):
        routine(test=1758)


def test_01759():
    with pytest.raises(NotImplementedError):
        routine(test=1759)


def test_01760():
    assert routine(test=1760)


def test_01761():
    assert routine(test=1761)


def test_01762():
    assert routine(test=1762)


def test_01763():
    assert routine(test=1763)


# def test_01764():  # FIXME
#     assert routine(test=1764)

# def test_01765():  # FIXME
#     assert routine(test=1765)

# def test_01766():  # FIXME
#     assert routine(test=1766)

# def test_01767():  # FIXME
#     assert routine(test=1767)


def test_01768():
    with pytest.raises(NotImplementedError):
        routine(test=1768)


def test_01769():
    with pytest.raises(NotImplementedError):
        routine(test=1769)


def test_01770():
    with pytest.raises(NotImplementedError):
        routine(test=1770)


def test_01771():
    with pytest.raises(NotImplementedError):
        routine(test=1771)


def test_01772():
    with pytest.raises(NotImplementedError):
        routine(test=1772)


# def test_01773():  # FIXME
#     assert routine(test=1773)

# def test_01774():  # FIXME
#     assert routine(test=1774)


def test_01775():
    with pytest.raises(NotImplementedError):
        routine(test=1775)


def test_01776():
    with pytest.raises(NotImplementedError):
        routine(test=1776)


def test_01777():
    with pytest.raises(NotImplementedError):
        routine(test=1777)


def test_01778():
    with pytest.raises(NotImplementedError):
        routine(test=1778)


def test_01779():
    with pytest.raises(NotImplementedError):
        routine(test=1779)


def test_01780():
    with pytest.raises(NotImplementedError):
        routine(test=1780)
