def test_imports():

    try:
        import betty as be
        BETTY_DEP = True
    except:
        BETTY_DEP = False

    assert BETTY_DEP
