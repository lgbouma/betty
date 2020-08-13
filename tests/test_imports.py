def test_imports():

    modules = ['betty', 'numpy', 'pymc3', 'corner', 'exoplanet']

    for m in modules:

        dep_worked = True

        try:
            exec(f"import {m}")
            dep_worked = True
        except:
            dep_worked = False

        assert dep_worked
