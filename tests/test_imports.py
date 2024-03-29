def test_imports():

    modules = ['betty', 'numpy', 'pymc3', 'corner', 'exoplanet',
               'astrobase', 'astropy', 'pandas']

    for m in modules:

        dep_worked = True

        try:
            exec(f"import {m}")
            dep_worked = True
        except Exception as e:
            print(e)
            dep_worked = False

        assert dep_worked

if __name__ == "__main__":
    test_imports()
