def test_3col_mosaic():

    from betty.plotting import given_N_axes_get_3col_mosaic_subplots

    assert (
        given_N_axes_get_3col_mosaic_subplots(6, return_axstr=1)
        == '\n012\n345\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(4, return_axstr=1)
        == '\n012\n.3.\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(5, return_axstr=1)
        == '\n001122\n.3344.\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(8, return_axstr=1)
        == '\n001122\n334455\n.6677.\n'
    )

if __name__ == "__main__":
    test_3col_mosaic()
