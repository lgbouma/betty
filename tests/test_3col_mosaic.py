def test_3col_mosaic():

    from betty.plotting import given_N_axes_get_3col_mosaic_subplots

    assert (
        given_N_axes_get_3col_mosaic_subplots(6, return_axstr=1)
        == '\n¡¢£\n¤¥¦\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(4, return_axstr=1)
        == '\n¡¢£\n.¤.\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(5, return_axstr=1)
        == '\n¡¡¢¢££\n.¤¤¥¥.\n'
    )

    assert (
        given_N_axes_get_3col_mosaic_subplots(8, return_axstr=1)
        == '\n¡¡¢¢££\n¤¤¥¥¦¦\n.§§¨¨.\n'
    )

    # this is the test case that drove the choice of weird unicode characters.
    # instead of say, string versions of numbers.
    assert (
        given_N_axes_get_3col_mosaic_subplots(11, return_axstr=1)
        == '\n¡¡¢¢££\n¤¤¥¥¦¦\n§§¨¨©©\n.ªª««.\n'
    )


if __name__ == "__main__":
    test_3col_mosaic()
