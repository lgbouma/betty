from os.path import join
from betty.paths import TESTDATADIR, TEXDATADIR, TESTRESULTSDIR
from betty.posterior_table import table_tex_to_pdf

tex_table_path = join(
    TESTDATADIR,
    'gaiatwo0000174888907925967232-0019-cam1-ccd3_tess_v01_llc_simpletransit_posteriortable.tex'
)

pdf_path = join(
    TESTRESULTSDIR,
    'gaiatwo0000174888907925967232-0019-cam1-ccd3_tess_v01_llc_simpletransit_posteriortable.pdf'
)

table_tex_to_pdf(tex_table_path, pdf_path)
