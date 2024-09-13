project(
    'pycorrfunc',
    'cpp',
    'c',
    version: '0.1.0',
    license: 'BSD',
    meson_version: '>= 1.1.0',
    default_options: [
        'cpp_std=c++11',
    ],
)

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')

simd = import('unstable-simd')

cc = meson.get_compiler('c')

cdata = configuration_data()

inc = include_directories('include', 'lib/theory/DD', 'lib/utils', 'tests')

rval = simd.check('mysimds',
  sse : 'lib/theory/DD/kernels_sse.c',
  # avx : 'lib/theory/DD/kernels_avx.c',
  # avx512 : 'lib/theory/DD/kernels_avx512.c',
  # fallback : 'lib/theory/DD/kernels_fallback.c',
  compiler : cc,
  include_directories : inc,
)

avx512_supported = cc.has_argument('avx512f')
fallback_source = files('lib/theory/DD/kernels_fallback.c')
avx512_source = []
if avx512_supported
  avx512_source += files('lib/theory/DD/kernels_avx512.c')
  add_project_arguments('-DHAVE_AVX512F', language: 'c')
endif

simdlibs = rval[0]
cdata.merge_from(rval[1])

sources = [
  'lib/theory/DD/countpairs.c',
  'lib/utils/gridlink_utils.c',
  'lib/main.cpp',
  'lib/utils/avx512_calls.c',
  'lib/utils/cpu_features.c',
  'lib/utils/gridlink_impl.c',
  'lib/utils/gridlink_mocks_impl.c',
  'lib/utils/gridlink_utils.c',
  'lib/utils/progressbar.c',
  'lib/utils/utils.c',
]

all_sources = sources + avx512_source + fallback_source

corrfunc_double = py.extension_module(
  '_corrfunc',
  all_sources,
  install : true,
  subdir: 'pycorrfunc',
  include_directories : inc,
  dependencies : [pybind11_dep],
  link_with : simdlibs,
  c_args : ['-DCORRFUNC_DOUBLE'],
  cpp_args : ['-DCORRFUNC_DOUBLE'],
)

corrfunc_float = py.extension_module(
  '_corrfuncf',
  all_sources,
  install : true,
  subdir: 'pycorrfunc',
  include_directories : inc,
  dependencies : [pybind11_dep], 
  link_with : simdlibs,
)

configure_file(output : 'simdconfig.h', configuration : cdata)

install_subdir(
  'src/pycorrfunc',
  install_dir: py.get_install_dir() / 'pycorrfunc',
  strip_directory: true,
)

# TODO: not working
# install_symlink(
#   'compile_commands.json',
#   install_dir: '..',
#   pointing_to: 'compile_commands.json',
# )
