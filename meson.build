project(
    'pycorrfunc',
    'cpp',
    'c',
    version: '0.1.0',
    license: 'BSD',
    meson_version: '>= 1.1.0',
    default_options: [
        'cpp_std=c++17',
        # 'b_sanitize=address,undefined',  # for debugging
        'warning_level=2',
        'werror=true',  # TODO: move to CI
    ],
)

py = import('python').find_installation(pure: false)
nanobind_dep = dependency('nanobind', static: true)

simd = import('unstable-simd')

cc = meson.get_compiler('c')

inc = include_directories('include', 'lib/theory/DD', 'lib/utils', 'tests')

simd_check_kwargs = {
  'sse42' : 'lib/theory/DD/kernels_sse42.c',
  'avx' : 'lib/theory/DD/kernels_avx.c',
  # 'avx512' : 'lib/theory/DD/kernels_avx512.c',
  'compiler' : cc,
  'include_directories' : inc,
}

cdata = configuration_data()
simdinfo_float = simd.check(
  'kernels_float',
  kwargs : simd_check_kwargs
)
simdlibs_float = simdinfo_float[0]
cdata.merge_from(simdinfo_float[1])

simdinfo_double = simd.check(
  'kernels_double',
  kwargs : simd_check_kwargs,
  c_args : ['-DCORRFUNC_USE_DOUBLE'],
)
simdlibs_double = simdinfo_double[0]

# avx512_supported = cc.has_argument('avx512f')
# avx512_source = []
# if avx512_supported
#   avx512_source += 'lib/theory/DD/kernels_avx512.c'
#   add_project_arguments('-DHAVE_AVX512F', language: 'c')
# endif

sources = [
  'lib/theory/DD/countpairs.c',
  'lib/utils/gridlink_utils.c',
  'lib/main.cpp',
  # 'lib/utils/avx512_calls.c',
  'lib/utils/gridlink_impl.c',
  'lib/utils/gridlink_mocks_impl.c',
  'lib/utils/gridlink_utils.c',
  'lib/utils/progressbar.c',
  'lib/utils/utils.c',
  'lib/theory/DD/kernels_fallback.c',
]

extension_kwargs = {
  'sources' : sources,
  'install' : true,
  'subdir' : 'pycorrfunc',
  'include_directories' : inc,
  'dependencies' : [nanobind_dep],
}

corrfunc_double = py.extension_module(
  '_corrfunc',
  kwargs : extension_kwargs,
  link_with : simdlibs_double,
  c_args : ['-DCORRFUNC_USE_DOUBLE'],
  cpp_args : ['-DCORRFUNC_USE_DOUBLE'],
)

corrfunc_float = py.extension_module(
  '_corrfuncf',
  kwargs : extension_kwargs,
  link_with : simdlibs_float,
)

# N.B. simdconfig. is precision independent, so we'll just use the cdata from float
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
