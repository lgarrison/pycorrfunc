project(
    'pycorrfunc',
    'cpp',
    'c',
    version: '0.1.0',
    license: 'BSD',
    meson_version: '>= 1.1.0',
    default_options: [
        'cpp_std=c++17',
        # 'b_sanitize=address,undefined',
        'warning_level=2',
        'werror=true',  # TODO: move to CI
        # 'openmp=disabled',
    ],
)

py = import('python').find_installation(pure: false)
nanobind_dep = dependency('nanobind', static: true)
openmp_dep = dependency('openmp', language: 'c', required: get_option('openmp'))

if get_option('double_accum').enabled()
  add_project_arguments('-DPYCORRFUNC_USE_DOUBLEACCUM', language: 'c')
  add_project_arguments('-DPYCORRFUNC_USE_DOUBLEACCUM', language: 'cpp')
endif

simd = import('unstable-simd')

cc = meson.get_compiler('c')

inc = include_directories('include', 'lib/theory/DD', 'lib/common', 'tests')

simd_check_kwargs = {
  'sse42' : 'lib/theory/DD/kernel_sse42.c',
  'avx' : 'lib/theory/DD/kernel_avx.c',
  'compiler' : cc,
  'include_directories' : inc,
}

simd_cdata = configuration_data()
simdinfo_float = simd.check(
  'kernels_float',
  kwargs : simd_check_kwargs
)
simdlibs_float = simdinfo_float[0]
simd_cdata.merge_from(simdinfo_float[1])

simdinfo_double = simd.check(
  'kernels_double',
  kwargs : simd_check_kwargs,
  c_args : ['-DPYCORRFUNC_USE_DOUBLE'],
)
simdlibs_double = simdinfo_double[0]

# AVX512 not natively supported by meson's SIMD module,
# implement it manually
avx512_supported = cc.has_argument('-mavx512f') and cc.has_argument('-mavx512vl')
if avx512_supported
  avx512_sources = [
    'lib/theory/DD/kernel_avx512.c',
    'lib/common/avx512_calls.c',
  ]

  avx512lib_float = static_library(
    'kernel_avx512_float',
    sources : avx512_sources,
    include_directories : inc,
    c_args : ['-mavx512f', '-mavx512vl', '-DHAVE_AVX512'],
  )

  avx512lib_double = static_library(
    'kernel_avx512_double',
    sources : avx512_sources,
    include_directories : inc,
    c_args : ['-mavx512f', '-mavx512vl', '-DHAVE_AVX512', '-DPYCORRFUNC_USE_DOUBLE'],
  )

  simd_cdata.set('HAVE_AVX512', 1, description : 'Compiler supports avx512.')

  simdlibs_float += avx512lib_float
  simdlibs_double += avx512lib_double
endif


sources = [
  'lib/common/cellarray.c',
  'lib/common/gridlink.c',
  'lib/common/gridlink_utils.c',
  'lib/common/gridlink_utils.c',
  'lib/common/options.c',
  'lib/common/progressbar.c',
  'lib/common/utils.c',
  'lib/common/weights.c',
  'lib/cpp_utils.cpp',
  'lib/main.cpp',
  'lib/theory/DD/countpairs.c',
  'lib/theory/DD/kernel_fallback.c',
]

extension_kwargs = {
  'sources' : sources,
  'install' : true,
  'subdir' : 'pycorrfunc',
  'include_directories' : inc,
  'dependencies' : [nanobind_dep, openmp_dep],
}

pycorrfunc_double = py.extension_module(
  '_pycorrfunc',
  kwargs : extension_kwargs,
  link_with : simdlibs_double,
  c_args : ['-DPYCORRFUNC_USE_DOUBLE', '-DNB_DOMAIN=d'],
  cpp_args : ['-DPYCORRFUNC_USE_DOUBLE', '-DNB_DOMAIN=d'],
)

pycorrfunc_float = py.extension_module(
  '_pycorrfuncf',
  kwargs : extension_kwargs,
  link_with : simdlibs_float,
  c_args : ['-DNB_DOMAIN=f'],
  cpp_args : ['-DNB_DOMAIN=f'],
)

# N.B. simdconfig.h is precision independent, so we'll just use the cdata from float
configure_file(output : 'simdconfig.h', configuration : simd_cdata)

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
