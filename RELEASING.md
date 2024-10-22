# Releasing
- Update the version number in `meson.build`, e.g. `2.0.0` (see https://github.com/mesonbuild/meson-python/issues/159)
- Tag the version: `git tag -a v2.0.0 -m 'Releasing version 2.0.0'`
- Push the tag: `git push origin v2.0.0`
- Check that the tests are passing
- Check that the build and upload to PyPI succeeded
- Make a GitHub release from the tag
- Bump the version in `meson.build`, e.g. `2.0.1dev`
