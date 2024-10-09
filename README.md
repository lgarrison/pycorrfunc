# pycorrfunc

[![Tests](https://github.com/lgarrison/pycorrfunc/actions/workflows/tests.yaml/badge.svg)](https://github.com/lgarrison/pycorrfunc/actions/workflows/tests.yaml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/lgarrison/pycorrfunc/main.svg)](https://results.pre-commit.ci/latest/github/lgarrison/pycorrfunc/main) [![Jenkins Tests](https://jenkins.flatironinstitute.org/buildStatus/icon?job=pycorrfunc%2Fmain&subject=Jenkins%20Tests)](https://jenkins.flatironinstitute.org/job/pycorrfunc/job/main/)

> [!WARNING]
> This repo is a proof-of-concept and is not intended for public use.

A pythonic rewrite of the software infrastructure surrounding Corrfunc. The same internals and performance, but improved packaging.

In particular, the primary goal is to get wheels working with true runtime SIMD dispatch.

Based on the summer project of Peter McMaster, advised by Lehman Garrison.

## License
pycorrfunc is [MIT licensed](LICENSE).

Sources from Corrfunc are used under Corrfunc's MIT license, which is reproduced here:

<details>
<summary>Corrfunc license</summary>

```
Copyright (C) 2014 Manodeep Sinha (manodeep@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```
</details>
