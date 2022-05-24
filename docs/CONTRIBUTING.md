# Contributing to SLEAP

As our community grows it is important to adhere to a set of contribution guidelines. These guidelines may change as needed. Please feel free to propose changes to source code in a pull request! 

#### Table Of Contents

1) [Code of Conduct](#code-of-conduct)
2) [Contributing](#contributing)
    * [Issues](#issues)
    * [Discussions](#discussions)
    * [Pull Requests](#pull-requests)
3) [Style Guides](#style-guides)

## Code of Conduct

Everyone contributing to SLEAP is governed by the [SLEAP Code of Conduct](CODE_OF_CONDUCT.md). As such, all are expected to abide by this code. Please report any unacceptable behavior to `talmo@salk.edu`.

## Contributing

Github has made it easy to separate issues from discussions. Generally speaking, if something warrants a pull request (e.g. bug fixes, TODOs) it should be submitted as an issue. Conversely, general questions about usage and improvement ideas should go into discussions. These both act as staging areas before adding the items to the SLEAP Roadmap. Once added to the Roadmap, items can be given priority, status, assignee(s), etc. 

### Issues

* Check [open/closed issues](https://github.com/talmolab/sleap/issues), [ideas](https://github.com/talmolab/sleap/discussions/categories/ideas), and [the help page](https://github.com/talmolab/sleap/discussions/categories/help) to make sure issue doesn't already exist / has been solved.
* Create new issue using the [issue template](https://github.com/talmolab/sleap/blob/arlo/contributing_guide/.github/ISSUE_TEMPLATE/bug_report.md).

### Discussions

* This is a place to go to ask questions and propose new ideas.
* 3 categories: Help, Ideas, General
   * **Help** - Having trouble with software, user experience issue, etc.
   * **Ideas** - Enhancements, things that would make the user experience nicer but not necessarily a problem with the code.
   * **General** - If it doesn't fall into help/ideas it goes here as long as it isn't bug fix (issue).

### Pull Requests

1) Install source code [`develop` branch](https://sleap.ai/installation.html#conda-from-source) and follow instructions to create conda env, etc.
2) Create a fork from the `develop` branch.
   * Either work on the `develop` branch or create a new branch (recommended if tackling multiple issues at a time).
   * If creating a branch, use your name followed by a relevant keyword for your changes, eg: `git checkout -b john/some_issue`
3) Make some changes/additions to the source code that tackle the issue(s).
4) Write [tests](https://github.com/talmolab/sleap/tree/develop/tests).
   * Can either write tests before creating a draft PR, or submit draft PR (to get code coverage statistics via codecov) and then write tests to narrow down error prone lines.
   * The test(s) should go into relevant subtest folders to the proposed change(s).
   * Test(s) should aim to hit every point in the proposed change(s) - cover edge cases to best of your ability.
   * Try to hit code coverage points.
5) Add files, commit, and push to origin.
6) Create a draft PR on [Github](https://github.com/talmolab/sleap/pulls) (follow instructions in template).
   * Make sure the tests pass and code coverage is good.
   * If either the tests or code coverage fail, repeat steps 3-5.
7) Once the draft PR looks good, convert to a finalized PR (hit the `ready for review` button).
   * IMPORTANT: Only convert to a finalized PR when you believe your changes are ready to be merged.
   * Optionally assign a reviewer on the right of the screen - otherwise a member of the SLEAP developer team will self-assign themselves.
8) If the reviewer requests changes, repeat steps 3-5 and `Re-request review`.
9) Once the reviewer signs off they will squash + merge the PR into the `develop` branch.
   * New feautures will be available on the `main` branch when a new release of SLEAP is released.

## Style Guides

* **Lint** - [Black](https://black.readthedocs.io/en/stable/) version 21.6b0 (see [dev_requirements](https://github.com/talmolab/sleap/blob/develop/dev_requirements.txt) for any changes).
* **Code** - Generally follow [PEP8](https://peps.python.org/pep-0008/). Type hinting is encouraged.
* **Documentation** - Use [Google-style comments and docstrings]([https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)) to document code.

#### Thank you for contributing to SLEAP! 
:heart:
